package auth

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
)

const cswapTestModel = "claude-opus-5-5"

type cswapTestClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *cswapTestClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *cswapTestClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.now = c.now.Add(d)
	c.mu.Unlock()
}

// cswapUsageServer fakes GET /api/oauth/usage, answering per bearer token.
type cswapUsageServer struct {
	t      *testing.T
	mu     sync.Mutex
	bodies map[string]string
	status map[string]int
	retry  map[string]string
	calls  map[string]int
}

func newCswapUsageServer(t *testing.T) (*cswapUsageServer, *httptest.Server) {
	fake := &cswapUsageServer{t: t, bodies: map[string]string{}, status: map[string]int{}, retry: map[string]string{}, calls: map[string]int{}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/oauth/usage" || r.Header.Get("anthropic-beta") != "oauth-2025-04-20" {
			t.Errorf("unexpected usage request %s %s beta=%q", r.Method, r.URL.Path, r.Header.Get("anthropic-beta"))
		}
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		fake.mu.Lock()
		fake.calls[token]++
		status, body, retry := fake.status[token], fake.bodies[token], fake.retry[token]
		fake.mu.Unlock()
		if retry != "" {
			w.Header().Set("Retry-After", retry)
		}
		if status != 0 && status != http.StatusOK {
			w.WriteHeader(status)
			return
		}
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return fake, server
}

func (f *cswapUsageServer) set(token, body string) {
	f.mu.Lock()
	f.bodies[token] = body
	f.status[token] = http.StatusOK
	f.retry[token] = ""
	f.mu.Unlock()
}

func (f *cswapUsageServer) fail(token string, status int, retryAfter string) {
	f.mu.Lock()
	f.status[token] = status
	f.retry[token] = retryAfter
	f.mu.Unlock()
}

func (f *cswapUsageServer) callCount(token string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[token]
}

// cswapUsage renders a usage response in the shape cswap reads.
func cswapUsageBody(fiveHour float64, fiveReset time.Time, sevenDay float64, sevenReset time.Time) string {
	return fmt.Sprintf(`{"five_hour":{"utilization":%g,"resets_at":%q},"seven_day":{"utilization":%g,"resets_at":%q}}`,
		fiveHour, fiveReset.UTC().Format(time.RFC3339Nano), sevenDay, sevenReset.UTC().Format(time.RFC3339Nano))
}

func newCswapTestSelector(t *testing.T, server *httptest.Server, clock *cswapTestClock) *CswapPolicySelector {
	t.Helper()
	selector := NewCswapPolicySelector(CswapPolicyConfig{})
	selector.usageURL = server.URL + "/api/oauth/usage"
	selector.now = clock.Now
	selector.randFloat = func() float64 { return 0.5 }
	// Tests drive pollOnce themselves; keep the background poller from starting.
	selector.startOnce.Do(func() {})
	t.Cleanup(selector.Stop)
	return selector
}

func cswapTestAuth(id string, expires time.Time) *Auth {
	return &Auth{
		ID:       id,
		Provider: "claude",
		Status:   StatusActive,
		Metadata: map[string]any{
			"type":          "claude",
			"access_token":  "token-" + id,
			"refresh_token": "refresh-" + id,
			"expired":       expires.Format(time.RFC3339),
		},
	}
}

// pollNow fetches every account once, ignoring the poll schedule, so policy
// tests do not depend on the cadence (TestCswapPolicyUsageBackoff covers it).
func pollNow(selector *CswapPolicySelector, auths []*Auth) {
	selector.mu.Lock()
	for _, acct := range selector.accounts {
		acct.nextPollAt = time.Time{}
	}
	selector.mu.Unlock()
	for range auths {
		selector.pollOnce(auths)
	}
}

func pickID(t *testing.T, selector Selector, opts cliproxyexecutor.Options, auths []*Auth) string {
	t.Helper()
	picked, errPick := selector.Pick(context.Background(), "claude", cswapTestModel, opts, auths)
	if errPick != nil {
		t.Fatalf("Pick: %v", errPick)
	}
	return picked.ID
}

func TestCswapPolicyPrefersSoonestSevenDayReset(t *testing.T) {
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	selector := newCswapTestSelector(t, server, clock)
	now := clock.Now()
	auths := []*Auth{cswapTestAuth("a", now.Add(8*time.Hour)), cswapTestAuth("b", now.Add(8*time.Hour)), cswapTestAuth("c", now.Add(8*time.Hour))}
	// a has the most headroom and sorts first; b resets soonest.
	fake.set("token-a", cswapUsageBody(5, now.Add(3*time.Hour), 10, now.Add(5*24*time.Hour)))
	fake.set("token-b", cswapUsageBody(40, now.Add(3*time.Hour), 50, now.Add(2*24*time.Hour)))
	fake.set("token-c", cswapUsageBody(20, now.Add(3*time.Hour), 30, now.Add(3*24*time.Hour)))
	pollNow(selector, auths)

	for i := 0; i < 3; i++ {
		if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "b" {
			t.Fatalf("pick %d = %q, want b (soonest 7-day reset)", i, got)
		}
	}

	// A sooner reset appears on c; consume-first moves there. The initial
	// assignment is not a switch, so no cooldown applies yet.
	clock.Advance(time.Minute)
	now = clock.Now()
	// a now also resets before b and has more headroom, but c resets first.
	fake.set("token-a", cswapUsageBody(5, now.Add(3*time.Hour), 10, now.Add(36*time.Hour)))
	fake.set("token-c", cswapUsageBody(20, now.Add(3*time.Hour), 30, now.Add(24*time.Hour)))
	pollNow(selector, auths)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "c" {
		t.Fatalf("after c resets sooner: pick = %q, want c", got)
	}

	// Inside the 300 s cooldown an even sooner account does not move the
	// preferred account; after it, consume-first does.
	clock.Advance(time.Minute)
	now = clock.Now()
	fake.set("token-a", cswapUsageBody(5, now.Add(3*time.Hour), 10, now.Add(12*time.Hour)))
	pollNow(selector, auths)
	clock.Advance(CswapDefaultCooldown - 2*time.Minute)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "c" {
		t.Fatalf("inside cooldown: pick = %q, want c", got)
	}
	clock.Advance(2 * time.Minute)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "a" {
		t.Fatalf("after cooldown: pick = %q, want a", got)
	}
}

func TestCswapPolicyThresholdMovesOutAndHysteresisBringsBack(t *testing.T) {
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	selector := newCswapTestSelector(t, server, clock)
	now := clock.Now()
	auths := []*Auth{cswapTestAuth("a", now.Add(8*time.Hour)), cswapTestAuth("b", now.Add(8*time.Hour))}
	sevenA := now.Add(24 * time.Hour)
	sevenB := now.Add(4 * 24 * time.Hour)
	// a resets sooner and its 7-day window sits at 90%, inside the hysteresis
	// band but never at the threshold.
	fake.set("token-a", cswapUsageBody(50, now.Add(2*time.Hour), 90, sevenA))
	fake.set("token-b", cswapUsageBody(10, now.Add(2*time.Hour), 20, sevenB))
	pollNow(selector, auths)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "a" {
		t.Fatalf("initial pick = %q, want a", got)
	}

	steps := []struct {
		fiveHour float64
		eligible string
		want     string
	}{
		{94.9, "a,b", "a"}, // below the threshold
		{95, "b", "b"},     // at the threshold: out of rotation, moved without waiting for a cooldown
		{90, "b", "b"},     // still inside the hysteresis band
		{85, "b", "b"},     // threshold - hysteresis is not yet below it
		{84.9, "a,b", "b"}, // back in rotation, but b holds for the cooldown after the switch
	}
	for _, step := range steps {
		clock.Advance(time.Minute)
		fake.set("token-a", cswapUsageBody(step.fiveHour, clock.Now().Add(2*time.Hour), 90, sevenA))
		pollNow(selector, auths)
		if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != step.want {
			t.Fatalf("a 5h at %.1f%%: pick = %q, want %q", step.fiveHour, got, step.want)
		}
		if got := strings.Join(authIDs(selector.filterSessionEligible("claude", cswapTestModel, auths)), ","); got != step.eligible {
			t.Fatalf("a 5h at %.1f%%: eligible = %s, want %s", step.fiveHour, got, step.eligible)
		}
	}
	// a is back in rotation although its binding window (7d at 90%) is still inside
	// the band: hysteresis applies to the window that tripped. Once the cooldown
	// has passed, consume-first returns to a.
	clock.Advance(CswapDefaultCooldown - 3*time.Minute)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "a" {
		t.Fatalf("after recovery and cooldown: pick = %q, want a", got)
	}
}

func TestCswapPolicyCooldownHoldsProactiveSwitch(t *testing.T) {
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	selector := newCswapTestSelector(t, server, clock)
	now := clock.Now()
	auths := []*Auth{cswapTestAuth("a", now.Add(8*time.Hour)), cswapTestAuth("b", now.Add(8*time.Hour)), cswapTestAuth("c", now.Add(8*time.Hour))}
	fiveReset := now.Add(3 * time.Hour)
	fake.set("token-a", cswapUsageBody(50, fiveReset, 10, now.Add(24*time.Hour)))
	fake.set("token-b", cswapUsageBody(50, fiveReset, 10, now.Add(2*24*time.Hour)))
	fake.set("token-c", cswapUsageBody(10, fiveReset, 10, now.Add(3*24*time.Hour)))
	pollNow(selector, auths)
	pickID(t, selector, cliproxyexecutor.Options{}, auths)
	fake.set("token-a", cswapUsageBody(99, fiveReset, 10, now.Add(24*time.Hour)))
	pollNow(selector, auths)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "b" {
		t.Fatalf("after a left rotation: pick = %q, want b", got)
	}

	// b reaches the threshold right after the switch. Like cswap's proactive
	// trigger, the move waits for the cooldown and b keeps its sessions.
	clock.Advance(time.Minute)
	fake.set("token-b", cswapUsageBody(96, fiveReset, 10, now.Add(2*24*time.Hour)))
	pollNow(selector, auths)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "b" {
		t.Fatalf("b over threshold inside cooldown: pick = %q, want b", got)
	}
	if got := strings.Join(authIDs(selector.filterSessionEligible("claude", cswapTestModel, auths)), ","); got != "b,c" {
		t.Fatalf("inside cooldown: eligible = %s, want b,c", got)
	}
	clock.Advance(CswapDefaultCooldown)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "c" {
		t.Fatalf("after cooldown: pick = %q, want c", got)
	}
	if got := strings.Join(authIDs(selector.filterSessionEligible("claude", cswapTestModel, auths)), ","); got != "c" {
		t.Fatalf("after cooldown: eligible = %s, want c", got)
	}

	// At its hard limit the preferred account escapes at once (cswap at-limit).
	fake.set("token-a", cswapUsageBody(10, fiveReset, 10, now.Add(24*time.Hour)))
	pollNow(selector, auths)
	clock.Advance(time.Minute)
	fake.set("token-c", cswapUsageBody(100, fiveReset, 10, now.Add(3*24*time.Hour)))
	pollNow(selector, auths)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "a" {
		t.Fatalf("c at 100%% inside cooldown: pick = %q, want a", got)
	}
}

func TestCswapPolicyEveryoneAboveThreshold(t *testing.T) {
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	selector := newCswapTestSelector(t, server, clock)
	now := clock.Now()
	auths := []*Auth{
		cswapTestAuth("a", now.Add(8*time.Hour)),
		cswapTestAuth("b", now.Add(8*time.Hour)),
		cswapTestAuth("c", now.Add(8*time.Hour)),
		cswapTestAuth("d", now.Add(8*time.Hour)),
	}
	// Everyone is at or over 95%. a resets its 7-day window first, b's 5-hour
	// window comes back first, c is bound by its 7-day window for days, d is at
	// its hard limit.
	fake.set("token-a", cswapUsageBody(97, now.Add(2*time.Hour), 60, now.Add(24*time.Hour)))
	fake.set("token-b", cswapUsageBody(96, now.Add(30*time.Minute), 50, now.Add(4*24*time.Hour)))
	fake.set("token-c", cswapUsageBody(20, now.Add(2*time.Hour), 99, now.Add(3*24*time.Hour)))
	fake.set("token-d", cswapUsageBody(100, now.Add(20*time.Minute), 70, now.Add(12*time.Hour)))
	pollNow(selector, auths)

	// Accounts with headroom stay usable for bound sessions; d at 100% does not.
	eligible := selector.filterSessionEligible("claude", cswapTestModel, auths)
	if ids := authIDs(eligible); strings.Join(ids, ",") != "a,b,c" {
		t.Fatalf("eligible = %v, want [a b c]", ids)
	}
	// The first preference escapes by soonest 7-day reset (cswap's at-limit order)...
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "a" {
		t.Fatalf("initial pick = %q, want a", got)
	}
	// ...then the all-above ranking moves to the account whose binding window
	// returns first, at least RECOVERY_HYSTERESIS_S sooner.
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "b" {
		t.Fatalf("all-above pick = %q, want b (5h back in 30m)", got)
	}
	// From b nothing comes back 300 s sooner, so b holds after the cooldown too.
	clock.Advance(CswapDefaultCooldown + time.Second)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "b" {
		t.Fatalf("after cooldown: pick = %q, want b", got)
	}
}

func TestCswapPolicyRecoveryHysteresisBlocksNearTies(t *testing.T) {
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	selector := newCswapTestSelector(t, server, clock)
	now := clock.Now()
	auths := []*Auth{cswapTestAuth("a", now.Add(8*time.Hour)), cswapTestAuth("b", now.Add(8*time.Hour))}
	// b comes back four minutes before a: inside the 300 s recovery hysteresis.
	fake.set("token-a", cswapUsageBody(97, now.Add(time.Hour), 60, now.Add(24*time.Hour)))
	fake.set("token-b", cswapUsageBody(97, now.Add(56*time.Minute), 50, now.Add(4*24*time.Hour)))
	pollNow(selector, auths)
	for i := 0; i < 2; i++ {
		if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "a" {
			t.Fatalf("pick %d = %q, want a to hold", i, got)
		}
	}
}

func TestCswapPolicySkipsExpiredTokens(t *testing.T) {
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	selector := newCswapTestSelector(t, server, clock)
	now := clock.Now()
	a := cswapTestAuth("a", now.Add(8*time.Hour))
	b := cswapTestAuth("b", now.Add(8*time.Hour))
	auths := []*Auth{a, b}
	fake.set("token-a", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(24*time.Hour)))
	fake.set("token-b", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(4*24*time.Hour)))
	pollNow(selector, auths)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "a" {
		t.Fatalf("initial pick = %q, want a", got)
	}

	a.Metadata["expired"] = now.Add(-time.Minute).Format(time.RFC3339)
	callsBefore := fake.callCount("token-a")
	for i := 0; i < 3; i++ {
		clock.Advance(time.Minute)
		pollNow(selector, auths)
		if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "b" {
			t.Fatalf("with a expired: pick = %q, want b", got)
		}
		if eligible := authIDs(selector.filterSessionEligible("claude", cswapTestModel, auths)); strings.Join(eligible, ",") != "b" {
			t.Fatalf("with a expired: eligible = %v, want [b]", eligible)
		}
	}
	if calls := fake.callCount("token-a"); calls != callsBefore {
		t.Fatalf("usage endpoint called %d times with an expired token", calls-callsBefore)
	}
}

func TestCswapPolicyUsageEndpointErrors(t *testing.T) {
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	selector := newCswapTestSelector(t, server, clock)
	now := clock.Now()
	auths := []*Auth{cswapTestAuth("a", now.Add(8*time.Hour)), cswapTestAuth("b", now.Add(8*time.Hour))}
	fake.set("token-a", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(24*time.Hour)))
	fake.set("token-b", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(4*24*time.Hour)))
	pollNow(selector, auths)
	if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != "a" {
		t.Fatalf("initial pick = %q, want a", got)
	}

	// a's usage stops being readable. Last-good data is trusted for an hour,
	// after which a counts as unknown and keeps its sessions until the third
	// consecutive failure, then fails over.
	fake.fail("token-a", http.StatusInternalServerError, "")
	clock.Advance(time.Hour + time.Minute)
	for failures := 1; failures <= cswapUnhealthyPolls; failures++ {
		before := fake.callCount("token-a")
		for i := 0; i < 20 && fake.callCount("token-a") == before; i++ {
			clock.Advance(15 * time.Second)
			selector.pollOnce(auths)
		}
		if fake.callCount("token-a") == before {
			t.Fatalf("failure %d: a was not re-polled", failures)
		}
		want := "a"
		if failures == cswapUnhealthyPolls {
			want = "b"
		}
		if got := pickID(t, selector, cliproxyexecutor.Options{}, auths); got != want {
			t.Fatalf("after %d failed polls: pick = %q, want %q", failures, got, want)
		}
	}
}

func TestCswapPolicyUsageBackoff(t *testing.T) {
	cases := []struct {
		name       string
		status     int
		retryAfter string
		body       string
		quiet      time.Duration // no poll before this
		next       time.Duration // a poll by this
	}{
		{name: "429 saturated edge", status: http.StatusTooManyRequests, retryAfter: "0", quiet: 299 * time.Second, next: 301 * time.Second},
		{name: "429 hour block takes the margin", status: http.StatusTooManyRequests, retryAfter: "3600", quiet: 4499 * time.Second, next: 4501 * time.Second},
		{name: "503 hour block has no margin", status: http.StatusServiceUnavailable, retryAfter: "3600", quiet: 3599 * time.Second, next: 3601 * time.Second},
		{name: "500 without retry-after", status: http.StatusInternalServerError, quiet: 29 * time.Second, next: 31 * time.Second},
		{name: "bad response", body: "not json", quiet: 29 * time.Second, next: 31 * time.Second},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clock := &cswapTestClock{now: time.Now()}
			fake, server := newCswapUsageServer(t)
			selector := newCswapTestSelector(t, server, clock)
			auths := []*Auth{cswapTestAuth("a", clock.Now().Add(8*time.Hour))}
			if tc.status != 0 {
				fake.fail("token-a", tc.status, tc.retryAfter)
			} else {
				fake.set("token-a", tc.body)
			}
			selector.pollOnce(auths)
			if fake.callCount("token-a") != 1 {
				t.Fatalf("first poll calls = %d", fake.callCount("token-a"))
			}
			clock.Advance(tc.quiet)
			selector.pollOnce(auths)
			if fake.callCount("token-a") != 1 {
				t.Fatalf("polled again after %s", tc.quiet)
			}
			clock.Advance(tc.next - tc.quiet)
			selector.pollOnce(auths)
			if fake.callCount("token-a") != 2 {
				t.Fatalf("not polled again by %s", tc.next)
			}
		})
	}
}

func TestCswapPolicyModelScopedWindow(t *testing.T) {
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	selector := newCswapTestSelector(t, server, clock)
	now := clock.Now()
	auths := []*Auth{cswapTestAuth("a", now.Add(8*time.Hour)), cswapTestAuth("b", now.Add(8*time.Hour))}
	reset := now.Add(24 * time.Hour).UTC().Format(time.RFC3339)
	fake.set("token-a", fmt.Sprintf(`{"five_hour":{"utilization":10,"resets_at":%q},"seven_day":{"utilization":20,"resets_at":%q},"limits":[{"kind":"weekly_scoped","percent":100,"resets_at":%q,"scope":{"model":{"display_name":"Opus"}}}]}`, reset, reset, reset))
	fake.set("token-b", cswapUsageBody(10, now.Add(2*time.Hour), 20, now.Add(4*24*time.Hour)))
	pollNow(selector, auths)
	pick := func(model string) string {
		t.Helper()
		picked, errPick := selector.Pick(context.Background(), "claude", model, cliproxyexecutor.Options{}, auths)
		if errPick != nil {
			t.Fatalf("Pick(%s): %v", model, errPick)
		}
		return picked.ID
	}
	if got := pick("claude-opus-5-5"); got != "b" {
		t.Fatalf("opus pick = %q, want b (a's Opus window is exhausted)", got)
	}
	if got := pick("claude-sonnet-5-5"); got != "a" {
		t.Fatalf("sonnet pick = %q, want a (soonest reset, Opus window irrelevant)", got)
	}
}

func TestCswapPolicySessionAffinityKeepsHealthyAccount(t *testing.T) {
	ctx := context.Background()
	clock := &cswapTestClock{now: time.Now()}
	fake, server := newCswapUsageServer(t)
	policy := newCswapTestSelector(t, server, clock)
	affinity := NewSessionAffinitySelectorWithConfig(SessionAffinityConfig{Fallback: policy, TTL: time.Hour})
	manager := NewManager(nil, nil, nil)
	manager.SetSelector(affinity)
	manager.RegisterExecutor(schedulerTestExecutor{provider: "claude"})
	now := clock.Now()
	for _, id := range []string{"cswap-aff-a", "cswap-aff-b", "cswap-aff-c"} {
		if _, errRegister := manager.Register(WithSkipPersist(ctx), cswapTestAuth(id, now.Add(8*time.Hour))); errRegister != nil {
			t.Fatalf("Register(%s): %v", id, errRegister)
		}
		registry.GetGlobalRegistry().RegisterClient(id, "claude", []*registry.ModelInfo{{ID: cswapTestModel}})
		authID := id
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(authID) })
	}
	fake.set("token-cswap-aff-a", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(2*24*time.Hour)))
	fake.set("token-cswap-aff-b", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(4*24*time.Hour)))
	fake.set("token-cswap-aff-c", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(5*24*time.Hour)))
	pollNow(policy, manager.List())

	session := func(id string) cliproxyexecutor.Options {
		return cliproxyexecutor.Options{Headers: http.Header{"X-Claude-Code-Session-Id": []string{id}}}
	}
	pick := func(opts cliproxyexecutor.Options) string {
		t.Helper()
		picked, _, errPick := manager.pickNext(ctx, "claude", cswapTestModel, opts, nil)
		if errPick != nil {
			t.Fatalf("pickNext: %v", errPick)
		}
		return picked.ID
	}

	if got := pick(session("s1")); got != "cswap-aff-a" {
		t.Fatalf("s1 cold binding = %q, want cswap-aff-a", got)
	}
	// c now resets soonest, so new sessions prefer it; s1 stays on healthy a.
	clock.Advance(time.Minute)
	now = clock.Now()
	fake.set("token-cswap-aff-c", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(24*time.Hour)))
	pollNow(policy, manager.List())
	if got := pick(session("s2")); got != "cswap-aff-c" {
		t.Fatalf("s2 cold binding = %q, want cswap-aff-c", got)
	}
	for i := 0; i < 3; i++ {
		if got := pick(session("s1")); got != "cswap-aff-a" {
			t.Fatalf("s1 request %d = %q, want cswap-aff-a (still healthy)", i, got)
		}
	}
	// a reaches the threshold: s1 moves to the preferred account and stays there.
	clock.Advance(CswapDefaultCooldown)
	now = clock.Now()
	fake.set("token-cswap-aff-a", cswapUsageBody(96, now.Add(2*time.Hour), 40, now.Add(2*24*time.Hour)))
	pollNow(policy, manager.List())
	for i := 0; i < 2; i++ {
		if got := pick(session("s1")); got != "cswap-aff-c" {
			t.Fatalf("s1 after a left rotation, request %d = %q, want cswap-aff-c", i, got)
		}
	}
	if got := pick(session("s2")); got != "cswap-aff-c" {
		t.Fatalf("s2 = %q, want cswap-aff-c", got)
	}
	// The preferred account c leaves rotation while only bound sessions arrive: the
	// sessions move to b and the preference moves with them, starting a cooldown.
	clock.Advance(time.Minute)
	now = clock.Now()
	fake.set("token-cswap-aff-c", cswapUsageBody(97, now.Add(2*time.Hour), 10, now.Add(24*time.Hour)))
	pollNow(policy, manager.List())
	for _, id := range []string{"s1", "s2"} {
		if got := pick(session(id)); got != "cswap-aff-b" {
			t.Fatalf("%s after c left rotation = %q, want cswap-aff-b", id, got)
		}
	}
	// c recovers inside that cooldown: new sessions stay on b until it ends.
	clock.Advance(time.Minute)
	now = clock.Now()
	fake.set("token-cswap-aff-c", cswapUsageBody(10, now.Add(2*time.Hour), 10, now.Add(24*time.Hour)))
	pollNow(policy, manager.List())
	if got := pick(session("s3")); got != "cswap-aff-b" {
		t.Fatalf("s3 inside the cooldown = %q, want cswap-aff-b", got)
	}
	clock.Advance(CswapDefaultCooldown)
	if got := pick(session("s4")); got != "cswap-aff-c" {
		t.Fatalf("s4 after the cooldown = %q, want cswap-aff-c (soonest reset)", got)
	}
	if got := pick(session("s1")); got != "cswap-aff-b" {
		t.Fatalf("s1 = %q, want cswap-aff-b (bound and healthy)", got)
	}
	// Everyone above the threshold (a's 5h at 96% resets first): new sessions follow
	// the preference to a, bound sessions stay on accounts that still have headroom.
	clock.Advance(CswapDefaultCooldown)
	now = clock.Now()
	fake.set("token-cswap-aff-b", cswapUsageBody(10, now.Add(2*time.Hour), 96, now.Add(4*24*time.Hour)))
	fake.set("token-cswap-aff-c", cswapUsageBody(10, now.Add(2*time.Hour), 97, now.Add(24*time.Hour)))
	pollNow(policy, manager.List())
	if got := pick(session("s5")); got != "cswap-aff-a" {
		t.Fatalf("s5 with everyone above the threshold = %q, want cswap-aff-a (soonest recovery)", got)
	}
	for id, want := range map[string]string{"s1": "cswap-aff-b", "s4": "cswap-aff-c", "s5": "cswap-aff-a"} {
		if got := pick(session(id)); got != want {
			t.Fatalf("%s with everyone above the threshold = %q, want %s", id, got, want)
		}
	}
}

func authIDs(auths []*Auth) []string {
	ids := make([]string, 0, len(auths))
	for _, auth := range auths {
		ids = append(ids, auth.ID)
	}
	return ids
}
