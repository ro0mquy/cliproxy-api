package auth

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"

	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
)

// The cswap-policy strategy ports the account-switching policy of the claude-swap
// (cswap) tool to the credential pool. Constants keep cswap's names and values
// (autoswitch.py, poll_policy.py, usage_store.py) so the two can be compared.
const (
	CswapDefaultThreshold     = 95.0
	CswapDefaultHysteresisPct = 10.0
	CswapDefaultCooldown      = 300 * time.Second

	// cswapUnhealthyPolls is cswap's unhealthy_ticks: consecutive failed usage polls
	// (with no trusted snapshot left) before an account counts as unhealthy.
	cswapUnhealthyPolls = 3

	// Every-account-above-threshold ranking (autoswitch.py).
	cswapRecoveryHysteresis = 300.0 // seconds, RECOVERY_HYSTERESIS_S
	cswapRecoveryHorizon    = 4 * 3600.0
	cswapHeadroomRatio      = 2.0
	cswapSpentHeadroomPct   = 3.0

	// Poll cadence (poll_policy.py). The usage endpoint allows roughly 30 requests
	// per hour per identity, and cswap may poll the same accounts from a laptop.
	cswapLoopInterval      = 5 * time.Second
	cswapMinInterval       = 180 * time.Second
	cswapDefaultInterval   = 300 * time.Second
	cswapMaxInterval       = 600 * time.Second
	cswapExhaustedInterval = 600 * time.Second
	cswapMovementPct       = 1.0
	cswapJitterFrac        = 0.1
	cswapResetSlack        = 60 * time.Second
	cswapPost429MinInt     = 360 * time.Second
	cswapPost429MaxInt     = 1800 * time.Second
	cswapRecent429Window   = time.Hour

	// Failure backoff and trust (usage_store.py).
	cswapBackoffBase           = 30 * time.Second
	cswapBackoffCap            = 600 * time.Second
	cswapEdgeBackoff           = 300 * time.Second
	cswapRetryAfterMargin      = 900 * time.Second
	cswapRetryAfterCap429      = 4500 * time.Second
	cswapRetryAfterCapOther    = 3600 * time.Second
	cswapTrustMaxAge           = time.Hour
	cswapRateLimitTrustMaxAge  = 2 * time.Hour
	cswapFetchTimeout          = 15 * time.Second
	cswapUsageEndpoint         = "https://api.anthropic.com/api/oauth/usage"
	cswapOAuthBetaHeader       = "oauth-2025-04-20"
	cswapMaxModelKeys          = 256
	cswapClaudeCodeUserAgent   = "claude-cli/2.1.280 (external, cli)" // executor default, helps/claude_device_profile.go
	cswapStrategyLogPrefix     = "cswap-policy"
	cswapWindowFiveHour        = "5h"
	cswapWindowSevenDay        = "7d"
	cswapScopedWindowKeyPrefix = "scoped:"
)

// CswapPolicyConfig tunes the cswap-policy strategy.
type CswapPolicyConfig struct {
	// Threshold is the binding-window utilization (percent) at which an account
	// leaves rotation.
	Threshold float64
	// HysteresisPct is how far below Threshold a window must fall before the
	// account returns to rotation.
	HysteresisPct float64
	// Cooldown bounds how often the preferred account changes voluntarily.
	Cooldown time.Duration
}

func (c CswapPolicyConfig) normalized() CswapPolicyConfig {
	if c.Threshold <= 0 {
		c.Threshold = CswapDefaultThreshold
	}
	c.Threshold = math.Min(math.Max(c.Threshold, 50), 99.9)
	if c.HysteresisPct <= 0 {
		c.HysteresisPct = CswapDefaultHysteresisPct
	}
	c.HysteresisPct = math.Min(c.HysteresisPct, 50)
	if c.Cooldown <= 0 {
		c.Cooldown = CswapDefaultCooldown
	}
	return c
}

// CswapPolicySelector routes Claude OAuth credentials by their Anthropic usage
// windows. A background poller reads GET /api/oauth/usage for each account; Pick
// performs no I/O and decides on the last known values:
//
//   - A window at or above the threshold takes the account out of rotation. It
//     returns once that window falls hysteresis points below the threshold.
//   - New sessions go to a preferred account per model. While it is healthy the
//     preferred account only moves to an account whose 7-day window resets
//     sooner (consume-first), at most once per cooldown.
//   - When every measured account is out of rotation, accounts that still have
//     headroom stay usable and the preferred account follows cswap's
//     soonest-recovery ranking.
//
// The selector never refreshes tokens. Accounts with an expired access token
// are skipped until the manager refreshes them. Non-Claude credentials are
// rotated round-robin.
type CswapPolicySelector struct {
	cfg      CswapPolicyConfig
	fallback *RoundRobinSelector

	mu       sync.Mutex
	accounts map[string]*cswapAccount
	models   map[string]*cswapModelState

	httpClient *http.Client
	usageURL   string
	now        func() time.Time
	randFloat  func() float64

	startOnce sync.Once
	stopOnce  sync.Once
	stopCh    chan struct{}
}

type cswapWindow struct {
	key      string
	pct      float64
	resetsAt time.Time
}

type cswapUsage struct {
	windows []cswapWindow
}

type cswapAccount struct {
	usage      *cswapUsage
	fetchedAt  time.Time
	nextPollAt time.Time
	interval   time.Duration
	failures   int
	lastError  string
	last429At  time.Time
	out        map[string]bool
}

type cswapModelState struct {
	current      string
	lastSwitchAt time.Time
}

type cswapStatus int

const (
	cswapHealthy cswapStatus = iota
	cswapUnknown
	cswapOut
	cswapUnhealthy
	cswapExpired
)

func (s cswapStatus) String() string {
	switch s {
	case cswapHealthy:
		return "healthy"
	case cswapUnknown:
		return "unknown"
	case cswapOut:
		return "out"
	case cswapUnhealthy:
		return "unhealthy"
	default:
		return "expired"
	}
}

// cswapView is one account's standing for one model at one instant.
type cswapView struct {
	auth     *Auth
	status   cswapStatus
	known    bool
	headroom float64 // 100 - binding window pct, valid when known
	reset7d  float64 // unix seconds of the 7-day reset, +Inf when unknown or past
	recovery float64 // unix seconds of the binding window reset, +Inf when unknown or past
}

// NewCswapPolicySelector creates the selector. The usage poller starts when the
// manager binds its credential list.
func NewCswapPolicySelector(cfg CswapPolicyConfig) *CswapPolicySelector {
	return &CswapPolicySelector{
		cfg:        cfg.normalized(),
		fallback:   &RoundRobinSelector{},
		accounts:   make(map[string]*cswapAccount),
		models:     make(map[string]*cswapModelState),
		httpClient: &http.Client{},
		usageURL:   cswapUsageEndpoint,
		now:        time.Now,
		randFloat:  rand.Float64,
		stopCh:     make(chan struct{}),
	}
}

// bindAuthList starts the background usage poller over the manager's credentials.
func (s *CswapPolicySelector) bindAuthList(list func() []*Auth) {
	if s == nil || list == nil {
		return
	}
	s.startOnce.Do(func() { go s.pollLoop(list) })
}

// Stop terminates the background usage poller.
func (s *CswapPolicySelector) Stop() {
	if s == nil {
		return
	}
	s.stopOnce.Do(func() { close(s.stopCh) })
}

// Pick selects a credential without network I/O.
func (s *CswapPolicySelector) Pick(ctx context.Context, provider, model string, opts cliproxyexecutor.Options, auths []*Auth) (*Auth, error) {
	now := s.now()
	available, err := getSelectorAvailableAuths(ctx, auths, provider, model, now)
	if err != nil {
		return nil, err
	}
	claude, others := splitCswapCandidates(available)
	if len(claude) == 0 {
		return s.fallback.Pick(ctx, provider, model, opts, others)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	picked := s.pickLocked(selectorLogEntry(ctx), model, claude, now)
	if picked == nil {
		return nil, &Error{Code: "auth_unavailable", Message: "no Claude credential has a valid access token"}
	}
	return picked, nil
}

// filterSessionEligible drops Claude credentials that are out of rotation, so a
// session binding only survives while its credential is healthy.
func (s *CswapPolicySelector) filterSessionEligible(_ string, model string, auths []*Auth) []*Auth {
	claude, _ := splitCswapCandidates(auths)
	if len(claude) == 0 {
		return auths
	}
	now := s.now()
	s.mu.Lock()
	views := s.viewsLocked(model, claude, now)
	eligible := s.eligibleLocked(canonicalCswapModel(model), views, now)
	s.mu.Unlock()
	keep := make(map[string]struct{}, len(eligible))
	for _, view := range eligible {
		keep[view.auth.ID] = struct{}{}
	}
	out := make([]*Auth, 0, len(auths))
	for _, auth := range auths {
		if !isCswapCandidate(auth) {
			out = append(out, auth)
			continue
		}
		if _, ok := keep[auth.ID]; ok {
			out = append(out, auth)
		}
	}
	return out
}

func splitCswapCandidates(auths []*Auth) (claude, others []*Auth) {
	for _, auth := range auths {
		if isCswapCandidate(auth) {
			claude = append(claude, auth)
		} else if auth != nil {
			others = append(others, auth)
		}
	}
	return claude, others
}

// isCswapCandidate reports whether the credential is a Claude OAuth account with
// usage windows. API-key credentials are excluded, as cswap excludes them.
func isCswapCandidate(auth *Auth) bool {
	return auth != nil &&
		strings.EqualFold(strings.TrimSpace(auth.Provider), "claude") &&
		auth.AuthKind() == AuthKindOAuth &&
		authAccessToken(auth) != ""
}

func canonicalCswapModel(model string) string {
	return strings.ToLower(canonicalModelKey(model))
}

func (s *CswapPolicySelector) pickLocked(entry *log.Entry, model string, candidates []*Auth, now time.Time) *Auth {
	modelKey := canonicalCswapModel(model)
	state := s.models[modelKey]
	if state == nil {
		if len(s.models) >= cswapMaxModelKeys {
			s.models = make(map[string]*cswapModelState)
		}
		state = &cswapModelState{}
		s.models[modelKey] = state
	}
	views := s.viewsLocked(model, candidates, now)
	eligible := s.eligibleLocked(modelKey, views, now)
	if len(eligible) == 0 {
		return nil
	}

	cur := findCswapView(eligible, state.current)
	if cur == nil {
		target := cswapRankFailover(eligible, "")
		if target == nil {
			target = &eligible[0]
		}
		offered := findCswapView(views, state.current)
		switch {
		case state.current == "":
			state.current = target.auth.ID
			entry.Infof("%s: preferred account | model=%s auth=%s status=%s", cswapStrategyLogPrefix, modelKey, target.auth.ID, target.status)
		case offered != nil:
			// The policy itself rejected the preferred account: commit the move.
			s.switchLocked(entry, modelKey, state, offered, target, cswapEscapeTrigger(offered), now)
		case s.accounts[state.current] == nil:
			// The poller no longer tracks it: the credential was removed or disabled.
			entry.Infof("%s: preferred account gone | model=%s from=%s to=%s", cswapStrategyLogPrefix, modelKey, state.current, target.auth.ID)
			state.current = target.auth.ID
		default:
			// The manager did not offer the preferred account (429 cooldown, retry,
			// expired token). Serve this request elsewhere and keep the preference.
		}
		return target.auth
	}

	trigger := ""
	switch cur.status {
	case cswapUnknown:
		return cur.auth
	case cswapUnhealthy:
		target := cswapRankFailover(eligible, cur.auth.ID)
		if target == nil {
			return cur.auth
		}
		s.switchLocked(entry, modelKey, state, cur, target, "failover", now)
		return target.auth
	case cswapHealthy:
		trigger = "consume-first"
	default:
		trigger = "proactive"
	}
	if s.inCooldownLocked(state, now) {
		return cur.auth
	}
	ranked := cswapRankCandidates(trigger, cur, views, eligible, now)
	if len(ranked) == 0 {
		return cur.auth
	}
	s.switchLocked(entry, modelKey, state, cur, &ranked[0], trigger, now)
	return ranked[0].auth
}

func cswapEscapeTrigger(view *cswapView) string {
	switch {
	case view.status == cswapOut && view.headroom <= 0:
		return "at-limit"
	case view.status == cswapOut:
		return "proactive"
	default:
		return "failover"
	}
}

func (s *CswapPolicySelector) switchLocked(entry *log.Entry, modelKey string, state *cswapModelState, from, to *cswapView, trigger string, now time.Time) {
	state.current = to.auth.ID
	state.lastSwitchAt = now
	entry.Infof("%s: switch | model=%s from=%s (%s%s) to=%s (%s%s) trigger=%s",
		cswapStrategyLogPrefix, modelKey,
		from.auth.ID, from.status, cswapUsageLabel(from),
		to.auth.ID, to.status, cswapUsageLabel(to), trigger)
}

func cswapUsageLabel(view *cswapView) string {
	if !view.known {
		return ""
	}
	return fmt.Sprintf(" %.0f%%", 100-view.headroom)
}

func (s *CswapPolicySelector) inCooldownLocked(state *cswapModelState, now time.Time) bool {
	return !state.lastSwitchAt.IsZero() && now.Sub(state.lastSwitchAt) < s.cfg.Cooldown
}

func findCswapView(views []cswapView, id string) *cswapView {
	if id == "" {
		return nil
	}
	for i := range views {
		if views[i].auth.ID == id {
			return &views[i]
		}
	}
	return nil
}

func (s *CswapPolicySelector) viewsLocked(model string, auths []*Auth, now time.Time) []cswapView {
	views := make([]cswapView, 0, len(auths))
	for _, auth := range auths {
		views = append(views, s.viewLocked(auth, model, now))
	}
	sort.SliceStable(views, func(i, j int) bool { return views[i].auth.ID < views[j].auth.ID })
	return views
}

func (s *CswapPolicySelector) viewLocked(auth *Auth, model string, now time.Time) cswapView {
	view := cswapView{auth: auth, status: cswapUnknown, reset7d: math.Inf(1), recovery: math.Inf(1)}
	if exp, ok := auth.AccessTokenExpirationTime(); ok && !exp.IsZero() && !exp.After(now) {
		view.status = cswapExpired
		return view
	}
	acct := s.accounts[auth.ID]
	if acct == nil {
		return view
	}
	windows := acct.relevantWindows(model, now)
	if len(windows) == 0 {
		if acct.failures >= cswapUnhealthyPolls {
			view.status = cswapUnhealthy
		}
		return view
	}
	view.known = true
	binding := windows[0]
	out := false
	for _, window := range windows {
		if window.pct > binding.pct {
			binding = window
		}
		if acct.out[window.key] {
			out = true
		}
		if window.key == cswapWindowSevenDay && window.resetsAt.After(now) {
			view.reset7d = cswapUnix(window.resetsAt)
		}
	}
	view.headroom = 100 - binding.pct
	if binding.resetsAt.After(now) {
		view.recovery = cswapUnix(binding.resetsAt)
	}
	view.status = cswapHealthy
	if out {
		view.status = cswapOut
	}
	return view
}

func cswapUnix(t time.Time) float64 {
	return float64(t.UnixNano()) / float64(time.Second)
}

// eligibleLocked returns the accounts requests may use, first usable tier wins:
//  1. healthy and not-yet-measured accounts, plus the preferred account while its
//     cooldown protects it (cswap holds a proactive switch for the cooldown);
//  2. everyone is out of rotation: accounts still below 100% and unhealthy ones;
//  3. every account with an unexpired token.
//
// Tier 1 needs a healthy or unmeasured account. Otherwise the preferred account
// would be the only eligible one for a cooldown after each switch, and bound
// sessions would move to it only during that window.
func (s *CswapPolicySelector) eligibleLocked(modelKey string, views []cswapView, now time.Time) []cswapView {
	protected := ""
	if state := s.models[modelKey]; state != nil && s.inCooldownLocked(state, now) {
		protected = state.current
	}
	inRotation := false
	tier1 := make([]cswapView, 0, len(views))
	tier2 := make([]cswapView, 0, len(views))
	tier3 := make([]cswapView, 0, len(views))
	for _, view := range views {
		switch view.status {
		case cswapExpired:
			continue
		case cswapHealthy, cswapUnknown:
			inRotation = true
			tier1 = append(tier1, view)
		case cswapOut:
			if view.headroom > 0 {
				if view.auth.ID == protected {
					tier1 = append(tier1, view)
				}
				tier2 = append(tier2, view)
			}
		case cswapUnhealthy:
			tier2 = append(tier2, view)
		}
		tier3 = append(tier3, view)
	}
	if inRotation {
		return tier1
	}
	if len(tier2) > 0 {
		return tier2
	}
	return tier3
}

type cswapRankKey struct {
	tier int
	a, b float64
}

func (k cswapRankKey) less(o cswapRankKey) bool {
	if k.tier != o.tier {
		return k.tier < o.tier
	}
	if k.a != o.a {
		return k.a < o.a
	}
	return k.b < o.b
}

type cswapRanked struct {
	key  cswapRankKey
	view cswapView
}

func sortCswapRanked(items []cswapRanked) []cswapView {
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].key != items[j].key {
			return items[i].key.less(items[j].key)
		}
		return items[i].view.auth.ID < items[j].view.auth.ID
	})
	out := make([]cswapView, 0, len(items))
	for _, item := range items {
		out = append(out, item.view)
	}
	return out
}

// cswapRankFailover ranks the escape targets (at-limit and failover in cswap):
// any measured account with headroom, soonest 7-day reset first, then most
// headroom. With nothing measured it returns nil.
func cswapRankFailover(eligible []cswapView, exclude string) *cswapView {
	items := make([]cswapRanked, 0, len(eligible))
	for _, view := range eligible {
		if view.auth.ID == exclude || !view.known || view.headroom <= 0 {
			continue
		}
		items = append(items, cswapRanked{key: cswapRankKey{a: view.reset7d, b: -view.headroom}, view: view})
	}
	ranked := sortCswapRanked(items)
	if len(ranked) == 0 {
		return nil
	}
	return &ranked[0]
}

// cswapRankCandidates ports autoswitch._rank_candidates for the consume-first
// strategy and the proactive and consume-first triggers.
func cswapRankCandidates(trigger string, cur *cswapView, views, eligible []cswapView, now time.Time) []cswapView {
	nowS := cswapUnix(now)
	candidates := make([]cswapView, 0, len(eligible))
	for _, view := range eligible {
		if view.auth.ID != cur.auth.ID {
			candidates = append(candidates, view)
		}
	}
	// Every account above the threshold: cswap measures this on all readable
	// candidates, not only the eligible ones.
	allAbove := false
	if cur.known && cur.status == cswapOut {
		measured := 0
		allAbove = true
		for _, view := range views {
			if view.auth.ID == cur.auth.ID || !view.known {
				continue
			}
			measured++
			if view.status != cswapOut {
				allAbove = false
			}
		}
		allAbove = allAbove && measured > 0
	}
	bestCandidateHeadroom := 0.0
	for _, view := range candidates {
		if view.known && view.headroom > bestCandidateHeadroom {
			bestCandidateHeadroom = view.headroom
		}
	}
	activeHeadroom := 0.0
	if cur.known {
		activeHeadroom = cur.headroom
	}

	qualifying := make([]cswapRanked, 0, len(candidates))
	fallback := make([]cswapRanked, 0)
	for _, view := range candidates {
		if !view.known || view.headroom <= 0 {
			continue
		}
		h := view.headroom
		if allAbove {
			byRecovery := cswapRecoveryIsUseful(view.recovery, cur.recovery, activeHeadroom, bestCandidateHeadroom, nowS)
			if byRecovery {
				if view.recovery >= cur.recovery-cswapRecoveryHysteresis {
					continue
				}
				qualifying = append(qualifying, cswapRanked{key: cswapRankKey{tier: 0, a: view.recovery, b: -h}, view: view})
				continue
			}
			if h < activeHeadroom*cswapHeadroomRatio {
				if activeHeadroom <= cswapSpentHeadroomPct && h >= activeHeadroom && view.recovery < cur.recovery-cswapRecoveryHysteresis {
					fallback = append(fallback, cswapRanked{key: cswapRankKey{tier: 0, a: view.recovery, b: -h}, view: view})
				}
				continue
			}
			qualifying = append(qualifying, cswapRanked{key: cswapRankKey{tier: 1, a: -h, b: view.recovery}, view: view})
			continue
		}
		// Landing must be healthy: an account out of rotation would re-trigger.
		if view.status == cswapOut {
			continue
		}
		if trigger == "consume-first" && (math.IsInf(view.reset7d, 1) || math.IsInf(cur.reset7d, 1) || view.reset7d >= cur.reset7d) {
			continue
		}
		qualifying = append(qualifying, cswapRanked{key: cswapRankKey{a: view.reset7d, b: -h}, view: view})
	}
	if len(qualifying) == 0 {
		qualifying = fallback
	}
	return sortCswapRanked(qualifying)
}

// cswapRecoveryIsUseful ports autoswitch._recovery_is_useful.
func cswapRecoveryIsUseful(candidateRecovery, activeRecovery, activeHeadroom, bestCandidateHeadroom, now float64) bool {
	if activeHeadroom <= cswapSpentHeadroomPct && bestCandidateHeadroom <= cswapSpentHeadroomPct {
		return true
	}
	return candidateRecovery-now <= cswapRecoveryHorizon || activeRecovery-now <= cswapRecoveryHorizon
}

// relevantWindows returns the windows that gate the model: always 5h and 7d,
// plus per-model weekly windows whose display name occurs in the model name.
// Usage older than its trust horizon reads as unknown.
func (a *cswapAccount) relevantWindows(model string, now time.Time) []cswapWindow {
	if a == nil || a.usage == nil || !a.trusted(now) {
		return nil
	}
	lowerModel := strings.ToLower(model)
	windows := make([]cswapWindow, 0, len(a.usage.windows))
	for _, window := range a.usage.windows {
		if name, scoped := strings.CutPrefix(window.key, cswapScopedWindowKeyPrefix); scoped && !strings.Contains(lowerModel, name) {
			continue
		}
		windows = append(windows, window)
	}
	return windows
}

// trusted mirrors cswap's decision trust: last-good data holds for an hour, and
// after a usage-endpoint 429 for two hours unless a window has reset since.
func (a *cswapAccount) trusted(now time.Time) bool {
	age := now.Sub(a.fetchedAt)
	if age <= cswapTrustMaxAge {
		return true
	}
	if a.lastError != "http-429" || age > cswapRateLimitTrustMaxAge {
		return false
	}
	for _, window := range a.usage.windows {
		if !window.resetsAt.IsZero() && !window.resetsAt.After(now) {
			return false
		}
	}
	return true
}

func (a *cswapAccount) bindingPct() (float64, bool) {
	if a == nil || a.usage == nil || len(a.usage.windows) == 0 {
		return 0, false
	}
	pct := a.usage.windows[0].pct
	for _, window := range a.usage.windows[1:] {
		pct = math.Max(pct, window.pct)
	}
	return pct, true
}

// pollLoop polls at most one due account per loop tick.
func (s *CswapPolicySelector) pollLoop(list func() []*Auth) {
	ticker := time.NewTicker(cswapLoopInterval)
	defer ticker.Stop()
	for {
		s.pollOnce(list())
		select {
		case <-s.stopCh:
			return
		case <-ticker.C:
		}
	}
}

type cswapPollTarget struct {
	id    string
	token string
}

// pollOnce fetches usage for the most overdue account and records the result.
func (s *CswapPolicySelector) pollOnce(auths []*Auth) {
	target, ok := s.nextPollTarget(auths)
	if !ok {
		return
	}
	usage, kind, retryAfter, errFetch := s.fetchUsage(target.token)
	s.recordPoll(target.id, usage, kind, retryAfter, errFetch)
}

func (s *CswapPolicySelector) nextPollTarget(auths []*Auth) (cswapPollTarget, bool) {
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	present := make(map[string]struct{}, len(auths))
	var best cswapPollTarget
	var bestFetched time.Time
	found := false
	for _, auth := range auths {
		if !isCswapCandidate(auth) || auth.Disabled || auth.Status == StatusDisabled {
			continue
		}
		present[auth.ID] = struct{}{}
		if exp, okExp := auth.AccessTokenExpirationTime(); okExp && !exp.IsZero() && !exp.After(now) {
			// The manager refreshes tokens; polling with an expired one only earns a 401.
			continue
		}
		acct := s.accounts[auth.ID]
		if acct == nil {
			acct = &cswapAccount{out: make(map[string]bool)}
			s.accounts[auth.ID] = acct
		}
		if now.Before(acct.nextPollAt) {
			continue
		}
		if !found || acct.fetchedAt.Before(bestFetched) || (acct.fetchedAt.Equal(bestFetched) && auth.ID < best.id) {
			best = cswapPollTarget{id: auth.ID, token: authAccessToken(auth)}
			bestFetched = acct.fetchedAt
			found = true
		}
	}
	for id := range s.accounts {
		if _, ok := present[id]; !ok {
			delete(s.accounts, id)
		}
	}
	return best, found
}

// fetchUsage mirrors cswap's oauth.request_usage_data and _classify_usage_error.
// The poll is background credential-quota work, never on a request path, so it
// carries its own deadline.
func (s *CswapPolicySelector) fetchUsage(token string) (*cswapUsage, string, *time.Duration, error) {
	ctx, cancel := context.WithTimeout(context.Background(), cswapFetchTimeout)
	defer cancel()
	go func() {
		select {
		case <-s.stopCh:
			cancel()
		case <-ctx.Done():
		}
	}()
	req, errRequest := http.NewRequestWithContext(ctx, http.MethodGet, s.usageURL, nil)
	if errRequest != nil {
		return nil, "bad-request", nil, errRequest
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("anthropic-beta", cswapOAuthBetaHeader)
	req.Header.Set("User-Agent", cswapClaudeCodeUserAgent)
	resp, errDo := s.httpClient.Do(req)
	if errDo != nil {
		var netErr net.Error
		if errors.Is(errDo, context.DeadlineExceeded) || (errors.As(errDo, &netErr) && netErr.Timeout()) {
			return nil, "timeout", nil, errDo
		}
		return nil, "network", nil, errDo
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			log.Debugf("%s: close usage response: %v", cswapStrategyLogPrefix, errClose)
		}
	}()
	body, errRead := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		return nil, "http-" + strconv.Itoa(resp.StatusCode), parseCswapRetryAfter(resp.Header.Get("Retry-After")), fmt.Errorf("usage endpoint returned %d", resp.StatusCode)
	}
	if errRead != nil {
		return nil, "network", nil, errRead
	}
	usage, errParse := parseCswapUsage(body)
	if errParse != nil {
		return nil, "bad-response", nil, errParse
	}
	return usage, "", nil, nil
}

// parseCswapRetryAfter accepts the seconds form only, like cswap.
func parseCswapRetryAfter(raw string) *time.Duration {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	seconds, errParse := strconv.ParseFloat(raw, 64)
	if errParse != nil || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
		return nil
	}
	wait := time.Duration(math.Max(0, seconds) * float64(time.Second))
	return &wait
}

// parseCswapUsage mirrors cswap's oauth.build_usage_result: five_hour and
// seven_day utilization, plus per-model weekly windows from limits[]. A response
// without five_hour/seven_day falls back to limits[] kinds session and
// weekly_all. It returns nil usage when the response carries no window.
func parseCswapUsage(body []byte) (*cswapUsage, error) {
	if !gjson.ValidBytes(body) {
		return nil, errors.New("usage response is not JSON")
	}
	root := gjson.ParseBytes(body)
	usage := &cswapUsage{}
	add := func(key string, pct, resetsAt gjson.Result) {
		if pct.Type != gjson.Number {
			return
		}
		window := cswapWindow{key: key, pct: pct.Float()}
		if parsed, errParse := time.Parse(time.RFC3339Nano, resetsAt.String()); errParse == nil {
			window.resetsAt = parsed
		}
		usage.windows = append(usage.windows, window)
	}
	add(cswapWindowFiveHour, root.Get("five_hour.utilization"), root.Get("five_hour.resets_at"))
	add(cswapWindowSevenDay, root.Get("seven_day.utilization"), root.Get("seven_day.resets_at"))
	haveFive, haveSeven := root.Get("five_hour.utilization").Exists(), root.Get("seven_day.utilization").Exists()
	root.Get("limits").ForEach(func(_, limit gjson.Result) bool {
		if name := strings.ToLower(strings.TrimSpace(limit.Get("scope.model.display_name").String())); name != "" {
			add(cswapScopedWindowKeyPrefix+name, limit.Get("percent"), limit.Get("resets_at"))
			return true
		}
		switch limit.Get("kind").String() {
		case "session":
			if !haveFive {
				add(cswapWindowFiveHour, limit.Get("percent"), limit.Get("resets_at"))
			}
		case "weekly_all":
			if !haveSeven {
				add(cswapWindowSevenDay, limit.Get("percent"), limit.Get("resets_at"))
			}
		}
		return true
	})
	if len(usage.windows) == 0 {
		return nil, nil
	}
	return usage, nil
}

func (s *CswapPolicySelector) recordPoll(id string, usage *cswapUsage, kind string, retryAfter *time.Duration, errFetch error) {
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	acct := s.accounts[id]
	if acct == nil {
		return
	}
	if errFetch != nil {
		acct.failures++
		acct.lastError = kind
		if kind == "http-429" {
			acct.last429At = now
		}
		backoff := cswapFailureBackoff(acct.failures, retryAfter, kind == "http-429")
		acct.nextPollAt = now.Add(backoff)
		cause := kind
		if retryAfter != nil {
			cause = fmt.Sprintf("%s, retry-after %.0fs", kind, retryAfter.Seconds())
		}
		log.Warnf("%s: usage fetch failed | auth=%s cause=%s failures=%d next_poll_in=%s", cswapStrategyLogPrefix, id, cause, acct.failures, backoff.Round(time.Second))
		return
	}
	prevPct, prevKnown := acct.bindingPct()
	prevInterval := acct.interval
	acct.usage = usage
	acct.fetchedAt = now
	acct.failures = 0
	acct.lastError = ""
	s.updateRotationLocked(id, acct)
	acct.nextPollAt, acct.interval = s.planNextPoll(acct, prevInterval, prevPct, prevKnown, now)
	pct, known := acct.bindingPct()
	log.Debugf("%s: usage | auth=%s binding=%.1f known=%t windows=%d next_poll_in=%s", cswapStrategyLogPrefix, id, pct, known, len(acct.windows()), acct.nextPollAt.Sub(now).Round(time.Second))
}

func (a *cswapAccount) windows() []cswapWindow {
	if a == nil || a.usage == nil {
		return nil
	}
	return a.usage.windows
}

// updateRotationLocked applies the threshold and hysteresis per window: a
// window at or above the threshold takes the account out of rotation, and it
// returns once that window drops below threshold - hysteresis.
func (s *CswapPolicySelector) updateRotationLocked(id string, acct *cswapAccount) {
	seen := make(map[string]struct{}, len(acct.windows()))
	for _, window := range acct.windows() {
		seen[window.key] = struct{}{}
		switch {
		case !acct.out[window.key] && window.pct >= s.cfg.Threshold:
			acct.out[window.key] = true
			log.Infof("%s: out of rotation | auth=%s window=%s usage=%.1f%% threshold=%.1f%%", cswapStrategyLogPrefix, id, window.key, window.pct, s.cfg.Threshold)
		case acct.out[window.key] && window.pct < s.cfg.Threshold-s.cfg.HysteresisPct:
			delete(acct.out, window.key)
			log.Infof("%s: back in rotation | auth=%s window=%s usage=%.1f%% below=%.1f%%", cswapStrategyLogPrefix, id, window.key, window.pct, s.cfg.Threshold-s.cfg.HysteresisPct)
		}
	}
	for key := range acct.out {
		if _, ok := seen[key]; !ok {
			delete(acct.out, key)
		}
	}
}

// planNextPoll ports poll_policy.plan_after_fetch without urgent mode.
func (s *CswapPolicySelector) planNextPoll(acct *cswapAccount, prevInterval time.Duration, prevPct float64, prevKnown bool, now time.Time) (time.Time, time.Duration) {
	base := prevInterval
	if base <= 0 {
		base = cswapDefaultInterval
	}
	newPct, newKnown := acct.bindingPct()
	interval := cswapDefaultInterval
	switch {
	case !prevKnown || !newKnown:
	case math.Abs(newPct-prevPct) >= cswapMovementPct:
		interval = maxDuration(cswapMinInterval, base/2)
	default:
		interval = minDuration(cswapMaxInterval, maxDuration(cswapMinInterval, time.Duration(float64(base)*1.5)))
	}
	if !acct.last429At.IsZero() && now.Sub(acct.last429At) < cswapRecent429Window {
		increased := maxDuration(time.Duration(float64(base)*1.5), cswapPost429MinInt)
		interval = minDuration(cswapPost429MaxInt, maxDuration(interval, increased))
	}
	exhausted := newKnown && newPct >= 100
	if exhausted {
		interval = maxDuration(interval, cswapExhaustedInterval)
	}
	jitter := 1 + cswapJitterFrac*(2*s.randFloat()-1)
	next := now.Add(time.Duration(float64(interval) * jitter))
	var bound time.Time
	for _, window := range acct.windows() {
		if !window.resetsAt.After(now) || (exhausted && window.pct < 100) {
			continue
		}
		switch {
		case exhausted && (bound.IsZero() || window.resetsAt.After(bound)):
			bound = window.resetsAt
		case !exhausted && (bound.IsZero() || window.resetsAt.Before(bound)):
			bound = window.resetsAt
		}
	}
	if !bound.IsZero() && bound.Add(cswapResetSlack).Before(next) {
		next = bound.Add(cswapResetSlack)
	}
	return next, interval
}

// cswapFailureBackoff ports usage_store._failure_backoff_s.
func cswapFailureBackoff(failures int, retryAfter *time.Duration, rateLimited bool) time.Duration {
	shift := failures - 1
	if shift < 0 {
		shift = 0
	}
	if shift > 5 {
		shift = 5
	}
	computed := minDuration(cswapBackoffBase*time.Duration(1<<shift), cswapBackoffCap)
	if retryAfter == nil {
		return computed
	}
	if *retryAfter == 0 {
		if !rateLimited {
			return computed
		}
		return minDuration(maxDuration(computed, cswapEdgeBackoff), cswapBackoffCap)
	}
	asked := *retryAfter
	if asked > cswapBackoffCap && rateLimited {
		asked += cswapRetryAfterMargin
	}
	if rateLimited {
		asked = minDuration(asked, cswapRetryAfterCap429)
	} else {
		asked = minDuration(asked, cswapRetryAfterCapOther)
	}
	return maxDuration(asked, computed)
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

func maxDuration(a, b time.Duration) time.Duration {
	if a > b {
		return a
	}
	return b
}
