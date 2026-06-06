"""V6 Phase-5 -- UCB exploration + Innovation Sandbox (PRIVATE / SHADOW + OFFLINE).

Standard-library only (math). No numpy/scipy hard dependency, no Mongo, no LLM.
Deterministic control-plane math: this module decides WHICH agent/lane to *explore*
next under a fixed exploration budget, on LOW-tail-risk lanes ONLY. It NEVER admits an
unproven agent to a money / security / live lane -- that gate is inherited 1:1 from the
V2 hard gate + the V5/V6 critical-risk classifier and is enforced here as a HARD filter.

Where this sits in the V-stack (see AGENT_MGMT_V6_CALIBRATION_PROMOTION_MANDATE_20260603):
    Phase 5 / Tier 4B = "UCB exploration on low-risk lanes + Innovation Sandbox
    (~5% budget, isolated)". Done = surfaces useful new agents, ZERO core incidents.
    Promotion gate = "sandbox-surfaced agents reach > median mu within 20 tasks at
    > 2x random base rate; zero isolation leakage to core".

Two public entry points (the deliverable contract):
    ucb_score(mean, n, total_n, c)        -> UCB1 index for one arm.
    sandbox_pick(candidates, budget, risk_caps) -> SandboxDecision.

Heavy math included (all closed-form, deterministic):
  * UCB1 index + Auer et al. (2002) logarithmic regret bound  (regret_bound_ucb1).
  * Hoeffding one-sided confidence radius  (the UCB radius is exactly this).
  * KL-UCB index for Bernoulli rewards via closed-form bisection on the
    Chernoff information  (kl_ucb_score) -- tighter than UCB1, same guarantees.
  * Wilson score interval -> lower confidence bound used to decide PROVEN vs UNPROVEN
    (an agent is "proven" only if its LCB on success clears a floor; this is the same
     philosophy as the V6 P0-1 fix: never trust a synthetic mean with no evidence).
  * Beta(1,1)-posterior mean/MoM "graduation" check against the fleet median mu.

Risk gating is reused, not reinvented: the money/security/live caps come from
governance_params.V2_CREDIT (the hard gate) and the V5/V6 critical-risk levels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

try:  # reuse the single source of truth; degrade to mirrored defaults if absent
    from quant.governance import governance_params as GP  # type: ignore
    _V2 = GP.V2_CREDIT
    _V5 = GP.V5_OPTIMIZATION
except Exception:  # pragma: no cover - keep module importable in isolation
    _V2 = {
        "MONEY_SECURITY_GATE_MIN_HP": 80.0,
        "MONEY_SECURITY_GATE_MIN_MU": 75.0,
        "MONEY_SECURITY_GATE_MAX_SIGMA": 15.0,
    }
    _V5 = {"LOW_CONFIDENCE_MIN_P": 0.72, "CRITICAL_CONFIDENCE_MIN_P": 0.86}


# ---------------------------------------------------------------------------
# Constants -- all derived from / consistent with the V6 mandate + governance_params.
# ---------------------------------------------------------------------------
# A lane is sandbox-eligible only if EVERY tail-risk dimension is at or below this.
# Mirrors the V5/V6 critical classifier: risk >= 3 is "critical" (money/security/live);
# the sandbox lives strictly UNDER that line. 2 = the highest level still non-critical.
SANDBOX_MAX_LANE_RISK = 2.0
CRITICAL_RISK_LEVEL = 3.0          # dispatcher._is_genuine_money_critical floor

# Default exploration budget = ~5% of fleet capacity (mandate: "~5% budget, isolated").
DEFAULT_SANDBOX_BUDGET_FRACTION = 0.05

# UCB1 default exploration coefficient. Auer et al. (2002) prove logarithmic regret for
# c = sqrt(2) when rewards are in [0,1]; we expose c so the controller can anneal it.
UCB1_C = math.sqrt(2.0)

# An agent is "proven" (NOT a sandbox candidate) once the lower Wilson bound on its
# success rate clears this floor with enough evidence. Kept distinct from -- and looser
# than -- the money/security hard gate, because the sandbox is a low-risk lane.
PROVEN_LCB_FLOOR = 0.55
PROVEN_MIN_SAMPLES = 8             # below this, evidence is too thin to call "proven"

# Wilson / confidence default two-sided level.
DEFAULT_CONFIDENCE = 0.95
_Z_BY_CONF = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


# ---------------------------------------------------------------------------
# Closed-form confidence helpers (stdlib only)
# ---------------------------------------------------------------------------
def _z_for(conf: float) -> float:
    """Two-sided normal quantile z_{1-(1-conf)/2}. Table for common levels; else a
    rational (Beasley-Springer/Moro-style Acklam) inverse-normal approximation."""
    if conf in _Z_BY_CONF:
        return _Z_BY_CONF[conf]
    p = 1.0 - (1.0 - conf) / 2.0
    return _inv_norm(p)


def _inv_norm(p: float) -> float:
    """Acklam's inverse standard-normal CDF approximation (same as governance_v6)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def hoeffding_radius(n: int, total_n: int) -> float:
    """One-sided Hoeffding confidence radius for a [0,1] mean after n pulls, with the
    UCB1 schedule (confidence tightens like total_n^-4 so the union bound holds):

        radius = sqrt( 2 * ln(total_n) / n ).

    This IS the UCB1 exploration bonus (Auer 2002). Unseen arm (n<=0) -> +inf."""
    if n <= 0:
        return math.inf
    t = max(total_n, 2)  # ln undefined/negative for t<=1; clamp to keep bonus >=0
    return math.sqrt(2.0 * math.log(t) / n)


def wilson_interval(successes: float, n: int, conf: float = DEFAULT_CONFIDENCE) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion -- the correct small-sample
    interval (never escapes [0,1], unlike the normal approximation). Returns (lo, hi).
    Used to compute a LOWER confidence bound on an agent's success rate so we only
    declare 'proven' on EVIDENCE, not on an optimistic point estimate."""
    if n <= 0:
        return (0.0, 1.0)
    z = _z_for(conf)
    phat = min(max(successes / n, 0.0), 1.0)
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _kl_bernoulli(p: float, q: float) -> float:
    """KL divergence D(Bern(p) || Bern(q)). Used by KL-UCB."""
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


# ---------------------------------------------------------------------------
# UCB family
# ---------------------------------------------------------------------------
def ucb_score(mean: float, n: int, total_n: int, c: float = UCB1_C) -> float:
    """UCB1 index for one arm (Auer, Cesa-Bianchi & Fischer 2002).

        UCB1(i) = mean_i + c * sqrt( ln(total_n) / n_i )

    mean    : empirical mean reward in [0,1] for this arm.
    n       : number of times this arm was pulled.
    total_n : total pulls across all arms so far.
    c       : exploration coefficient (sqrt(2) -> the proven logarithmic-regret bound;
              the radius = (c/sqrt(2)) * hoeffding_radius).

    An UNSEEN arm (n<=0) returns +inf so every arm is tried once before exploitation --
    this is what makes UCB pick the high-uncertainty arm EARLY, then exploit as the
    bonus shrinks like 1/sqrt(n)."""
    if n <= 0:
        return math.inf
    t = max(total_n, 2)
    bonus = c * math.sqrt(math.log(t) / n)
    return mean + bonus


def kl_ucb_score(successes: float, n: int, total_n: int, c: float = 3.0,
                 tol: float = 1e-6) -> float:
    """KL-UCB index (Garivier & Cappe 2011): the largest q in [mean, 1] such that

        n * D(mean || q) <= ln(total_n) + c * ln(ln(total_n)).

    Solved by closed-form bisection on the convex Chernoff information. Tighter than
    UCB1 for Bernoulli rewards (asymptotically optimal) yet still deterministic and
    cheap. Unseen arm -> 1.0 (max optimism)."""
    if n <= 0:
        return 1.0
    t = max(total_n, 3)  # need ln(ln(t)) defined and >0 -> t >= e ~ 2.72
    mean = min(max(successes / n, 0.0), 1.0)
    rhs = (math.log(t) + c * math.log(max(math.log(t), 1e-9))) / n
    if mean >= 1.0 - 1e-12:
        return 1.0
    lo, hi = mean, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _kl_bernoulli(mean, mid) > rhs:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def regret_bound_ucb1(gaps: Sequence[float], horizon: int, c: float = UCB1_C) -> float:
    """Auer et al. (2002) finite-horizon UCB1 regret upper bound (a control-stability
    guarantee on the exploration budget):

        E[Regret(T)] <= sum_{i: gap_i>0} ( (8 ln T) / gap_i ) + (1 + pi^2/3) * sum gap_i

    gaps    : sub-optimality gaps Delta_i = mu* - mu_i (>=0) for each non-optimal arm.
    horizon : T, the number of exploration pulls budgeted.
    Returns the regret ceiling -- O(ln T), i.e. exploration cost grows only
    logarithmically. (The 8 constant is exact for c = sqrt(2).)"""
    T = max(int(horizon), 1)
    pos = [g for g in gaps if g and g > 0.0]
    if not pos:
        return 0.0
    coeff = 8.0 * (c * c / 2.0)  # 8 at c=sqrt(2); scales with the bonus
    main = sum(coeff * math.log(T) / g for g in pos)
    const = (1.0 + (math.pi ** 2) / 3.0) * sum(pos)
    return main + const


# ---------------------------------------------------------------------------
# Sandbox model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SandboxAllocation:
    agent_id: str
    lane: str
    ucb: float
    mean: float
    n: int
    lane_risk: float
    note: str


@dataclass(frozen=True)
class SandboxDecision:
    picks: list[SandboxAllocation]
    budget: int
    spent: int
    rejected_high_risk: list[str] = field(default_factory=list)
    rejected_proven: list[str] = field(default_factory=list)
    rejected_no_lane: list[str] = field(default_factory=list)
    note: str = ""

    def picked_lanes(self) -> set[str]:
        return {p.lane for p in self.picks}

    def picked_agents(self) -> set[str]:
        return {p.agent_id for p in self.picks}


def _num(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _lane_tail_risk(lane: dict, risk_caps: dict | None) -> float:
    """Worst-case tail-risk level of a lane in {money,security,live}, in [0,5].

    Resolution order (most authoritative first):
      1. explicit per-dimension fields money_risk/security_risk/live_risk on the lane,
      2. a single 'tail_risk' / 'risk' scalar,
      3. a name lookup in risk_caps['lane_risk'] (operator override table),
      4. otherwise CRITICAL (fail-closed): an unclassified lane is treated as money-grade
         so the sandbox NEVER leaks into an unlabeled lane."""
    explicit = [lane.get("money_risk"), lane.get("security_risk"), lane.get("live_risk")]
    vals = [_num(x, -1.0) for x in explicit]
    vals = [v for v in vals if v >= 0.0]
    if vals:
        return max(0.0, min(5.0, max(vals)))
    for key in ("tail_risk", "risk", "risk_level"):
        if key in lane:
            return max(0.0, min(5.0, _num(lane.get(key), CRITICAL_RISK_LEVEL)))
    name = str(lane.get("lane") or lane.get("name") or lane.get("domain") or "")
    table = (risk_caps or {}).get("lane_risk") or {}
    if name in table:
        return max(0.0, min(5.0, _num(table.get(name), CRITICAL_RISK_LEVEL)))
    return CRITICAL_RISK_LEVEL  # fail-closed


def _lane_is_sandbox_safe(lane_risk: float, risk_caps: dict | None) -> bool:
    """A lane is sandbox-safe iff its tail risk is at/under the cap. Cap defaults to
    SANDBOX_MAX_LANE_RISK (=2, strictly below the critical line of 3) and can only be
    LOWERED by risk_caps, never raised above the critical line."""
    cap = SANDBOX_MAX_LANE_RISK
    if risk_caps and "max_lane_risk" in risk_caps:
        cap = min(cap, _num(risk_caps.get("max_lane_risk"), SANDBOX_MAX_LANE_RISK))
    # Hard ceiling: never permit a critical lane regardless of caps.
    cap = min(cap, CRITICAL_RISK_LEVEL - 1e-9)
    return lane_risk <= cap


def is_proven(agent: dict) -> bool:
    """An agent is 'proven' (graduated out of the sandbox) when the LOWER Wilson bound on
    its success rate clears PROVEN_LCB_FLOOR with >= PROVEN_MIN_SAMPLES of evidence.
    Same evidence-first philosophy as the V6 P0-1 fix (no credit on synthetic defaults)."""
    n = int(_num(agent.get("n"), 0))
    if n < PROVEN_MIN_SAMPLES:
        return False
    succ = agent.get("successes")
    if succ is None:
        succ = _num(agent.get("mean"), 0.0) * n
    lo, _ = wilson_interval(_num(succ, 0.0), n)
    return lo >= PROVEN_LCB_FLOOR


def is_high_risk_agent_request(agent: dict) -> bool:
    """True if the agent is REQUESTING (or carrying) a money/security/live-critical
    assignment -- such an agent must never be routed via the sandbox. Reuses the V5/V6
    critical-risk threshold (>=3)."""
    for k in ("money_risk", "security_risk", "live_risk", "requested_lane_risk"):
        if _num(agent.get(k), 0.0) >= CRITICAL_RISK_LEVEL:
            return True
    return False


def graduation_ready(agent: dict, fleet_median_mu: float, min_tasks: int = 20,
                     base_rate: float = 0.0) -> dict:
    """V6 promotion gate for a sandbox-surfaced agent:
        'reach > median mu within 20 tasks at > 2x random base rate'.

    Uses a Beta(1,1)-posterior mean (method-of-moments on Bernoulli outcomes, i.e. the
    Laplace-smoothed success rate) as the agent's measured quality, and the Wilson LCB to
    require the >2x-base-rate lift to hold with confidence (not just in the point estimate).
    Returns a dict verdict + the numbers behind it."""
    n = int(_num(agent.get("n"), 0))
    succ = agent.get("successes")
    if succ is None:
        succ = _num(agent.get("mean"), 0.0) * n
    succ = _num(succ, 0.0)
    # Beta(1,1) posterior mean = (s+1)/(n+2) -- Laplace / MoM smoothing.
    posterior_mean = (succ + 1.0) / (n + 2.0)
    measured_mu = _num(agent.get("mu"), posterior_mean * 100.0)  # mu is on a 0-100 scale
    lo, _ = wilson_interval(succ, n) if n > 0 else (0.0, 1.0)
    beats_median = measured_mu > fleet_median_mu and n >= min_tasks
    lift_ok = (base_rate <= 0.0) or (lo > 2.0 * base_rate)
    return {
        "ready": bool(beats_median and lift_ok),
        "posterior_mean": posterior_mean,
        "measured_mu": measured_mu,
        "success_lcb": lo,
        "fleet_median_mu": fleet_median_mu,
        "n": n,
        "beats_median": beats_median,
        "lift_over_base_ok": lift_ok,
    }


def sandbox_pick(candidates: Sequence[dict], budget: int | float,
                 risk_caps: dict | None = None,
                 c: float = UCB1_C) -> SandboxDecision:
    """Allocate a bounded exploration budget to UNPROVEN agents on LOW-tail-risk lanes only.

    candidates : list of dicts, each:
        {
          "agent_id": str,
          "n":  int,            # times this (agent,lane) pair was explored
          "mean": float,        # empirical reward in [0,1]  (or "successes": float)
          "lane": dict | str,   # the lane being requested; risk read via _lane_tail_risk
          # optional: "successes", money_risk/security_risk/live_risk, mu, requested_lane_risk
        }
    budget     : exploration slots. If a float in (0,1], treated as a FRACTION of the
                 candidate pool (~5% mandate default) -> floor to an int (>=1 if any).
    risk_caps  : {"max_lane_risk": <=2, "lane_risk": {name: level}, ...} -- can only
                 TIGHTEN the default caps, never loosen past the critical line.

    HARD invariants (enforced, tested):
      * No money/security/live (risk>=3) lane is EVER selected -- fail-closed on unlabeled lanes.
      * No agent requesting a critical assignment is selected.
      * Already-PROVEN agents are excluded (the budget is for the UNPROVEN).
      * len(picks) <= budget  AND  each agent appears at most once.

    Selection rule: among the eligible unproven candidates, rank by UCB1 index
    (ucb_score) so that high-uncertainty arms (unseen / few pulls) are explored EARLY,
    and as evidence accrues the bonus decays like 1/sqrt(n) and the ranking tilts toward
    the higher-mean arms (exploit). Ties broken deterministically by agent_id."""
    cands = list(candidates)
    # Resolve a fractional budget against the *pool* size (mandate ~5%).
    if isinstance(budget, float) and 0.0 < budget <= 1.0:
        k = int(math.floor(budget * len(cands)))
        budget_int = max(1, k) if cands else 0
    else:
        budget_int = max(0, int(budget))

    rejected_high_risk: list[str] = []
    rejected_proven: list[str] = []
    rejected_no_lane: list[str] = []

    # total_n for the UCB bonus = total exploration pulls observed across candidates,
    # +1 so a cold start (all n=0) still yields ln(total_n)>=0 and finite bonuses where n>0.
    total_n = sum(max(0, int(_num(cd.get("n"), 0))) for cd in cands) + 1

    scored: list[tuple[float, str, SandboxAllocation]] = []
    for cd in cands:
        aid = str(cd.get("agent_id") or "?")
        lane_obj = cd.get("lane")
        lane = lane_obj if isinstance(lane_obj, dict) else {"lane": str(lane_obj or "")}
        lane_name = str(lane.get("lane") or lane.get("name") or lane.get("domain") or lane_obj or "")
        lane_risk = _lane_tail_risk(lane, risk_caps)

        # (1) HARD: agent must not be requesting a critical assignment.
        if is_high_risk_agent_request(cd):
            rejected_high_risk.append(aid)
            continue
        # (2) HARD: lane must be sandbox-safe (low tail risk, fail-closed).
        if not _lane_is_sandbox_safe(lane_risk, risk_caps):
            rejected_high_risk.append(aid)
            continue
        # (3) the budget is for the UNPROVEN -- skip graduated agents.
        if is_proven(cd):
            rejected_proven.append(aid)
            continue

        n = max(0, int(_num(cd.get("n"), 0)))
        succ = cd.get("successes")
        if succ is not None and n > 0:
            mean = min(max(_num(succ, 0.0) / n, 0.0), 1.0)
        else:
            mean = min(max(_num(cd.get("mean"), 0.0), 0.0), 1.0)
        ucb = ucb_score(mean, n, total_n, c=c)
        note = "cold-start (n=0): forced explore" if n == 0 else "ucb"
        scored.append((ucb, aid, SandboxAllocation(
            agent_id=aid, lane=lane_name, ucb=ucb, mean=mean, n=n,
            lane_risk=lane_risk, note=note)))

    # Deterministic ranking: UCB desc, then agent_id asc (stable across runs).
    # +inf (unseen) sort first -> high-uncertainty arms explored early.
    scored.sort(key=lambda row: (-(row[0] if math.isfinite(row[0]) else math.inf),
                                 row[1]) if math.isfinite(row[0]) else (-math.inf, row[1]))
    # The lambda above keeps +inf ahead of any finite score while staying total-order safe:
    scored.sort(key=lambda row: (0 if math.isinf(row[0]) else 1, -row[0]
                                 if math.isfinite(row[0]) else 0.0, row[1]))

    picks: list[SandboxAllocation] = []
    seen: set[str] = set()
    for _ucb, aid, alloc in scored:
        if len(picks) >= budget_int:
            break
        if aid in seen:
            continue
        seen.add(aid)
        picks.append(alloc)

    return SandboxDecision(
        picks=picks,
        budget=budget_int,
        spent=len(picks),
        rejected_high_risk=rejected_high_risk,
        rejected_proven=rejected_proven,
        rejected_no_lane=rejected_no_lane,
        note=(f"explored {len(picks)}/{budget_int} slots on low-risk lanes; "
              f"blocked {len(rejected_high_risk)} high-risk, "
              f"skipped {len(rejected_proven)} proven"),
    )


__all__ = [
    "ucb_score", "kl_ucb_score", "regret_bound_ucb1",
    "hoeffding_radius", "wilson_interval",
    "sandbox_pick", "SandboxDecision", "SandboxAllocation",
    "is_proven", "is_high_risk_agent_request", "graduation_ready",
    "SANDBOX_MAX_LANE_RISK", "CRITICAL_RISK_LEVEL", "UCB1_C",
]


if __name__ == "__main__":
    # Read-only demo; no Mongo / no fleet side effects.
    print("UCB unseen arm           :", ucb_score(0.0, 0, 10))
    print("UCB seen (0.5, n=4, T=50):", round(ucb_score(0.5, 4, 50), 4))
    print("KL-UCB (3/4, T=50)       :", round(kl_ucb_score(3, 4, 50), 4))
    print("Hoeffding radius (4,50)  :", round(hoeffding_radius(4, 50), 4))
    print("Wilson(7/8)              :", tuple(round(x, 4) for x in wilson_interval(7, 8)))
    print("UCB1 regret bound        :", round(regret_bound_ucb1([0.3, 0.1], 1000), 2))
    demo = [
        {"agent_id": "new_A", "n": 0, "mean": 0.0, "lane": {"lane": "DesignDocs", "tail_risk": 1}},
        {"agent_id": "new_B", "n": 2, "mean": 0.5, "lane": {"lane": "Reporting", "tail_risk": 1}},
        {"agent_id": "money_C", "n": 0, "mean": 0.0, "lane": {"lane": "Risk", "money_risk": 4}},
    ]
    d = sandbox_pick(demo, budget=2)
    print("sandbox picks            :", [(p.agent_id, p.lane) for p in d.picks], "|", d.note)
