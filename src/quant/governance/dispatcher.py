"""Risk-adjusted auction dispatcher for QuantChoAI bridge workers.

HP remains a survival valve. Routing is driven by expected value:
ticket value, Bayesian domain skill, model cost, tail risk, context bloat,
and host disk pressure.
"""
from __future__ import annotations

import datetime as _dt
import functools
import math
import os
import re
import shutil
import threading
import time as _time
from dataclasses import dataclass, replace
from typing import Any

from pymongo import ASCENDING, ReturnDocument

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional host telemetry
    psutil = None

# CVaR UNIFY (peer point 1): the live tail term must be priced on a coherent
# Expected-Shortfall estimate, NOT the crude exp(money+sec+live) proxy. Pull the
# hardened tail-risk layer (deterministic, no LLM/Mongo) into the hot path. We keep
# governance_v6.cvar as the historical fallback in case the modelver layer is absent.
try:
    from quant.governance import modelver_cvar_budget as _cvar_budget  # type: ignore
except Exception:  # pragma: no cover - module must always be present in this workspace
    _cvar_budget = None
try:
    from quant.governance import governance_v6 as _gov6  # type: ignore
except Exception:  # pragma: no cover
    _gov6 = None


Q = "agent_queue"
L = "agent_ledger"
LOG = "system_logs"

DISK_YELLOW_GB = float(os.environ.get("QUANTCHO_DISK_YELLOW_GB", "10.0"))
DISK_CRITICAL_GB = float(os.environ.get("QUANTCHO_DISK_CRITICAL_GB", "5.0"))
DISK_CRITICAL_BYTES = int(DISK_CRITICAL_GB * 1024 * 1024 * 1024)
AUCTION_ACTIVE_MIN = int(os.environ.get("QUANTCHO_AUCTION_ACTIVE_MIN", "90"))
AUCTION_CANDIDATE_LIMIT = int(os.environ.get("QUANTCHO_AUCTION_CANDIDATE_LIMIT", "200"))
AUCTION_CYCLE_SEC = int(os.environ.get("QUANTCHO_AUCTION_CYCLE_SEC", "60"))
STARVATION_BYPASS_CYCLES = int(os.environ.get("QUANTCHO_STARVATION_BYPASS_CYCLES", "10"))
CPU_HOT_PERCENT = float(os.environ.get("QUANTCHO_CPU_HOT_PERCENT", "85.0"))
WARN_LOG_THROTTLE_SEC = float(os.environ.get("QUANTCHO_WARN_LOG_THROTTLE_SEC", "300"))

LAMBDA = float(os.environ.get("QUANTCHO_AUCTION_LAMBDA", "0.018"))
GAMMA = float(os.environ.get("QUANTCHO_AUCTION_GAMMA", "0.025"))
ALPHA = float(os.environ.get("QUANTCHO_AUCTION_ALPHA", "3.0"))
BETA = float(os.environ.get("QUANTCHO_AUCTION_BETA", "0.0015"))
THETA = float(os.environ.get("QUANTCHO_AUCTION_THETA", "18.0"))

MIN_POSITIVE_UTILITY = float(os.environ.get("QUANTCHO_AUCTION_MIN_UTILITY", "0.0"))
LOW_CONFIDENCE_MIN_P = float(os.environ.get("QUANTCHO_AUCTION_MIN_RISK_P", "0.72"))
CRITICAL_CONFIDENCE_MIN_P = float(os.environ.get("QUANTCHO_AUCTION_CRITICAL_RISK_P", "0.86"))

# CVaR UNIFY (peer point 1) -------------------------------------------------------
# Confidence level the live tail term is priced at. 0.95 matches the shadow-scorer
# backtest (governance_v6.cvar / modelver_cvar_budget.DEFAULT_ALPHA) so the LIVE
# auction and the SHADOW budget speak the same Expected-Shortfall units.
CVAR_ALPHA = float(os.environ.get("QUANTCHO_AUCTION_CVAR_ALPHA", "0.95"))
# Calm base mass: the bulk of a ticket's loss outcomes are small. We seed the loss
# vector with CVAR_CALM_MASS draws of CVAR_CALM_LOSS so the tail (the exp(risk) legs)
# is measured AGAINST a realistic body, not in isolation. This is what makes CVaR a
# tail-of-a-distribution number rather than a relabeled max(). Deterministic.
CVAR_CALM_MASS = int(os.environ.get("QUANTCHO_AUCTION_CVAR_CALM_MASS", "99"))
CVAR_CALM_LOSS = float(os.environ.get("QUANTCHO_AUCTION_CVAR_CALM_LOSS", "1.0"))

# GOODHART HARD-COST (peer point 2) ----------------------------------------------
# A NAMED hard cost (charged in utility units AND in ledger credit) so "decline the
# hard eligible task" and "emit safe low-information garbage" are never dominant
# strategies. These mirror governance_v6.value_adjustment's intent but make the cost
# an explicit, testable term on the LIVE auction objective rather than a post-hoc HP
# event. DECLINE is charged as a fraction of the value forgone; LOW-INFO is charged
# as a fraction of the value that an empty deliverable falsely claims.
GOODHART_DECLINE_COST_FRAC = float(os.environ.get("QUANTCHO_GOODHART_DECLINE_FRAC", "0.5"))
GOODHART_LOWINFO_COST_FRAC = float(os.environ.get("QUANTCHO_GOODHART_LOWINFO_FRAC", "0.6"))
# An output whose information content is below this is "safe garbage".
GOODHART_LOWINFO_THRESHOLD = float(os.environ.get("QUANTCHO_GOODHART_LOWINFO_THRESHOLD", "0.2"))

# MARGINAL-CVaR + VENDOR-CONCENTRATION (roadmap item 9) --------------------------
# PROBLEM: the live auction prices each ticket's tail risk MARGINALLY/INDEPENDENTLY.
# Vendor-correlated failure (shared model / shared infra) is modeled in governance_v6
# (factor_decomposition / concentration_check / frn_barrier) but NOT priced live, so the
# fleet can pile critical work onto one vendor with no live penalty. FIX: add the
# assignment's MARGINAL contribution to FLEET CVaR given the current in-flight vendor
# correlation (Euler / marginal-CVaR allocation), and enforce the concentration cap as a
# live eligibility constraint.
#
# MATH (the spec's correlated-tail model, NOT reinvented): for k positions on one vendor at
# pairwise correlation rho, the correlated tail magnitude scales as
#     sqrt(k + k*(k-1)*rho)          (vs the independent sqrt(k)).
# Adding the (k+1)-th position raises the vendor's correlated-tail scale from
#     T(k)   = sqrt(k   + k*(k-1)*rho)        to
#     T(k+1) = sqrt(k+1 + (k+1)*k*rho).
# The MARGINAL (Euler) contribution of the new assignment is the INCREMENT T(k+1) - T(k),
# expressed as a multiplier on the ticket's own tail term. At k=0 (vendor empty) the
# multiplier is exactly 1.0 (independent pricing recovers); it rises monotonically as the
# vendor's in-flight share grows, so piling tickets on one vendor costs strictly more live.
#
# FLAG: gated behind QUANTCHO_MARGINAL_CVAR_ENABLED (default ON, shadow-compare). When OFF
# the independent per-ticket pricing is the fallback and the concentration cap is not
# enforced live -- byte-identical to the pre-wire behaviour.
MARGINAL_CVAR_ENABLED = os.environ.get("QUANTCHO_MARGINAL_CVAR_ENABLED", "TRUE").upper() == "TRUE"
# Pairwise vendor failure correlation (shared model / RLHF / infra). The factor model
# (modelver_correlation_regime) estimates this from the trouble matrix in the slow loop;
# here it is the live default the marginal-CVaR allocation prices on. rho in [0,1].
VENDOR_CORR_RHO = float(os.environ.get("QUANTCHO_VENDOR_CORR_RHO", "0.40"))
# Concentration cap: max fleet share of in-flight (incl. the prospective) critical work on
# any single vendor. Mirrors governance_v6.concentration_check's 0.40 default. An assignment
# that would push a vendor's share strictly above this is INELIGIBLE (live hard constraint).
VENDOR_CONCENTRATION_CAP = float(os.environ.get("QUANTCHO_VENDOR_CONCENTRATION_CAP", "0.40"))
# Concentration is a FLEET-correlation control on critical work. A bare-keyword doc ticket
# is not critical, so the cap is enforced only for genuine money/security-critical tickets
# (the same narrow gate the rest of the auction uses), and only when >= this many positions
# are already in flight fleet-wide (a 1-of-1 assignment trivially has share 1.0 and must not
# be blocked -- there is nothing to concentrate against yet).
VENDOR_CONCENTRATION_MIN_FLEET = int(os.environ.get("QUANTCHO_VENDOR_CONCENTRATION_MIN_FLEET", "3"))

ALL_DOMAINS = [
    "Analytics", "Trading", "Execution", "Pricing", "Risk", "Reporting",
    "Accounting", "Tax", "Ops", "Compliance", "Infra", "FrontOffice",
    "Integration", "DesignDocs", "Committee", "Governance", "PM", "Chaos",
]

MONEY_RE = re.compile(
    r"\b(var|cvar|pnl|p&l|nav|gross mv|market value|pricing|valuation|"
    r"position|positions|t9|reseed|margin|exposure|risk|tax|accounting|"
    r"ledger|fill|trade repository|traderepository|t3)\b",
    re.I,
)
SECURITY_RE = re.compile(
    r"\b(auth|tenant|permission|rbac|security|compliance_token|token|"
    r"ofac|sanction|zero trust|isolation|validator|schema validator)\b",
    re.I,
)
LIVE_RE = re.compile(
    r"\b(real money|live broker|production broker|submit order|place order|"
    r"execution api|alpaca live|interactive brokers live|kill switch)\b",
    re.I,
)
REAL_ADAPTER_RE = re.compile(
    r"\b(real money|live broker|production broker|alpaca live|ibkr live|"
    r"interactive brokers live|broker adapter|execution adapter|place order|submit order)\b",
    re.I,
)
PAPER_RE = re.compile(r"\b(paper|uat|sandbox|simulated|dry-run|dry run|demo)\b", re.I)
ELITE_ONLY_DOMAINS = {"money_math", "committee_judgment", "core_dev"}

# --- V6 narrowed money-critical classification --------------------------------
# Validated 2026-06-03: cuts auction over-blocking from ~47% to ~6% while still
# gating genuine money-math (see bridge/SHADOW_SCORER_FINDINGS_20260603.md).
# A ticket is money/security-critical ONLY for a real mutation / live / protected-path
# action -- NOT a bare keyword mention in a doc/summary/triage ticket.
_MUT_RE = re.compile(r"\b(insert|write|writes|writing|written|merge|merged|reseed|submit|place\s*order|apply|mutate|mutated|settle|settled|deploy|migrat)\w*", re.I)
_MONEY_NOUN_RE = re.compile(r"\b(var|cvar|pnl|p&l|nav|gross mv|market value|position|positions|margin|exposure|ledger|trade|fill|valuation|pricing|cost basis|1099|wash|tax)\b", re.I)
_PROTECTED_RE = re.compile(r"\b(risk\.py|positions|trade repository|traderepository|t3|t9|broker adapter|execution adapter|kill switch|place order|submit order)\b", re.I)


def _explicit_risk_field(ticket: dict, key: str) -> float | None:
    """Single source of truth for reading an explicit risk field (P1-3 unification).

    Precedence: ticket-level first, then payload. A field that is ABSENT *or present
    but None* is treated as "not supplied" (returns None) so the caller falls through
    to keyword inference. A present numeric/coercible value is clamped to [0, 5].

    Both money classifiers (`_is_genuine_money_critical` and `_risk_level`) now route
    every explicit-field read through here, so a present-but-null field can no longer
    disable the keyword fallback in one path while leaving it active in the other.
    """
    for src in (ticket, ticket.get("payload")):
        if not isinstance(src, dict):
            continue
        if key not in src:
            continue
        v = src.get(key)
        if v is None:
            # present-but-null -> treat as absent, but keep scanning the other source
            continue
        try:
            return max(0.0, min(5.0, float(v)))
        except (TypeError, ValueError):
            # present-but-non-numeric -> treat as absent for this source
            continue
    return None


def _is_genuine_money_critical(ticket: dict) -> bool:
    """Narrow gate: explicit risk field >=3, OR a mutation verb co-located with a money
    noun, OR a protected path/control. Bare keyword mentions do NOT qualify."""
    for k in ("money_risk", "security_risk", "live_risk"):
        v = _explicit_risk_field(ticket, k)
        if v is not None and v >= 3.0:
            return True
    text = _doc_text(ticket)
    if _PROTECTED_RE.search(text):
        return True
    return bool(_MUT_RE.search(text) and _MONEY_NOUN_RE.search(text))


_LAST_LOCAL_LOG: dict[tuple[str, str, str | None], float] = {}

# ROSTER TTL CACHE (efficiency 2026-06-04) ---------------------------------------
# `_recent_active_agents` fires a full agent_ledger scan (+ a psutil.pid_exists probe
# per row) on EVERY auction_pick call. The recently-active roster changes only when an
# agent registers (init_agent) or is reaped (guillotine) -- far less often than the
# auction polls. We cache (result, computed_at, current_agent_id) for a short TTL of
# AUCTION_CYCLE_SEC/2 so a burst of polls within one cycle reuses one scan, while a
# stale roster can never linger longer than half a cycle. The cache is keyed on the
# CALLING agent_id (the ledger query and the "always include self" branch are
# caller-specific), so two different callers never see each other's roster. init_agent
# and guillotine call invalidate_roster_cache() to drop the entry immediately on a
# membership change. Lock-guarded for the multi-threaded poller; deterministic.
_ROSTER_CACHE_LOCK = threading.Lock()
# agent_id -> (agents_list, computed_at_monotonic)
_ROSTER_CACHE: dict[str, tuple[list[dict], float]] = {}


def _roster_ttl_sec() -> float:
    return max(0.0, AUCTION_CYCLE_SEC / 2.0)


def invalidate_roster_cache(agent_id: str | None = None) -> None:
    """Drop the recently-active-agents TTL cache.

    Called by ledger.init_agent / guillotine on a roster membership change so a freshly
    registered or reaped agent is reflected immediately rather than after the TTL. With
    no agent_id the whole cache is cleared (membership changes are global to the ledger);
    a specific agent_id drops only that caller's entry. Safe to call when caching is off.
    """
    with _ROSTER_CACHE_LOCK:
        if agent_id is None:
            _ROSTER_CACHE.clear()
        else:
            _ROSTER_CACHE.pop(agent_id, None)

SKILL_BY_DOMAIN = {
    "Risk": "risk_math_skill",
    "Pricing": "risk_math_skill",
    "Analytics": "risk_math_skill",
    "Trading": "execution_skill",
    "Execution": "execution_skill",
    "Compliance": "security_skill",
    "Infra": "security_skill",
    "Ops": "security_skill",
    "FrontOffice": "frontend_skill",
    "Reporting": "frontend_skill",
    "Committee": "committee_judgment",
    "Governance": "committee_judgment",
    "Chaos": "chaos_reproduction",
}


@dataclass(frozen=True)
class TicketFeatures:
    business_impact: float
    urgency: float
    downstream_blocked_count: int
    age_hours: float
    money_risk: float
    security_risk: float
    live_risk: float
    asset_loss_score: float
    difficulty: float
    failed_requeue_count: int
    traceback_len: int
    auction_deadlock_cycles: int
    starvation_factor: float
    value: float
    risk_skew: float


@dataclass(frozen=True)
class UtilityResult:
    utility: float
    p_success: float
    mu: float
    sigma: float
    model_cost: float
    risk_penalty: float
    context_penalty: float
    disk_penalty: float
    reason: str
    features: TicketFeatures
    # GOODHART HARD-COST (peer point 2): the named hard cost charged to the auction
    # objective when the scored ACTION is a decline of an eligible task or a low-info
    # ("safe garbage") output. 0.0 for an honest full-effort assignment. Defaulted so
    # every existing UtilityResult(...) construction site stays valid.
    goodhart_cost: float = 0.0
    # MARGINAL-CVaR (roadmap item 9): the vendor this assignment loads, the marginal-CVaR
    # multiplier applied to the tail term (1.0 == independent pricing / vendor empty / flag
    # off), and the prospective fleet share this assignment would give the vendor. Defaulted
    # so every existing UtilityResult(...) construction site stays valid.
    vendor: str = ""
    marginal_cvar_mult: float = 1.0
    vendor_share: float = 0.0


def _now() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


def _monotonic() -> float:
    """Monotonic clock for cache TTLs -- immune to wall-clock adjustments. Indirected
    so tests can patch the time source deterministically."""
    return _time.monotonic()


def _parse_ts(value: Any) -> _dt.datetime | None:
    if isinstance(value, _dt.datetime):
        return value.replace(tzinfo=None)
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _doc_text(ticket: dict) -> str:
    payload = ticket.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {"desc": str(payload)}
    parts = [
        str(ticket.get("ticket_id", "")),
        str(ticket.get("target_domain", "")),
        str(ticket.get("task_type", "")),
        str(payload.get("desc", "")),
        str(payload.get("expected", "")),
    ]
    return "\n".join(parts)


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _priority(ticket: dict) -> int:
    return max(0, min(5, _int(ticket.get("priority", 5), 5)))


def disk_free_gb(path: str = "C:\\") -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def disk_free_bytes(path: str = "C:\\") -> int:
    return shutil.disk_usage(path).free


def cpu_hot() -> tuple[bool, float | None]:
    if psutil is None:
        return False, None
    try:
        pct = float(psutil.cpu_percent(interval=0.0))
        return pct > CPU_HOT_PERCENT, pct
    except Exception:
        return False, None


def trigger_reaper_hard_cleanup(db, reason: str = "disk_lockout") -> dict:
    """Minimal deterministic cleanup hook used before dispatch freezes.

    It avoids personal files and live worktrees: reclaim orphaned queue work and
    remove only Python cache directories inside the bridge folder.
    """
    result: dict[str, Any] = {"reason": reason, "reaper": None, "cache_dirs_removed": 0}
    try:
        import ticket_queue
        result["reaper"] = ticket_queue.reap_stale(timeout_min=int(os.environ.get("QUANTCHO_ORPHAN_MIN", "20")))
    except Exception as exc:
        result["reaper_error"] = repr(exc)
    try:
        bridge_dir = os.path.dirname(__file__)
        for root, dirs, _ in os.walk(bridge_dir):
            for name in list(dirs):
                if name != "__pycache__":
                    continue
                path = os.path.join(root, name)
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    result["cache_dirs_removed"] += 1
                except Exception:
                    pass
    except Exception as exc:
        result["cache_error"] = repr(exc)
    try:
        db[LOG].insert_one({
            "ts": _now(),
            "agent_id": "dispatcher",
            "event": "trigger_reaper_hard_cleanup",
            "status": "DONE",
            "level": "WARN",
            "msg": str(result)[:500],
        })
    except Exception:
        pass
    return result


def _business_impact_default(ticket: dict, text: str) -> float:
    priority = _priority(ticket)
    base = {0: 120.0, 1: 85.0, 2: 48.0, 3: 22.0, 4: 7.0, 5: 3.0}.get(priority, 3.0)
    if MONEY_RE.search(text):
        base *= 1.45
    if SECURITY_RE.search(text):
        base *= 1.25
    if str(ticket.get("target_domain")) in {"DesignDocs", "PM"}:
        base *= 0.55
    return base


def _urgency_default(ticket: dict, text: str) -> float:
    priority = _priority(ticket)
    urgency = {0: 5.0, 1: 3.8, 2: 2.4, 3: 1.35, 4: 0.65, 5: 0.35}.get(priority, 0.35)
    if "P0" in text or "catastrophic" in text.lower():
        urgency *= 1.35
    return urgency


def _risk_level(ticket: dict, key: str, text: str, regex: re.Pattern[str]) -> float:
    # P1-3 fix (2026-06-04): route the explicit-field read through the SAME helper that
    # `_is_genuine_money_critical` uses. The old `if key in ticket` test honored a
    # present-but-None field (returning 0.0 and SHORT-CIRCUITING the keyword fallback),
    # while the genuine-critical path treated None as absent -> the two classifiers
    # disagreed on a null field. `_explicit_risk_field` returns None for absent OR null,
    # so a null field now falls through to keyword inference in BOTH paths.
    explicit = _explicit_risk_field(ticket, key)
    if explicit is not None:
        return explicit
    # V6 narrowed: keyword inference fires only for genuine money-critical tickets,
    # so a doc that merely MENTIONS accounting/tax/etc is not auto-rated high-risk.
    if key in ("money_risk", "security_risk", "live_risk") and not _is_genuine_money_critical(ticket):
        return 0.0
    if not regex.search(text):
        return 0.0
    if key == "live_risk":
        return 4.5 if not PAPER_RE.search(text) else 1.0
    if key == "money_risk":
        return 4.0 if re.search(r"\b(var|pnl|nav|position|t9|trade|pricing|margin)\b", text, re.I) else 2.5
    if key == "security_risk":
        return 3.5 if re.search(r"\b(token|auth|tenant|validator|permission)\b", text, re.I) else 2.0
    return 0.0


def _risk_loss_vector(money_risk: float, security_risk: float, live_risk: float) -> list[float]:
    """Build the per-ticket LOSS sample whose tail the auction prices.

    The old proxy collapsed the three risk legs into ONE number, exp(m+s+l), and used
    it directly -- a point estimate with no notion of a distribution, so it could not
    distinguish a thin-tailed from a fat-tailed risk profile. Here each active risk leg
    contributes a loss magnitude exp(risk_i) (a leg that is calm contributes nothing
    beyond the base mass), and the legs sit on top of a body of CVAR_CALM_MASS small
    'normal day' outcomes. CVaR_alpha of THIS vector is the coherent tail term.

    Determinism: pure arithmetic on the three scalar risks; no RNG, no I/O.
    """
    losses: list[float] = [CVAR_CALM_LOSS] * max(1, CVAR_CALM_MASS)
    for risk in (money_risk, security_risk, live_risk):
        if risk and risk > 0.0:
            # cap the exponent exactly as the old proxy did (min(10, .)) so a single
            # leg can never dominate beyond the legacy ceiling -- but now it is a TAIL
            # draw, not the whole term.
            losses.append(math.exp(min(10.0, float(risk))))
    return losses


@functools.lru_cache(maxsize=4096)
def _cvar_tail_term(money_risk: float, security_risk: float, live_risk: float) -> float:
    """CVaR/Expected-Shortfall tail term that REPLACES exp(money+sec+live).

    Prices the (1-CVAR_ALPHA) tail of the per-ticket loss vector using the hardened
    Cornish-Fisher Expected Shortfall (skew/kurt aware, works on a small sample, O(n)
    on the tail -- hot-path safe). Falls back to the historical Rockafellar-Uryasev ES
    and finally to governance_v6.cvar so the LIVE path always prices tail risk on an ES
    estimate, never on the crude proxy. A no-risk ticket returns the calm base mass
    (~1.0), preserving the old behaviour that a benign ticket carries ~unit tail weight.

    EFFICIENCY (2026-06-04): this is a PURE function of the three risk legs, each drawn
    from a tiny discrete grid (the explicit-field clamp [0,5] plus the keyword-inferred
    set {0,1,2,2.5,3.5,4,4.5}). The body sorts an ~100-element loss vector and runs
    several O(n) ES passes per (agent,ticket) -- but the SAME three legs recur across
    every candidate agent for a ticket and across cycles. We memoize on the exact float
    triple via functools.lru_cache. The cache is keyed on the RAW floats (no rounding),
    so a hit returns the byte-identical value recomputation would produce -- the test
    asserts this against a fresh, cache-cleared recompute. maxsize=4096 comfortably
    covers the ~343 discrete-grid outcomes plus any explicit-field one-offs; an arbitrary
    explicit float simply misses and recomputes, exactly as before.
    """
    losses = _risk_loss_vector(money_risk, security_risk, live_risk)
    if _cvar_budget is not None:
        try:
            # PRIMARY: historical Rockafellar-Uryasev Expected Shortfall. Coherent,
            # sub-additive, monotone in the tail mass, and the exact estimator the
            # shadow-scorer backtest measured (VaR 21.8 vs CVaR 337). This is the number
            # the live tail term is priced on.
            hist = _cvar_budget.historical_cvar(losses, CVAR_ALPHA)
            # UPLIFT (bounded): the parametric Cornish-Fisher ES captures skew/kurt fat
            # tails the empirical sample may under-resolve. But on a near-degenerate
            # sample (a few huge legs over a calm body) the Edgeworth expansion can blow
            # up and even INVERT monotonicity -- exactly the "too clean to be true"
            # regime the V6 D3 anti-Goodhart discipline rejects. So we only let CF LIFT
            # the historical number, and cap the lift at the historical CVaR itself
            # (i.e. <= 2x hist). This keeps the term monotone in tail mass while still
            # crediting a genuine fat tail.
            cf = _cvar_budget.cornish_fisher_cvar(losses, CVAR_ALPHA)
            cf_uplift = min(max(0.0, cf - hist), hist)
            return max(0.0, hist + cf_uplift)
        except Exception:  # pragma: no cover - defensive; fall through to gov6
            pass
    if _gov6 is not None:
        try:
            return max(0.0, float(_gov6.cvar(losses, CVAR_ALPHA).get("CVaR", 0.0)))
        except Exception:  # pragma: no cover
            pass
    # last-resort: the legacy proxy, so the auction never crashes on a missing module.
    return math.exp(min(10.0, money_risk + security_risk + live_risk))


# MARGINAL-CVaR + VENDOR-CONCENTRATION (roadmap item 9) --------------------------
def _agent_vendor(agent: dict) -> str:
    """Resolve the VENDOR / shared-failure key for an agent.

    Vendor-correlated failure is driven by what the agent SHARES with other agents: the
    same underlying model / policy tier / infra. We read, in precedence order, an explicit
    `vendor`, then `model_id`, then `model_tier` (the customer-selected policy band from
    model_registry), then the rank-derived cost tier as a last-resort coarse bucket. The
    return is a stable string key the in-flight counter and concentration_check group on.
    Deterministic, no I/O.
    """
    for key in ("vendor", "model_id", "model_tier"):
        v = agent.get(key)
        if v:
            return str(v)
    # last-resort coarse bucket: agents with no declared model share the rank cost-tier.
    rank = _agent_rank(agent)
    return f"tier-{rank}"


def _correlated_tail_scale(k: int, rho: float) -> float:
    """Correlated-tail scale for k positions on ONE vendor at pairwise correlation rho.

    The spec's model (reused, NOT reinvented): for k correlated unit positions the tail
    magnitude scales as sqrt(k + k*(k-1)*rho) -- the sqrt of the sum of a kxk correlation
    matrix with unit diagonal and rho off-diagonal. rho=0 collapses to the independent
    sqrt(k); rho=1 gives the fully-correlated k (no diversification). Monotone in both k
    and rho. k<=0 -> 0.0 (an empty book has no tail).
    """
    if k <= 0:
        return 0.0
    r = max(0.0, min(1.0, float(rho)))
    return math.sqrt(k + k * (k - 1) * r)


def _correlated_marginal(k: int, rho: float) -> float:
    """Euler marginal contribution of the (k+1)-th CORRELATED position to the vendor's
    correlated-tail scale: T(k+1) - T(k) with T(n) = sqrt(n + n*(n-1)*rho)."""
    return _correlated_tail_scale(k + 1, rho) - _correlated_tail_scale(max(0, k), rho)


def _independent_marginal(k: int) -> float:
    """Euler marginal contribution of the (k+1)-th INDEPENDENT (rho=0) position to the
    vendor's tail scale: sqrt(k+1) - sqrt(k). The diversified baseline the correlated
    marginal is priced AGAINST."""
    return _correlated_tail_scale(k + 1, 0.0) - _correlated_tail_scale(max(0, k), 0.0)


def _marginal_cvar_multiplier(inflight_k: int, rho: float = VENDOR_CORR_RHO) -> float:
    """Euler / marginal-CVaR multiplier on a ticket's own (independent, unit) tail term.

    `inflight_k` = positions ALREADY in flight on this assignment's vendor (excluding the
    prospective one). The Euler/marginal allocation asks: how much tail does adding THIS
    position to a CORRELATED vendor contribute, relative to adding it to a fully DIVERSIFIED
    (independent) slot? That ratio is the multiplier on the ticket's independent unit tail:

        multiplier = [T_corr(k+1) - T_corr(k)] / [T_indep(k+1) - T_indep(k)]

    with T_corr(n) = sqrt(n + n*(n-1)*rho)  and  T_indep(n) = sqrt(n).

    Why the ratio (not the bare increment): the bare correlated increment T_corr(k+1)-T_corr(k)
    is CONCAVE-decreasing in k (sqrt is concave), so it would PAY a vendor for piling on -- the
    opposite of the intent. The Euler allocation normalizes by the independent marginal, which
    is decreasing at the SAME concave rate; their ratio isolates the EXCESS tail caused purely
    by CORRELATION, which rises monotonically as the vendor's in-flight share grows. Properties:
      * k=0 (vendor empty): T_corr(1)-T_corr(0) = T_indep(1)-T_indep(0) = 1 -> multiplier = 1.0
        EXACTLY. Independent pricing recovers when there is nothing to correlate against.
      * rho=0 (uncorrelated spread): numerator == denominator -> 1.0 at every k. A diversified
        spread is UNPENALIZED.
      * rho>0: multiplier is monotonically NON-DECREASING in k -> piling tickets on one vendor
        raises its marginal-CVaR penalty. This is the central wired behaviour. Deterministic.
    """
    k = max(0, int(inflight_k))
    indep = _independent_marginal(k)
    if indep <= 0.0:  # pragma: no cover - sqrt(k+1)-sqrt(k) > 0 for all k>=0
        return 1.0
    return _correlated_marginal(k, rho) / indep


def _inflight_vendor_counts(db, vendor_counts: dict | None = None) -> dict[str, int]:
    """In-flight assignment count per vendor across the fleet (the live correlation book).

    `vendor_counts` is an explicit override (deterministic, no Mongo) used by callers/tests
    that already hold the book. Otherwise we read the queue: every IN_PROGRESS ticket carries
    the vendor that claimed it (stamped `picked_vendor` at claim time), so a $group/count
    reconstructs the current per-vendor exposure. Missing/absent -> empty book. O(in-flight).
    """
    if vendor_counts is not None:
        return {str(k): int(v) for k, v in vendor_counts.items() if v}
    counts: dict[str, int] = {}
    if db is None:
        return counts
    try:
        cursor = db[Q].find({"status": "IN_PROGRESS"}, {"picked_vendor": 1})
        for row in cursor:
            v = row.get("picked_vendor")
            if v:
                counts[str(v)] = counts.get(str(v), 0) + 1
    except Exception:  # pragma: no cover - defensive; live book unavailable -> price independent
        return {}
    return counts


def _prospective_vendor_share(vendor: str, vendor_counts: dict[str, int]) -> float:
    """Fleet share this vendor WOULD hold if the prospective assignment is granted.

    Reuses governance_v6.concentration_check's definition (count / total) so the live cap
    speaks the SAME units as the shadow concentration model -- we do not reinvent it. The
    prospective assignment is added to BOTH the vendor's count and the fleet total.
    """
    inflight = max(0, int(vendor_counts.get(vendor, 0)))
    total = sum(max(0, int(c)) for c in vendor_counts.values()) + 1  # +1 for the prospective
    return (inflight + 1) / total if total > 0 else 1.0


def _concentration_blocks(vendor: str, vendor_counts: dict[str, int],
                          cap: float = VENDOR_CONCENTRATION_CAP) -> bool:
    """Live concentration-cap eligibility constraint (reuses governance_v6.concentration_check).

    True iff granting the prospective assignment would push `vendor`'s fleet share strictly
    above `cap`. Enforced only once at least VENDOR_CONCENTRATION_MIN_FLEET positions are in
    flight fleet-wide -- a 1-of-1 (or near-empty) fleet trivially has share 1.0 and there is
    nothing to concentrate against, so blocking it would deadlock dispatch.

    Authoritative check: we build the prospective assignment-vendor list (one entry per
    in-flight position + the prospective one) and run governance_v6.concentration_check on
    it, so the LIVE verdict is exactly what the shadow concentration model would report. A
    local share computation is used as the fallback when governance_v6 is unavailable.
    """
    total_inflight = sum(max(0, int(c)) for c in vendor_counts.values())
    if total_inflight + 1 < VENDOR_CONCENTRATION_MIN_FLEET:
        return False
    if _gov6 is not None:
        try:
            assignment_vendors: list[str] = []
            for v, c in vendor_counts.items():
                assignment_vendors.extend([str(v)] * max(0, int(c)))
            assignment_vendors.append(str(vendor))  # the prospective assignment
            res = _gov6.concentration_check(assignment_vendors, cap=cap)
            # concentration_check reports the cap violation for the MAX vendor; the assignment
            # is blocked iff THIS vendor is the violating one (its share strictly exceeds cap).
            frac = res.get("fractions", {}).get(str(vendor), 0.0)
            return bool(frac > cap)
        except Exception:  # pragma: no cover - defensive; fall through to local share
            pass
    return _prospective_vendor_share(vendor, vendor_counts) > cap


def infer_ticket_features(ticket: dict, db=None) -> TicketFeatures:
    text = _doc_text(ticket)
    priority = _priority(ticket)
    enq = _parse_ts(ticket.get("enqueued_at") or ticket.get("created_at"))
    age_hours = 0.0
    if enq:
        age_hours = max(0.0, (_dt.datetime.utcnow() - enq).total_seconds() / 3600.0)

    downstream = _int(ticket.get("downstream_blocked_count"), 0)
    if downstream <= 0 and db is not None:
        tid = ticket.get("ticket_id")
        try:
            downstream = int(db[Q].count_documents({"blocked_by": tid, "status": {"$nin": ["DONE", "FAILED"]}}))
        except Exception:
            downstream = 0

    business_impact = _num(ticket.get("business_impact"), _business_impact_default(ticket, text))
    urgency = _num(ticket.get("urgency"), _urgency_default(ticket, text))
    unblock_factor = 1.0 + math.log1p(max(0, downstream))
    deadlock_cycles = max(0, _int(ticket.get("auction_deadlock_cycles"), 0))
    starvation_delta = max(age_hours, float(deadlock_cycles))
    starvation = math.exp(min(6.0, LAMBDA * starvation_delta))
    value = business_impact * urgency * unblock_factor * starvation

    money_risk = _risk_level(ticket, "money_risk", text, MONEY_RE)
    security_risk = _risk_level(ticket, "security_risk", text, SECURITY_RE)
    live_risk = _risk_level(ticket, "live_risk", text, LIVE_RE)
    # CVaR UNIFY (peer point 1): the tail term is now a coherent Expected-Shortfall of
    # the per-ticket loss distribution, not the crude exp(money+sec+live) proxy. The
    # field name `risk_skew` is kept for snapshot/back-compat; its VALUE is now CVaR.
    risk_skew = _cvar_tail_term(money_risk, security_risk, live_risk)
    asset_loss = _num(ticket.get("asset_loss_score"), 12.0)
    if money_risk >= 3.0:
        asset_loss = max(asset_loss, 500.0)
    if security_risk >= 3.0:
        asset_loss = max(asset_loss, 350.0)
    if live_risk >= 3.0:
        asset_loss = max(asset_loss, 2000.0)

    difficulty = _num(ticket.get("ci_gate_difficulty"), {0: 82.0, 1: 74.0, 2: 62.0, 3: 50.0, 4: 36.0, 5: 28.0}.get(priority, 45.0))
    difficulty += min(18.0, money_risk * 2.5 + security_risk * 2.0 + live_risk * 4.0)
    if str(ticket.get("task_type") or "").lower() in {"epic", "architectural", "architecture"}:
        difficulty += 8.0

    reports = ticket.get("autopsy_reports") or []
    traceback_len = len(str(ticket.get("last_error") or ""))
    for report in reports[-5:]:
        traceback_len += len(str(report.get("report") or report.get("last_traceback") or ""))
    failed_count = max(_int(ticket.get("failed_requeue_count"), 0), len(reports), _int(ticket.get("attempts"), 0) - 1)

    return TicketFeatures(
        business_impact=business_impact,
        urgency=urgency,
        downstream_blocked_count=downstream,
        age_hours=age_hours,
        money_risk=money_risk,
        security_risk=security_risk,
        live_risk=live_risk,
        asset_loss_score=asset_loss,
        difficulty=difficulty,
        failed_requeue_count=max(0, failed_count),
        traceback_len=max(0, traceback_len),
        auction_deadlock_cycles=deadlock_cycles,
        starvation_factor=starvation,
        value=value,
        risk_skew=risk_skew,
    )


def _skill_name(domain: str) -> str:
    return SKILL_BY_DOMAIN.get(domain, "general_delivery")


def _norm_key(s: Any) -> str:
    """Canonicalize a skill/domain key for tolerant matching: lowercase, strip a
    trailing '_skill', collapse separators. Deterministic, no I/O."""
    k = str(s or "").strip().lower()
    # normalize separators FIRST so 'risk-math-skill' and 'risk_math_skill' converge
    # BEFORE the '_skill' suffix is stripped.
    for ch in (" ", "-", "."):
        k = k.replace(ch, "_")
    while "__" in k:
        k = k.replace("__", "_")
    k = k.strip("_")
    if k.endswith("_skill"):
        k = k[: -len("_skill")]
    return k.strip("_")


def measured_skill_row(agent: dict, domain: str) -> dict:
    """Insight-A fix (2026-06-04): resolve an agent's MEASURED skill row for `domain`
    tolerantly so a legitimately-proven agent is not excluded by a mere key mismatch.

    Resolution order (first hit wins):
      1. exact mapped skill key   (SKILL_BY_DOMAIN[domain], e.g. 'risk_math_skill')
      2. exact bare domain key    (e.g. 'Risk')
      3. normalized match         (case/separator/'_skill'-suffix-insensitive) against
                                  EITHER the mapped skill key OR the bare domain.

    This does NOT loosen the gate: the mu>75 / sigma<15 / hp>80 thresholds and the
    n>=1 "measured" requirement still apply downstream. We only stop dropping a row
    that exists under a differently-cased / aliased key. Returns {} if nothing matches.
    """
    skills = agent.get("domain_skills") or {}
    if not isinstance(skills, dict):
        return {}
    skill = _skill_name(domain)
    # 1 + 2: exact keys (cheap, preserves prior behavior when keys already match)
    row = skills.get(skill)
    if isinstance(row, dict) and row:
        return row
    row = skills.get(domain)
    if isinstance(row, dict) and row:
        return row
    # 3: normalized fallback. Match against the mapped skill OR the bare domain so an
    # agent that stored credit under, e.g., 'Risk_Math_Skill' or 'risk' still resolves.
    targets = {_norm_key(skill), _norm_key(domain)}
    targets.discard("")
    for key, value in skills.items():
        if isinstance(value, dict) and value and _norm_key(key) in targets:
            return value
    return {}


def _agent_hp(agent: dict) -> float:
    hp = _num(agent.get("hp", agent.get("weight", 50.0)), 50.0)
    if 0.0 <= hp <= 1.0:
        hp *= 100.0
    return max(0.0, min(100.0, hp))


def _agent_rank(agent: dict) -> int:
    hp = _agent_hp(agent)
    if hp > 80.0:
        return max(3, _int(agent.get("rank"), 3))
    if hp >= 40.0:
        return max(2, min(3, _int(agent.get("rank"), 2)))
    return max(1, min(3, _int(agent.get("rank"), 1)))


def _default_mu_sigma(agent: dict) -> tuple[float, float]:
    rank = _agent_rank(agent)
    if rank >= 3:
        return 78.0, 14.0
    if rank == 2:
        return 62.0, 24.0
    return 48.0, 32.0


def agent_skill(agent: dict, domain: str) -> tuple[float, float, str]:
    skill = _skill_name(domain)
    # Insight-A: tolerant measured-row lookup (case/separator/alias) so a proven agent's
    # mu/sigma is honored even when stored under a differently-cased key.
    row = measured_skill_row(agent, domain)
    default_mu, default_sigma = _default_mu_sigma(agent)
    mu = max(0.0, min(100.0, _num(row.get("mu"), default_mu)))
    sigma = max(3.0, min(60.0, _num(row.get("sigma"), default_sigma)))
    return mu, sigma, skill


def _model_cost(agent: dict) -> float:
    if "model_tier_cost" in agent:
        return max(0.0, _num(agent.get("model_tier_cost"), 0.05))
    aid = str(agent.get("agent_id") or "")
    rank = _agent_rank(agent)
    if "Committee" in aid or rank >= 3:
        return 0.15
    if rank == 2:
        return 0.055
    return 0.012


def _hard_role_eligible(agent: dict, domain: str) -> bool:
    # P0-1 fix (2026-06-03): the money/security hard gate must rest on MEASURED credit,
    # not the synthetic rank-3 default (78/14) that an unproven elite-HP agent would
    # otherwise clear with zero evidence. Require an actual measured skill row.
    hp = _agent_hp(agent)
    # Insight-A (2026-06-04): resolve the measured row tolerantly so a proven agent that
    # stored credit under a differently-cased / aliased key is NOT wrongly treated as
    # unmeasured. Thresholds are unchanged: still require a REAL measured row (n>=1) AND
    # hp>80 AND mu>75 AND sigma<15. The lookup is widened; the gate is not loosened.
    row = measured_skill_row(agent, domain)
    measured = _int(row.get("n"), 0) >= 1
    mu, sigma, _ = agent_skill(agent, domain)
    return measured and hp > 80.0 and mu > 75.0 and sigma < 15.0


def _p_success(mu: float, sigma: float, difficulty: float) -> float:
    z = (mu - difficulty) / max(8.0, sigma)
    p = 1.0 / (1.0 + math.exp(-0.717 * z))
    return max(0.01, min(0.99, p))


def _context_penalty(features: TicketFeatures) -> float:
    return ALPHA * features.failed_requeue_count + BETA * features.traceback_len


def _disk_penalty(free_gb: float) -> float:
    gap = free_gb - DISK_CRITICAL_GB
    if gap <= 0:
        return float("inf")
    return THETA / gap


def _disk_yellow(free_gb: float) -> bool:
    return free_gb < DISK_YELLOW_GB


def _live_allowed() -> bool:
    return os.environ.get("QUANTCHOAI_LIVE_SWITCH", "") == "TRUE"


_LIVE_ACTION_RE = re.compile(
    r"\b(place\s*order|submit\s*order|real money|live broker|production broker|"
    r"go[- ]?live|alpaca live|ibkr live|interactive brokers live|kill switch)\b", re.I)


def _touches_real_trading_adapter(ticket: dict) -> bool:
    text = _doc_text(ticket)
    if PAPER_RE.search(text):
        return False
    # P1-4 fix (2026-06-04): a ticket that merely MENTIONS a broker/execution adapter (e.g. a
    # design or triage doc) must not be frozen as live. Freeze only on an explicit live_risk>=3,
    # or a real live ACTION verb -- not the bare adapter noun.
    for src in (ticket, ticket.get("payload") if isinstance(ticket.get("payload"), dict) else {}):
        try:
            if float((src or {}).get("live_risk") or 0) >= 3.0:
                return True
        except (TypeError, ValueError):
            pass
    return bool(_LIVE_ACTION_RE.search(text))


def _action_descriptor(ticket: dict, action: dict | None) -> dict:
    """Resolve the proposed-action descriptor for the Goodhart hard-cost.

    Precedence: an explicit `action` arg, else ticket['proposed_action'] / ticket
    top-level fields. A normal full-effort assignment has no descriptor (returns {}).
    Recognised keys:
        declined / decline      : bool  -> the agent is refusing an eligible task.
        info_content            : float in [0,1] -> 1.0 = full deliverable, ~0 = empty.
        low_info                : bool  -> explicit "safe garbage" flag.
    """
    if isinstance(action, dict):
        src = action
    elif isinstance(ticket.get("proposed_action"), dict):
        src = ticket["proposed_action"]
    else:
        src = ticket
    out: dict[str, Any] = {}
    if "declined" in src or "decline" in src:
        out["declined"] = bool(src.get("declined", src.get("decline")))
    if "low_info" in src:
        out["low_info"] = bool(src.get("low_info"))
    ic = src.get("info_content")
    if ic is not None:
        try:
            out["info_content"] = max(0.0, min(1.0, float(ic)))
        except (TypeError, ValueError):
            pass
    return out


def _goodhart_cost(features: TicketFeatures, action: dict) -> tuple[float, str | None]:
    """NAMED hard cost (peer point 2): declining an eligible task OR emitting low-info
    output must COST utility, so 'refuse the hard task' and 'ship safe garbage' are not
    dominant strategies.

    Returns (cost, reason). Deterministic; charged in the SAME units as the rest of the
    objective (utility = value-scaled). The cost is anchored to `value` (the prize the
    action forgoes / falsely claims), so refusing a high-value task costs more than
    refusing a trivial one -- a flat penalty would be gameable by only refusing the
    expensive tickets.
    """
    if not action:
        return 0.0, None
    value = max(0.0, float(features.value))
    if action.get("declined"):
        # Refusing an ELIGIBLE task forgoes its value AND consumes a queue slot: charge a
        # fraction of the prize so net utility of declining is driven below honest effort.
        return GOODHART_DECLINE_COST_FRAC * value, "GOODHART_DECLINE_COST"
    ic = action.get("info_content")
    low = action.get("low_info") or (ic is not None and ic < GOODHART_LOWINFO_THRESHOLD)
    if low:
        # Near-empty output that still claims the ticket: charge a fraction of the value
        # it falsely banks, scaled UP as info_content -> 0 (emptier = costlier).
        emptiness = 1.0 if ic is None else (1.0 - ic / GOODHART_LOWINFO_THRESHOLD)
        emptiness = max(0.0, min(1.0, emptiness))
        return GOODHART_LOWINFO_COST_FRAC * value * emptiness, "GOODHART_LOW_INFO_COST"
    return 0.0, None


def utility_for(ticket: dict, agent: dict, disk_free: float | None = None, db=None,
                action: dict | None = None, features: TicketFeatures | None = None,
                vendor_counts: dict | None = None) -> UtilityResult:
    """Score one (ticket, agent) pair.

    PERF (2026-06-04): `features` is agent-INDEPENDENT -- infer_ticket_features (and the
    CVaR tail term inside it) depends only on the ticket, while only p_success / model_cost
    / mu / sigma / goodhart_cost differ per candidate agent. When a caller scores the SAME
    ticket against many agents (auction_pick) it computes the features ONCE and passes them
    here via `features=`, so the per-ticket inference + ledger downstream-count + ES tail
    pricing run once per ticket instead of once per (ticket,agent). When `features` is None
    (every existing call site, single-shot scoring) we infer exactly as before -- the
    branch is the only difference, so the scored utility is byte-identical either way.

    MARGINAL-CVaR (roadmap item 9): `vendor_counts` is the live per-vendor in-flight book
    (vendor -> count). It is agent-AND-ticket-independent (a fleet snapshot), so auction_pick
    reads it ONCE per cycle and threads it here. When omitted we read it from the queue (or,
    with no db, treat the book as empty -> the marginal multiplier is 1.0 and independent
    pricing recovers). Behind the QUANTCHO_MARGINAL_CVAR_ENABLED flag: when ON, the tail term
    is scaled by this assignment's MARGINAL (Euler) contribution to fleet CVaR given the
    vendor's current correlated in-flight share, AND the vendor concentration cap is enforced
    as a live eligibility constraint; when OFF, neither applies (independent fallback).
    """
    free_bytes = disk_free_bytes() if disk_free is None else int(disk_free * 1024 * 1024 * 1024)
    free = free_bytes / (1024 ** 3)
    if features is None:
        features = infer_ticket_features(ticket, db=db)
    if free_bytes < DISK_CRITICAL_BYTES:
        return UtilityResult(
            utility=-float("inf"),
            p_success=0.0,
            mu=0.0,
            sigma=0.0,
            model_cost=0.0,
            risk_penalty=0.0,
            context_penalty=_context_penalty(features),
            disk_penalty=float("inf"),
            reason="DISK_SENTINEL_FREEZE",
            features=features,
        )
    if _touches_real_trading_adapter(ticket) and not _live_allowed():
        return UtilityResult(
            utility=-float("inf"),
            p_success=0.0,
            mu=0.0,
            sigma=0.0,
            model_cost=0.0,
            risk_penalty=float("inf"),
            context_penalty=_context_penalty(features),
            disk_penalty=_disk_penalty(free),
            reason="LIVE_SWITCH_GATE",
            features=features,
        )

    mu, sigma, _ = agent_skill(agent, str(ticket.get("target_domain") or ""))
    p_success = _p_success(mu, sigma, features.difficulty)
    model_cost = _model_cost(agent) * (1.0 + GAMMA * (sigma ** 2))

    # MARGINAL-CVaR + VENDOR-CONCENTRATION (roadmap item 9) -----------------------
    # The tail term `risk_skew` is the per-ticket CVaR priced INDEPENDENTLY. Under the flag,
    # scale it by this assignment's MARGINAL (Euler) contribution to FLEET CVaR given the
    # vendor's current correlated in-flight share, and enforce the concentration cap as a
    # live eligibility constraint. Flag off (or empty vendor book) -> multiplier 1.0 ==
    # the independent fallback, byte-identical to the pre-wire path.
    vendor = _agent_vendor(agent)
    marginal_mult = 1.0
    vendor_share = 0.0
    concentration_blocked = False
    if MARGINAL_CVAR_ENABLED:
        book = _inflight_vendor_counts(db, vendor_counts)
        inflight_k = max(0, int(book.get(vendor, 0)))
        marginal_mult = _marginal_cvar_multiplier(inflight_k, VENDOR_CORR_RHO)
        vendor_share = _prospective_vendor_share(vendor, book)
        # Concentration cap is a CRITICAL-work correlation control. A bare-keyword doc ticket
        # is not critical; only genuine money/security-critical tickets are gated, matching
        # the rest of the auction's narrow gate. The marginal-CVaR PRICE (multiplier) still
        # applies to every ticket so non-critical pile-up is discouraged economically too.
        if _is_genuine_money_critical(ticket) and _concentration_blocks(vendor, book):
            concentration_blocked = True

    risk_penalty = (1.0 - p_success) * features.risk_skew * features.asset_loss_score * marginal_mult
    context_penalty = _context_penalty(features)
    disk_penalty = _disk_penalty(free)
    # GOODHART HARD-COST (peer point 2): a named term on the LIVE objective, charged when
    # the scored action is a decline / low-information output. 0.0 for honest effort.
    goodhart_cost, goodhart_reason = _goodhart_cost(features, _action_descriptor(ticket, action))
    utility = ((p_success * features.value)
               - model_cost - risk_penalty - context_penalty - disk_penalty - goodhart_cost)

    reason = "OK"
    if goodhart_reason:
        reason = goodhart_reason
    critical_risk = max(features.money_risk, features.security_risk, features.live_risk)
    min_p = CRITICAL_CONFIDENCE_MIN_P if critical_risk >= 4.0 else LOW_CONFIDENCE_MIN_P
    if critical_risk >= 3.0 and p_success < min_p:
        reason = "SKIP_LOW_CONFIDENCE_FOR_RISK"
        utility = -float("inf")
    # Live concentration cap: a >cap assignment on one vendor is INELIGIBLE (hard constraint),
    # checked AFTER the confidence gate so the named reason surfaces deterministically.
    if concentration_blocked:
        reason = "VENDOR_CONCENTRATION_CAP"
        utility = -float("inf")

    return UtilityResult(
        utility=utility,
        p_success=p_success,
        mu=mu,
        sigma=sigma,
        model_cost=model_cost,
        risk_penalty=risk_penalty,
        context_penalty=context_penalty,
        disk_penalty=disk_penalty,
        reason=reason,
        features=features,
        goodhart_cost=goodhart_cost,
        vendor=vendor,
        marginal_cvar_mult=marginal_mult,
        vendor_share=vendor_share,
    )


def _agent_domains(agent: dict) -> list[str]:
    aid = str(agent.get("agent_id") or "")
    squad = str(agent.get("squad") or "")
    if aid.startswith(("CTO-", "Integration-R3", "FinOps-", "HR-")):
        return ALL_DOMAINS
    if squad in {"Committee", "Governance"}:
        return [squad, "Integration", "Risk", "Pricing", "Compliance", "Accounting", "Tax"]
    return [squad] if squad else []


def _domain_locked(agent: dict, domain: str) -> bool:
    locks = agent.get("domain_locks") or {}
    until = locks.get(domain)
    dt = _parse_ts(until)
    return bool(dt and dt > _dt.datetime.utcnow())


def _ticket_is_money_math(ticket: dict) -> bool:
    # V6 narrowed: only genuine money-math mutations, not bare keyword mentions.
    return _is_genuine_money_critical(ticket)


def _eligible_agent(agent: dict, ticket: dict) -> bool:
    hp = _agent_hp(agent)
    if hp < 20.0:
        return False
    domain = str(ticket.get("target_domain") or "")
    if domain not in _agent_domains(agent):
        return False
    if _domain_locked(agent, domain):
        return False
    rank = _agent_rank(agent)
    if rank < _int(ticket.get("min_rank"), 1):
        return False
    if domain.lower() in ELITE_ONLY_DOMAINS and not _hard_role_eligible(agent, domain):
        return False
    if domain == "Committee" and not _hard_role_eligible(agent, domain):
        return False
    if _ticket_is_money_math(ticket) and not _hard_role_eligible(agent, domain):
        return False
    return True


def _recent_active_agents_uncached(db, current_agent: dict) -> list[dict]:
    cutoff = (_dt.datetime.utcnow() - _dt.timedelta(minutes=AUCTION_ACTIVE_MIN)).isoformat() + "Z"
    candidates = list(db[L].find({
        "active": {"$ne": False},
        "$or": [{"last_seen_at": {"$gte": cutoff}}, {"agent_id": current_agent["agent_id"]}],
    }))
    agents: list[dict] = []
    for agent in candidates:
        pid = agent.get("process_id")
        if agent.get("agent_id") == current_agent.get("agent_id"):
            agents.append(agent)
            continue
        if psutil is not None and pid is not None:
            try:
                if not psutil.pid_exists(int(pid)):
                    continue
            except Exception:
                pass
        agents.append(agent)
    if not any(a.get("agent_id") == current_agent.get("agent_id") for a in agents):
        agents.append(current_agent)
    return agents


def _recent_active_agents(db, current_agent: dict) -> list[dict]:
    """TTL-cached recently-active roster (efficiency 2026-06-04).

    Returns the SAME list the underlying ledger scan would, but reuses a recent scan for
    up to AUCTION_CYCLE_SEC/2 so a burst of polls within one cycle does not re-scan the
    ledger (and re-probe every pid) once per call. The result is byte-identical to the
    uncached scan within the TTL window because the roster only changes on
    init_agent/guillotine, both of which call invalidate_roster_cache(). The "always
    include self" contract is preserved: `current_agent` is appended to the uncached
    result when absent, and the cache is keyed per caller so self is never another
    caller's stale entry.
    """
    aid = current_agent.get("agent_id")
    if aid is None:
        # No stable cache key -> fall back to a fresh scan (never cache an anonymous caller).
        return _recent_active_agents_uncached(db, current_agent)
    ttl = _roster_ttl_sec()
    now = _monotonic()
    with _ROSTER_CACHE_LOCK:
        entry = _ROSTER_CACHE.get(aid)
        if entry is not None and ttl > 0.0 and (now - entry[1]) < ttl:
            return entry[0]
    # Compute outside the lock (the ledger scan can block); store under the lock.
    agents = _recent_active_agents_uncached(db, current_agent)
    with _ROSTER_CACHE_LOCK:
        _ROSTER_CACHE[aid] = (agents, now)
    return agents


def _deps_done(db, ticket: dict) -> bool:
    deps = ticket.get("blocked_by") or []
    if not deps:
        return True
    remaining = db[Q].count_documents({"ticket_id": {"$in": deps}, "status": {"$ne": "DONE"}})
    return remaining == 0


def _touch_auction_cycle(db, ticket: dict) -> dict:
    now = _dt.datetime.utcnow()
    last = _parse_ts(ticket.get("last_auction_timestamp"))
    if last and (now - last).total_seconds() < AUCTION_CYCLE_SEC:
        return ticket
    update = {
        "$set": {"last_auction_timestamp": _now()},
        "$inc": {"auction_deadlock_cycles": 1},
    }
    updated = db[Q].find_one_and_update(
        {"_id": ticket["_id"], "status": {"$in": ["QUEUED", "READY"]}},
        update,
        return_document=ReturnDocument.AFTER,
    )
    return updated or ticket


def _agent_has_empty_worktree_slot(db, agent: dict) -> bool:
    aid = agent.get("agent_id")
    if not aid:
        return False
    return db[Q].count_documents({"picked_by": aid, "status": "IN_PROGRESS"}) == 0


def _starvation_bypass_agent(db, ticket: dict, agents: list[dict]) -> dict | None:
    if _int(ticket.get("auction_deadlock_cycles"), 0) <= STARVATION_BYPASS_CYCLES:
        return None
    if _touches_real_trading_adapter(ticket) and not _live_allowed():
        return None
    eligible: list[tuple[float, float, dict]] = []
    domain = str(ticket.get("target_domain") or "")
    for agent in agents:
        if _agent_rank(agent) < 3:
            continue
        if not _eligible_agent(agent, ticket):
            continue
        if not _agent_has_empty_worktree_slot(db, agent):
            continue
        mu, sigma, _ = agent_skill(agent, domain)
        eligible.append((mu, -sigma, agent))
    if not eligible:
        return None
    eligible.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return eligible[0][2]


def _bypass_result(result: UtilityResult) -> UtilityResult:
    stripped_utility = (
        result.p_success * result.features.value
        - result.risk_penalty
        - result.context_penalty
        - result.disk_penalty
    )
    return replace(result, utility=stripped_utility, model_cost=0.0, reason="STARVATION_BYPASS_R3")


def _log(db, agent_id: str, event: str, ticket_id: str | None = None, status: str | None = None, msg: str = "", level: str = "INFO"):
    try:
        db[LOG].insert_one({
            "ts": _now(),
            "agent_id": agent_id,
            "event": event,
            "ticket_id": ticket_id,
            "status": status,
            "level": level,
            "msg": msg[:500],
        })
    except Exception:
        pass


def _log_throttled(
    db,
    agent_id: str,
    event: str,
    ticket_id: str | None = None,
    status: str | None = None,
    msg: str = "",
    level: str = "INFO",
    throttle_sec: float = WARN_LOG_THROTTLE_SEC,
):
    key = (agent_id, event, status)
    current = _dt.datetime.utcnow().timestamp()
    last = _LAST_LOCAL_LOG.get(key, 0.0)
    if current - last < throttle_sec:
        return
    _LAST_LOCAL_LOG[key] = current
    _log(db, agent_id, event, ticket_id=ticket_id, status=status, msg=msg, level=level)


def auction_pick(db, agent_id: str, domains: list[str], max_rank: int = 1, locked_domains: list[str] | None = None) -> dict | None:
    """Claim the best ticket only if this worker wins the global utility auction.

    The current process is still pull-based for compatibility, but each poll
    compares the caller against recently active eligible workers before claiming.
    """
    free_bytes = disk_free_bytes()
    free = free_bytes / (1024 ** 3)
    if free_bytes < DISK_CRITICAL_BYTES:
        _log(db, agent_id, "CRITICAL_DISK_LOCKOUT", status="BLOCKED_BY_DISK",
             msg=f"C: free {free:.2f}GB < hard ceiling {DISK_CRITICAL_GB:.2f}GB", level="ERROR")
        trigger_reaper_hard_cleanup(db, reason="CRITICAL_DISK_LOCKOUT")
        return None
    if _disk_yellow(free):
        _log_throttled(db, agent_id, "DISK_YELLOW_DISPATCH_CONTINUES", status="WARN_DISK",
                       msg=f"C: free {free:.2f}GB < yellow {DISK_YELLOW_GB:.2f}GB; dispatch continues with disk penalty, no scale-up", level="WARN")
    is_hot, cpu_pct = cpu_hot()
    if is_hot:
        _log_throttled(db, agent_id, "DISPATCH_PAUSED_CPU_HOT", status="PAUSED",
                       msg=f"CPU {cpu_pct:.1f}% > {CPU_HOT_PERCENT:.1f}%; no new allocations", level="WARN")
        return None

    current = db[L].find_one({"agent_id": agent_id}) or {
        "agent_id": agent_id,
        "rank": max_rank,
        "squad": domains[0] if domains else "",
        "hp": 50.0,
        "weight": 50.0,
        "active": True,
        "last_seen_at": _now(),
    }
    current["rank"] = max_rank
    current["squad"] = current.get("squad") or (domains[0] if domains else "")

    eligible_domains = [d for d in domains if d not in set(locked_domains or [])]
    if not eligible_domains:
        return None

    q = {
        "status": {"$in": ["QUEUED", "READY"]},
        "target_domain": {"$in": eligible_domains},
        "$or": [{"min_rank": {"$exists": False}}, {"min_rank": {"$lte": max_rank}}],
    }
    candidates = list(db[Q].find(q).sort([("priority", ASCENDING), ("enqueued_at", ASCENDING)]).limit(AUCTION_CANDIDATE_LIMIT))
    if not candidates:
        return None

    active_agents = _recent_active_agents(db, current)
    best_for_current: tuple[float, dict, UtilityResult] | None = None
    governor_blocks: list[tuple[dict, UtilityResult]] = []

    # PER-TICKET FEATURE CACHE (efficiency 2026-06-04): infer_ticket_features (incl. the
    # CVaR tail term) is agent-INDEPENDENT, yet without this it ran once per (ticket,agent)
    # inside utility_for -- O(agents) redundant inferences + ledger downstream scans per
    # ticket. We compute it ONCE per ticket per auction cycle and pass it into every
    # candidate's utility_for. The cache is scoped to THIS auction_pick call (one cycle) and
    # keyed by the post-touch ticket _id, so a deadlock-cycle bump from _touch_auction_cycle
    # is reflected in the single inference all agents then share. Result rankings are
    # identical to the no-cache path (the test asserts this); only the agent-specific
    # p_success/model_cost/goodhart legs vary per agent.
    feature_cache: dict[Any, TicketFeatures] = {}

    # MARGINAL-CVaR vendor book (roadmap item 9): the live per-vendor in-flight exposure is a
    # FLEET snapshot -- ticket- and agent-independent -- so we read it ONCE per auction cycle
    # and thread it into every utility_for call. The marginal-CVaR multiplier and the
    # concentration cap both price against THIS book. Empty/flag-off -> independent fallback.
    vendor_book = _inflight_vendor_counts(db) if MARGINAL_CVAR_ENABLED else {}

    for ticket in candidates:
        ticket = _touch_auction_cycle(db, ticket)
        if not _deps_done(db, ticket):
            continue
        cache_key = ticket.get("_id", ticket.get("ticket_id"))
        ticket_features = feature_cache.get(cache_key)
        if ticket_features is None:
            ticket_features = infer_ticket_features(ticket, db=db)
            feature_cache[cache_key] = ticket_features
        scored: list[tuple[float, dict, UtilityResult]] = []
        bypass_agent: dict | None = None
        bypass_result: UtilityResult | None = None
        for agent in active_agents:
            if not _eligible_agent(agent, ticket):
                continue
            if not _agent_has_empty_worktree_slot(db, agent):
                continue
            result = utility_for(ticket, agent, disk_free=free, db=db, features=ticket_features,
                                 vendor_counts=vendor_book)
            if result.reason == "LIVE_SWITCH_GATE":
                governor_blocks.append((ticket, result))
                continue
            if math.isfinite(result.utility):
                scored.append((result.utility, agent, result))
            if agent.get("agent_id") == agent_id:
                bypass_result = result
        bypass_agent = _starvation_bypass_agent(db, ticket, active_agents)
        if not scored:
            if bypass_agent and bypass_agent.get("agent_id") == agent_id and bypass_result:
                bypass_result = _bypass_result(bypass_result)
                if best_for_current is None or bypass_result.utility > best_for_current[0]:
                    best_for_current = (bypass_result.utility, ticket, bypass_result)
            continue
        scored.sort(key=lambda row: row[0], reverse=True)
        utility, winning_agent, result = scored[0]
        if utility <= MIN_POSITIVE_UTILITY:
            if bypass_agent and bypass_agent.get("agent_id") == agent_id and bypass_result:
                bypass_result = _bypass_result(bypass_result)
                if best_for_current is None or bypass_result.utility > best_for_current[0]:
                    best_for_current = (bypass_result.utility, ticket, bypass_result)
            continue
        if winning_agent.get("agent_id") != agent_id:
            continue
        if best_for_current is None or utility > best_for_current[0]:
            best_for_current = (utility, ticket, result)

    if best_for_current is None:
        # Only physical/live blockers change ticket state here. Agent-specific
        # low confidence must not freeze a ticket that an elite worker can solve.
        for ticket, result in governor_blocks[:3]:
            db[Q].update_one(
                {"_id": ticket["_id"], "status": {"$in": ["QUEUED", "READY"]}},
                {"$set": {
                    "status": "BLOCKED_BY_LIVE_SWITCH_GATE",
                    "risk_governor_reason": result.reason,
                    "risk_governor_at": _now(),
                    "auction_features": result.features.__dict__,
                }},
            )
            _log(db, agent_id, "risk_governor_block", ticket.get("ticket_id"),
                 "BLOCKED_BY_LIVE_SWITCH_GATE", msg=result.reason, level="WARN")
        return None

    utility, ticket, result = best_for_current
    claimed = db[Q].find_one_and_update(
        {"_id": ticket["_id"], "status": {"$in": ["QUEUED", "READY"]}},
        {"$set": {
            "status": "IN_PROGRESS",
            "picked_by": agent_id,
            "picked_at": _now(),
            "auction_utility": float(utility),
            "auction_p_success": float(result.p_success),
            "auction_mu": float(result.mu),
            "auction_sigma": float(result.sigma),
            "auction_reason": result.reason,
            "auction_features": result.features.__dict__,
            # MARGINAL-CVaR (roadmap item 9): stamp the vendor + the marginal/concentration
            # telemetry so the NEXT auction cycle's in-flight book counts this position
            # against its vendor (the live correlation exposure), and the shadow-compare can
            # audit the marginal multiplier / prospective share that priced this claim.
            "picked_vendor": result.vendor,
            "auction_marginal_cvar_mult": float(result.marginal_cvar_mult),
            "auction_vendor_share": float(result.vendor_share),
        }, "$inc": {"attempts": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if claimed:
        _log(db, agent_id, "auction_pick_up", claimed.get("ticket_id"), "IN_PROGRESS",
             msg=f"utility={utility:.2f} p={result.p_success:.3f} mu={result.mu:.1f} sigma={result.sigma:.1f}")
    return claimed
