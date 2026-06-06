"""QuantChoAI unified governance parameter registry (V1-V5).

Single source of truth for every survival / credit / hazard / propulsion /
optimization constant in the bridge control plane.

STATUS: REFERENCE REGISTRY ONLY (created during the 2026-06-02 Claude handover).
    This module is NOT yet wired into the runtime. The live values still live in:
        - ledger.py      lines ~48-70   (V1 survival, V3 hazard, V4 propulsion)
        - dispatcher.py  lines ~29-47   (V2 credit spread, V5 optimization, disk/cpu gates)
        - dispatcher._hard_role_eligible (the money/security hard gate: hp>80, mu>75, sigma<15)
    The defaults / env-var keys below are mirrored 1:1 from those files as of
    2026-06-02 so this can later become a drop-in source of truth WITHOUT changing
    behaviour. Do NOT rewire ledger.py / dispatcher.py to import this while Codex
    is actively working the branch -- that is a follow-up for the next session.

DESIGN RULES (inherited from V1-V5 mandates):
    - All governance math is local Python / Mongo only. No LLM in the control plane.
    - V1 survival rules are hard/active. V2 informs routing.
    - V3 hazard + V4 propulsion are SHADOW telemetry until calibrated (500-1000 events).
    - V5 risk-budgeted dispatch (dispatcher.auction_pick) is written + tested but the
      live worker loop (base_agent.loop) still uses ticket_queue.pick_up (priority FIFO).
      V5 is "engine built, not plugged in" -- promote only after shadow replay evidence.
    - Disk hard gate overrides fleet ambition: C: < 10GB yellow, < 5GB red.
"""
from __future__ import annotations

import os


def _f(env_key: str, default: float) -> float:
    """Env-overridable float, matching the resolution used in ledger/dispatcher."""
    try:
        return float(os.environ.get(env_key, str(default)))
    except (TypeError, ValueError):
        return default


def _i(env_key: str, default: int) -> int:
    try:
        return int(os.environ.get(env_key, str(default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# V1 -- HP / Guillotine / Soft Recycle (SURVIVAL GATE) -- HARD / ACTIVE
# Mirrors ledger.py:48-53. HP decides only whether a process stays alive;
# it does NOT decide ticket routing (that is V2+V3+V5).
# ---------------------------------------------------------------------------
V1_SURVIVAL = {
    "DEFAULT_HP": 50.0,               # spawn HP
    "MAX_HP": 100.0,                  # ceiling
    "GUILLOTINE_HP": 20.0,            # below this -> kill/replace
    "SOFT_RECYCLE_HP": 40.0,          # 20<=hp<40 -> soft recycle candidate (if unleased)
    "SOFT_RECYCLE_COOLDOWN_HOURS": 6, # min gap between soft recycles
    "IDLE_DECAY_HP": -2.0,            # HP drift per idle cycle
    "ROTATION_LOCK_MINUTES": 30,      # domain rotation lock after a switch
    # Survival bands (derived, for readability):
    #   elite      : hp > 80
    #   standard   : 40 <= hp <= 80
    #   struggling : 20 <= hp < 40
    #   guillotine : hp < 20
}


# ---------------------------------------------------------------------------
# V2 -- Bayesian mu/sigma credit + sigma credit spread (ALLOCATION) -- INFORMS ROUTING
# Mirrors dispatcher.py GAMMA + _hard_role_eligible + default mu/sigma tiers.
# adjusted_model_cost = base_model_cost * (1 + GAMMA * sigma^2)
# ---------------------------------------------------------------------------
V2_CREDIT = {
    "GAMMA_SIGMA_SPREAD": _f("QUANTCHO_AUCTION_GAMMA", 0.025),  # sigma^2 credit spread (eta in whitepaper)
    # Default mu/sigma by rank when an agent has no measured domain skill row
    # (dispatcher._default_mu_sigma):
    "DEFAULT_MU_SIGMA_BY_RANK": {
        3: (78.0, 14.0),
        2: (62.0, 24.0),
        1: (48.0, 32.0),
    },
    "SIGMA_CLAMP": (3.0, 60.0),   # agent_skill() clamps sigma into this range
    "MU_CLAMP": (0.0, 100.0),
    # Model tier cost proxies (dispatcher._model_cost), pre sigma-spread:
    "MODEL_COST_ELITE_R3": 0.15,
    "MODEL_COST_R2": 0.055,
    "MODEL_COST_R1": 0.012,
    # --- HARD GATE for money-math / security / committee / live lanes ---
    # (dispatcher._hard_role_eligible) -- currently hardcoded; surfaced here so it
    # is no longer invisible. ALL THREE must hold for an agent to touch these lanes.
    "MONEY_SECURITY_GATE_MIN_HP": 80.0,    # hp > 80
    "MONEY_SECURITY_GATE_MIN_MU": 75.0,    # mu > 75
    "MONEY_SECURITY_GATE_MAX_SIGMA": 15.0, # sigma < 15
    # P2-5 drift fix (2026-06-04): spawn credit prior for an agent with no measured row.
    # Mirrors ledger.default_credit_rating() 1:1 (do NOT diverge -- the drift test asserts it).
    "DEFAULT_CREDIT_RATING": {"mu": 50.0, "sigma": 30.0},
    # Measured-row requirement for the hard gate (dispatcher._hard_role_eligible): a real
    # measured skill row (n >= MEASURED_ROW_MIN_N) is required before mu/sigma are honored.
    "MEASURED_ROW_MIN_N": 1,
}


# ---------------------------------------------------------------------------
# V3 -- Hazard rate / white-noise filter (CONTEXT-RISK OVERLAY) -- SHADOW
# Mirrors ledger.py:56-62. The a/b/c/d weights you remembered are these:
#   a = WEIGHT_CONTEXT, b = WEIGHT_STREAK, c = WEIGHT_MINUTES, d = WEIGHT_REQUEUE
# hazard = clamp(BASE + a*ctx + b*streak + c*minutes + d*requeue, 0, 1)
# ---------------------------------------------------------------------------
V3_HAZARD = {
    "BASE_HAZARD": _f("QUANTCHO_HAZARD_BASE", 0.05),
    "WEIGHT_CONTEXT": _f("QUANTCHO_HAZARD_WEIGHT_CONTEXT", 0.00001),  # a
    "WEIGHT_STREAK": _f("QUANTCHO_HAZARD_WEIGHT_STREAK", 0.1),        # b
    "WEIGHT_MINUTES": _f("QUANTCHO_HAZARD_WEIGHT_MINUTES", 0.005),    # c
    "WEIGHT_REQUEUE": _f("QUANTCHO_HAZARD_WEIGHT_REQUEUE", 0.15),     # d
    "SOFT_RECYCLE_THRESHOLD": _f("QUANTCHO_HAZARD_SOFT_RECYCLE", 0.6),
    "GUILLOTINE_THRESHOLD": _f("QUANTCHO_HAZARD_GUILLOTINE", 0.9),
    # White-noise filter: these error classes are INFRA noise -> requeue, no HP/mu
    # penalty. Logic failures (pytest/AssertionError/invariant breach/etc) DO penalize.
    "INFRA_NOISE_MARKERS": (
        "TimeoutExpired", "No space left on device", "git.exc.LockFailed",
        "index.lock", "ServerSelectionTimeout", "MongoDB Connection Timeout",
        "Vite out of memory", "JavaScript heap out of memory",
        "ECONNREFUSED", "ETIMEDOUT",
    ),
    "EVENT_TTL_SECONDS": 259200,  # agent_events retention = 72h
    "CALIBRATION_MIN_EVENTS": 500,    # min ticket events before any hard hazard gate
    "CALIBRATION_PREFERRED_EVENTS": 1000,
}


# ---------------------------------------------------------------------------
# V4 -- Warm-state / propulsion efficiency (COMPUTE-EFFICIENCY OVERLAY) -- SHADOW
# Mirrors ledger.py:63-70. effective_thrust = mu*specific_impulse*warm_eff - drag.
# Do NOT hard-kill on V4 alone until calibrated.
# ---------------------------------------------------------------------------
V4_PROPULSION = {
    "WARM_EFFICIENCY_ALPHA": _f("QUANTCHO_WARM_EFFICIENCY_ALPHA", 0.08),
    "WARM_EFFICIENCY_FLOOR": _f("QUANTCHO_WARM_EFFICIENCY_FLOOR", 0.20),
    "CONTEXT_DRAG_PER_10K": _f("QUANTCHO_PROP_CONTEXT_DRAG_PER_10K", 0.05),
    "TRACEBACK_DRAG_PER_10K": _f("QUANTCHO_PROP_TRACEBACK_DRAG_PER_10K", 0.03),
    "REPO_CHURN_DRAG": _f("QUANTCHO_PROP_REPO_CHURN_DRAG", 0.04),
    "STALE_LOG_DRAG": _f("QUANTCHO_PROP_STALE_LOG_DRAG", 0.02),
    "NO_PROGRESS_DRAG": _f("QUANTCHO_PROP_NO_PROGRESS_DRAG", 0.12),
    "NO_PROGRESS_LIMIT": _i("QUANTCHO_PROP_NO_PROGRESS_LIMIT", 3),
    # Shadow-guillotine guardrail fires only when ALL THREE hold:
    #   effective_thrust < escape_velocity
    #   hazard_rate >= V3 GUILLOTINE_THRESHOLD
    #   repeated_no_progress_cycles >= NO_PROGRESS_LIMIT
}


# ---------------------------------------------------------------------------
# V5 -- Risk-budgeted dispatch / expected-utility auction (MASTER OPTIMIZATION)
# Mirrors dispatcher.py:29-47. The auction (dispatcher.auction_pick) is BUILT +
# TESTED but NOT wired into base_agent.loop (still ticket_queue.pick_up FIFO).
#   utility = p_success*value - model_cost - risk_penalty - context_penalty - disk_penalty
#   value   = business_impact * urgency * (1+log1p(downstream)) * exp(LAMBDA*age)
#   risk_skew = exp(money_risk + security_risk + live_risk)   (systemic VaR skew)
# ---------------------------------------------------------------------------
V5_OPTIMIZATION = {
    # Objective weights
    "LAMBDA_STARVATION": _f("QUANTCHO_AUCTION_LAMBDA", 0.018),   # exp(lambda*queue_age) starvation lift
    "ALPHA_REQUEUE_PENALTY": _f("QUANTCHO_AUCTION_ALPHA", 3.0),  # context penalty per failed requeue
    "BETA_TRACEBACK_PENALTY": _f("QUANTCHO_AUCTION_BETA", 0.0015),  # context penalty per traceback char
    "THETA_DISK_PENALTY": _f("QUANTCHO_AUCTION_THETA", 18.0),    # disk_penalty = THETA / (free_gb - critical)
    # Acceptance gates
    "MIN_POSITIVE_UTILITY": _f("QUANTCHO_AUCTION_MIN_UTILITY", 0.0),
    "LOW_CONFIDENCE_MIN_P": _f("QUANTCHO_AUCTION_MIN_RISK_P", 0.72),       # risk>=3 needs p_success>=this
    "CRITICAL_CONFIDENCE_MIN_P": _f("QUANTCHO_AUCTION_CRITICAL_RISK_P", 0.86),  # risk>=4 needs p_success>=this
    # Auction loop controls
    "ACTIVE_WINDOW_MIN": _i("QUANTCHO_AUCTION_ACTIVE_MIN", 90),
    "CANDIDATE_LIMIT": _i("QUANTCHO_AUCTION_CANDIDATE_LIMIT", 200),
    "CYCLE_SEC": _i("QUANTCHO_AUCTION_CYCLE_SEC", 60),
    "STARVATION_BYPASS_CYCLES": _i("QUANTCHO_STARVATION_BYPASS_CYCLES", 10),
    # --- NOT YET CODE (V5 addendum theory; absent from dispatcher.py) ---
    # Surfaced here so the gap is explicit, not forgotten:
    "KAPPA_CONCURRENCY_SLIPPAGE": None,  # concurrency_slippage = base*exp(kappa*rho); rho=workers/safe_cap
    "FLEET_RISK_BUDGET": None,           # sum(active_ticket_risk * agent_sigma) <= budget (systemic VaR cap)
    "SAFE_WORKER_CAPACITY": None,        # denominator for rho; machine-dependent, ~1-2 under low disk
}


# ---------------------------------------------------------------------------
# V6 -- Systematic-vs-idiosyncratic REGIME DETECTION (correlation regime) -- SHADOW
# One-factor decomposition of the domain trouble/return matrix. systematic_share =
# top-eigenvalue share of total variance. When the fleet's trouble stops being
# spread across domains (idiosyncratic) and starts moving TOGETHER (systematic), a
# common-factor shock is in play -> raise the eligibility bar / freeze critical work.
#
# Empirical anchor (SHADOW_SCORER_FINDINGS_20260603 / v6_backtest):
#   systematic_share = 0.027 on the real historical trouble set => IDIOSYNCRATIC.
#   The detector must NOT alarm at that level; it watches for a future spike.
#
# ALARM_THRESHOLD is the share at which the regime flips to "systematic". 0.50 means
# the single dominant factor explains a majority of variance (one factor > all the
# idiosyncratic noise combined). WARN_THRESHOLD is an earlier amber light.
# MIN_FACTORS / MIN_OBS_PER_FACTOR guard against alarming on a degenerate matrix
# (one domain, or one observation each) where "share" is meaningless / inflated.
# ---------------------------------------------------------------------------
V6_REGIME = {
    "SYSTEMATIC_ALARM_THRESHOLD": _f("QUANTCHO_REGIME_ALARM", 0.50),   # share >= this -> SYSTEMATIC alarm
    "SYSTEMATIC_WARN_THRESHOLD": _f("QUANTCHO_REGIME_WARN", 0.30),     # share >= this -> amber watch
    "MIN_FACTORS": _i("QUANTCHO_REGIME_MIN_FACTORS", 3),              # need >=3 domains to judge a regime
    "MIN_OBS_PER_FACTOR": _i("QUANTCHO_REGIME_MIN_OBS", 2),           # need >=2 obs/domain for within-var
    "MIN_TOTAL_OBS": _i("QUANTCHO_REGIME_MIN_TOTAL_OBS", 8),         # min sample before any verdict
    # Bootstrap confidence on systematic_share (deterministic, seeded resampling).
    "BOOTSTRAP_RESAMPLES": _i("QUANTCHO_REGIME_BOOTSTRAP", 400),
    "BOOTSTRAP_SEED": _i("QUANTCHO_REGIME_SEED", 20260604),
    "CONFIDENCE_LEVEL": _f("QUANTCHO_REGIME_CONF", 0.90),            # two-sided CI level
    # Control stability: the bar lift driven by systematic_share must be a contraction
    # (Lipschitz constant of the lift map < 1) so the feedback loop cannot oscillate.
    "BAR_LIFT_KAPPA": _f("QUANTCHO_REGIME_BAR_KAPPA", 0.30),         # lift = kappa * share
    "REGIME_BACKTEST_SHARE": 0.027,  # documented regression target from real trouble set
}


# ---------------------------------------------------------------------------
# V6 -- NARROW money-critical classification regexes (ROUTING GATE) -- WIRED
# P2-5 drift fix (2026-06-04): these three regex SOURCE strings were live in
# dispatcher.py (_MUT_RE / _MONEY_NOUN_RE / _PROTECTED_RE) but ABSENT from this
# registry, so the "single source of truth" had drifted from the code. Mirrored here
# 1:1 (re.IGNORECASE) so the registry-vs-live drift test (test_audit_fixes.py) can
# assert dispatcher's compiled .pattern == these strings. A ticket is money/security
# critical ONLY for an explicit risk field >=3, a mutation verb co-located with a money
# noun, or a protected path/control -- NOT a bare keyword mention in a doc.
# Validated 2026-06-03: cuts auction over-blocking ~47% -> ~6%.
# ---------------------------------------------------------------------------
V6_MONEY_CRITICAL = {
    "MUT_RE": (
        r"\b(insert|write|writes|writing|written|merge|merged|reseed|submit|"
        r"place\s*order|apply|mutate|mutated|settle|settled|deploy|migrat)\w*"
    ),
    "MONEY_NOUN_RE": (
        r"\b(var|cvar|pnl|p&l|nav|gross mv|market value|position|positions|margin|"
        r"exposure|ledger|trade|fill|valuation|pricing|cost basis|1099|wash|tax)\b"
    ),
    "PROTECTED_RE": (
        r"\b(risk\.py|positions|trade repository|traderepository|t3|t9|broker adapter|"
        r"execution adapter|kill switch|place order|submit order)\b"
    ),
    "EXPLICIT_RISK_CRITICAL_THRESHOLD": 3.0,  # money_risk/security_risk/live_risk >= this
}


# ---------------------------------------------------------------------------
# Physical / host gates -- HARD / ACTIVE. Disk overrides everything.
# Mirrors dispatcher.py:29-37. < red -> DISK_SENTINEL_FREEZE (utility = -inf).
# ---------------------------------------------------------------------------
PHYSICAL_GATES = {
    "DISK_YELLOW_GB": _f("QUANTCHO_DISK_YELLOW_GB", 10.0),   # do not scale fleet below this
    "DISK_CRITICAL_GB": _f("QUANTCHO_DISK_CRITICAL_GB", 5.0),  # hard pause + cleanup below this
    "CPU_HOT_PERCENT": _f("QUANTCHO_CPU_HOT_PERCENT", 85.0),   # no new allocations above this
    "WARN_LOG_THROTTLE_SEC": _f("QUANTCHO_WARN_LOG_THROTTLE_SEC", 300.0),
    "ORPHAN_MIN": _i("QUANTCHO_ORPHAN_MIN", 20),  # reaper: IN_PROGRESS lease older than this -> requeue
    "PROTECTED_PORTS": (3004, 8003, 27017, 27018),  # frontend / backend / bridge-mongo / business-mongo
}


# ---------------------------------------------------------------------------
# COMPUTE HEDGING / VERIFICATION COMMITTEE (V3 counterparty risk + V5 CDS premium)
# Whitepaper sections 2.4 / 3.1 / 3.2 + V3 addendum section 2.6.
# STATUS: THEORY / PARTIALLY OPERATIONAL.
#   - Committee ROUTING gate is live in dispatcher (Committee/money-math need the
#     V2 hard gate hp>80,mu>75,sigma<15) -- see V2_CREDIT above.
#   - The QUANTITATIVE hedging economics below (how many redundant verifiers K,
#     the CDS-premium <= expected-tail-loss test, the majority-vote weights) are
#     NOT yet code. Surfaced here with None so the gap is explicit, not forgotten.
# CDS rule:  sum_k(alpha * token_cost_k) <= P(tail_risk) * L_catastrophic
# ---------------------------------------------------------------------------
COMPUTE_HEDGING = {
    "CDS_REDUNDANT_VERIFIERS_K": None,   # # of extra verifier agents on a tail-risk ticket
    "CDS_PREMIUM_BUDGET_RATIO": None,    # max review spend as a fraction of P(tail)*L_catastrophic
    "COMMITTEE_VOTE_QUORUM": None,       # min independent verdicts for a money/security merge
    "COMMITTEE_MAJORITY_FRACTION": None, # argmax majority threshold (Faraday-cage grounding)
    # Tickets that ARE eligible for compute hedging (apply premium); everything
    # else must NOT pay it (no hedging on docs / cosmetic / triage):
    "HEDGE_ELIGIBLE_LANES": (
        "T3_trade_insert", "T9_position_reseed", "VaR", "PnL", "NAV", "gross_MV",
        "tenant_isolation", "execution_adapter", "security_boundary", "release_train",
    ),
    # Counterparty risk (V3 2.5): downstream must not blindly trust upstream output.
    "AUTOPSY_REQUIRED_FOR_REQUEUE": True,   # compact RCA, never raw traceback, to next agent
    "BUSINESS_IMPACT_DIFF_REQUIRED_FOR_MONEY": True,
}


# ===========================================================================
# FORMULA / MECHANISM LEDGER -- every governing rule in one place, mapped to
# params, code location, and HONEST wiring status. Status vocabulary refined
# 2026-06-02 per operator correction (do NOT overstate maturity):
#   WIRED              = runs in the live path today
#   PARTIAL_WIRED      = code exists + runs, but not stable on all paths / all failures
#   OPERATIONAL_PARTIAL= enforced as operating practice + some code; not proven all-path
#   WIRED_AS_POLICY    = enforced by governance/human policy; per-script enforcement varies
#   SHADOW_WIRED       = computed + stored as telemetry, does NOT act
#   BUILT_MOSTLY_OFF   = helper used in one live spot, but NOT the main routing logic
#   SKELETON_OFF       = code/skeleton exists but is NOT the production dispatcher
#   DOCUMENTED_*_OFF   = written in addendum; partial/no code; not hot path
#   PROCESS_EXISTS     = exists as a workflow/concept; no strict formula/aggregator
#   THEORY             = whitepaper only, not yet code
#
# >>> HANDOVER WARNING (read before any wiring work): <<<
#   Do NOT assume the V5 expected-utility auction is live. It is a DESIGN TARGET,
#   not the production dispatcher. base_agent.loop still uses ticket_queue.pick_up
#   (priority FIFO). First consolidate params (this file) + add SHADOW scoring to
#   collect 500-1000 replay events; only then consider wiring. "Disk freeze" is
#   operational practice (keepalive disabled, fleet paused, worktrees cleaned),
#   NOT a proven all-path dispatcher hard-freeze -- audit dispatcher.py /
#   base_agent.py / fleet_warden.py before claiming full coverage.
# ===========================================================================
FORMULAS = {
    # ---- V1 SURVIVAL ----
    "hp_guillotine_recycle": {
        "layer": "V1",
        "eq": "alive if HP>=20; soft-recycle if 20<=HP<40 & unleased; guillotine if HP<20",
        "uses": ["DEFAULT_HP", "GUILLOTINE_HP", "SOFT_RECYCLE_HP"],
        "status": "WIRED",
        "code": "ledger.py (HP update, guillotine_self_if_needed, soft_recycle_agent)",
        "note": "HP/guillotine/soft-recycle coded; actual kill/respawn depends on fleet state.",
    },
    # ---- V2 CREDIT AGENT (the agent-as-credit-risk model) ----
    "credit_bayesian_update": {
        "layer": "V2",
        "eq": ("mu' = clamp(mu + k * q * (outcome - expected), 0, 100); "
               "k = clamp(sigma/2.7, 2, 12); "
               "sigma' = sigma*0.92..0.96 if success else min(60, sigma*1.08 + 1.5)"),
        "uses": ["mu", "sigma"],
        "status": "WIRED",
        "code": "ledger.update_domain_skill (ledger.py:702-718)",
        "note": "THE learning rule. mu/sigma move on every outcome -- the live credit/"
                "reward/promotion engine.",
    },
    "credit_p_success": {
        "layer": "V2",
        "eq": "P(success) = 1 / (1 + exp(-0.717 * z)),  z = (mu - difficulty) / max(8, sigma)",
        "uses": ["mu", "sigma", "difficulty"],
        "status": "BUILT_MOSTLY_OFF",
        "code": "ledger._skill_expected (used in skill update) + dispatcher._p_success (auction, OFF)",
        "note": "The sigmoid IS used inside the live skill update; it is NOT the dispatcher "
                "pickup main logic (that path is the OFF auction).",
    },
    "credit_sigma_spread": {
        "layer": "V2/V5",
        "eq": "adjusted_model_cost = base_cost * (1 + GAMMA * sigma^2)",
        "uses": ["GAMMA_SIGMA_SPREAD", "sigma"],
        "status": "DOCUMENTED_PARTIAL_OFF",
        "code": "dispatcher.utility_for (dispatcher.py:460) -- not a live routing factor",
        "note": "Formula written, partial utility wiring; NOT a live dispatcher hard-routing factor.",
    },
    # ---- V3 HAZARD / SRE ----
    "white_noise_filter": {
        "layer": "V3/SRE",
        "eq": "if error in INFRA_NOISE_MARKERS: requeue, no HP/mu penalty; else penalize",
        "uses": ["INFRA_NOISE_MARKERS"],
        "status": "PARTIAL_WIRED",
        "code": "evaluate_failure_noise() + repair logic",
        "note": "Idea + helper exist; not every failed reclassification runs stably "
                "(can itself fail when Mongo is unstable). THE one data-supported finding.",
    },
    "hazard_rate": {
        "layer": "V3",
        "eq": "H = clamp(BASE + a*ctx + b*streak + c*minutes + d*requeue, 0, 1)",
        "uses": ["BASE_HAZARD", "WEIGHT_CONTEXT(a)", "WEIGHT_STREAK(b)",
                 "WEIGHT_MINUTES(c)", "WEIGHT_REQUEUE(d)"],
        "status": "SHADOW_WIRED",
        "code": "ledger.calculate_hazard (ledger.py:273)",
    },
    "structured_autopsy": {
        "layer": "V3",
        "eq": "downstream reads compact RCA, never raw traceback (entropy reduction)",
        "uses": ["AUTOPSY_REQUIRED_FOR_REQUEUE"],
        "status": "WIRED_PARTIAL",
        "code": "RCA/autopsy generation in bridge",
        "note": "Exists + compresses; could be made stricter (e.g. hard 3-line constraint).",
    },
    # ---- V4 PROPULSION ----
    "effective_thrust": {
        "layer": "V4",
        "eq": "effective_thrust = mu * specific_impulse * warm_efficiency - drag_penalty",
        "uses": ["WARM_EFFICIENCY_ALPHA", "WARM_EFFICIENCY_FLOOR", "*_DRAG"],
        "status": "SHADOW_WIRED",
        "code": "ledger.calculate_propulsion (ledger.py:316-330)",
    },
    # ---- V5 OPTIMIZATION (DESIGN TARGET -- not production dispatcher) ----
    "dispatch_utility": {
        "layer": "V5",
        "eq": ("U(i,j) = P(success) * value "
               "- model_cost - risk_penalty - context_penalty - disk_penalty"),
        "uses": ["all V2/V5 params"],
        "status": "SKELETON_OFF",
        "code": "dispatcher.utility_for (dispatcher.py:464) -- NOT the production dispatcher",
        "note": "V5 addendum + skeleton. base_agent.loop still uses FIFO pick_up. "
                "DO NOT assume this is 'built, flip a switch'.",
    },
    "ticket_value": {
        "layer": "V5",
        "eq": "value = business_impact * urgency * (1 + log1p(downstream)) * exp(LAMBDA * age)",
        "uses": ["LAMBDA_STARVATION"],
        "status": "DOCUMENTED_OFF",
        "code": "dispatcher.infer_ticket_features (dispatcher.py:290-294) -- not hot path",
    },
    "systemic_var_skew": {
        "layer": "V5",
        "eq": "risk_skew = exp(money_risk + security_risk + live_risk)",
        "uses": [],
        "status": "DOCUMENTED_OFF",
        "code": "dispatcher.py:299,461 -- concept clear; no fleet-wide enforced budget",
    },
    "disk_critical_freeze": {
        "layer": "V5/SRE",
        "eq": "C_free < 5GB -> hard pause; < 10GB -> no scale-up; penalty = THETA/(free-crit)",
        "uses": ["THETA_DISK_PENALTY", "DISK_CRITICAL_GB", "DISK_YELLOW_GB"],
        "status": "OPERATIONAL_PARTIAL",
        "code": "dispatcher._disk_penalty/freeze (dispatcher.py:405,431) + manual ops practice",
        "note": "We DID hard-pause + disable keepalive + clean worktrees. Whether the "
                "dispatcher hard-freeze covers ALL paths needs an audit of dispatcher/"
                "base_agent/fleet_warden before claiming full coverage.",
    },
    # ---- GOVERNANCE ----
    "operator_gates_t3_t9": {
        "layer": "Governance",
        "eq": "T1/T2/T9 + live-broker tickets blocked until explicit operator release",
        "uses": [],
        "status": "WIRED_AS_POLICY",
        "code": "seeded BLOCKED_OPERATOR; per-script/dispatcher enforcement varies",
        "note": "Repeatedly preserved as policy; verify concrete gate enforcement per script.",
    },
    # ---- THEORY / PROCESS (not formula-wired) ----
    "concurrency_slippage": {
        "layer": "V5",
        "eq": "concurrency_slippage = base_cost * exp(KAPPA * rho),  rho = workers / safe_capacity",
        "uses": ["KAPPA_CONCURRENCY_SLIPPAGE", "SAFE_WORKER_CAPACITY"],
        "status": "THEORY",
        "code": "(absent -- disk/cpu hard gates approximate it today)",
    },
    "fleet_risk_budget": {
        "layer": "V5",
        "eq": "sum(active_ticket_risk * agent_sigma) <= FLEET_RISK_BUDGET",
        "uses": ["FLEET_RISK_BUDGET"],
        "status": "THEORY",
        "code": "(absent -- no enforcement)",
    },
    "cds_premium": {
        "layer": "V3/V5",
        "eq": "sum_k(alpha * token_cost_k) <= P(tail_risk) * L_catastrophic",
        "uses": ["CDS_REDUNDANT_VERIFIERS_K", "CDS_PREMIUM_BUDGET_RATIO"],
        "status": "PROCESS_PRINCIPLE",
        "code": "(committee/chaos/reviewer is a process idea; no formula-scheduled hedging)",
    },
    "committee_vote": {
        "layer": "V3",
        "eq": "y_final = argmax_y sum_k w_k * 1[y_k == y]   (strict K-of-N majority)",
        "uses": ["COMMITTEE_VOTE_QUORUM", "COMMITTEE_MAJORITY_FRACTION"],
        "status": "PROCESS_EXISTS",
        "code": "Committee concept + queue tickets exist; NO strict majority-vote aggregator",
        "note": "Verification Committee is a process, not a wired K-of-N vote program.",
    },
}


# Convenience: every group, in V-stack order.
REGISTRY = {
    "V1_SURVIVAL": V1_SURVIVAL,
    "V2_CREDIT": V2_CREDIT,
    "V3_HAZARD": V3_HAZARD,
    "V4_PROPULSION": V4_PROPULSION,
    "V5_OPTIMIZATION": V5_OPTIMIZATION,
    "V6_REGIME": V6_REGIME,
    "V6_MONEY_CRITICAL": V6_MONEY_CRITICAL,
    "COMPUTE_HEDGING": COMPUTE_HEDGING,
    "PHYSICAL_GATES": PHYSICAL_GATES,
    "FORMULAS": FORMULAS,
}

__all__ = [
    "V1_SURVIVAL", "V2_CREDIT", "V3_HAZARD", "V4_PROPULSION",
    "V5_OPTIMIZATION", "V6_REGIME", "V6_MONEY_CRITICAL", "COMPUTE_HEDGING",
    "PHYSICAL_GATES", "FORMULAS", "REGISTRY",
]


if __name__ == "__main__":
    # Read-only dump; safe to run, no Mongo / no fleet side effects.
    import json
    print(json.dumps(REGISTRY, indent=2, default=str))
