"""V6 advanced models -- PRIVATE / SHADOW + OFFLINE only.

Standard-library only (math, statistics). No numpy/scipy, no Mongo, no LLM.
These are the deterministic models the V6 mandate governs. NONE of these are wired
into live dispatch. They run in the slow/offline loop, enter as SHADOW telemetry,
and earn promotion via the V6 gate (see AGENT_MGMT_V6_..._MANDATE).

Covers:
  V2/V3 helpers      : poisson_failure, lognormal_latency
  Tail risk (§5)     : cvar (Expected Shortfall) -- the auction prices tail on CVaR,
                       unified (one definition; hardened in modelver_cvar_budget)
  Non-i.i.d. risk    : factor_decomposition, concentration_check   (vendor correlation)
  Fragility          : fracture_score                              (consolidation candidate)
  Grounding          : oracle_tier                                  (anti-gaming)
  Anti-Goodhart      : value_adjustment

V1-V5 production engines already live in ledger.py / dispatcher.py -- not reimplemented here.
"""
from __future__ import annotations

import math
import statistics as st


# --------------------------------------------------------------------------- #
# small stats helpers (stdlib only)
# --------------------------------------------------------------------------- #
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _inv_norm(p: float) -> float:
    """Acklam's inverse standard-normal CDF approximation."""
    p = min(max(p, 1e-9), 1 - 1e-9)
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


# --------------------------------------------------------------------------- #
# V3 -- Poisson failure model: probability of failure CLUSTERING in a window
# --------------------------------------------------------------------------- #
def poisson_failure(lmbda: float, k: int) -> float:
    """P(>= k failures | rate lmbda) for the next window. Closed form, microseconds.
    lmbda = EWMA of recent failures per window. Gate work if this exceeds the budget."""
    if k <= 0:
        return 1.0
    cdf_lt_k = sum(math.exp(-lmbda) * lmbda**i / math.factorial(i) for i in range(k))
    return max(0.0, min(1.0, 1.0 - cdf_lt_k))


# --------------------------------------------------------------------------- #
# V4/V5 -- Log-normal latency (right-skewed, > 0). Normal is wrong for latency.
# --------------------------------------------------------------------------- #
def lognormal_latency(samples: list[float]) -> dict:
    """Fit log-normal to latency samples; return tail quantiles + an exceedance fn."""
    xs = [x for x in samples if x and x > 0]
    if len(xs) < 3:
        return {"error": "need >=3 positive samples"}
    logs = [math.log(x) for x in xs]
    mu, sigma = st.mean(logs), (st.pstdev(logs) or 1e-6)
    return {
        "mu": mu, "sigma": sigma,
        "p50": math.exp(mu),
        "p95": math.exp(mu + sigma * _inv_norm(0.95)),
        "p99": math.exp(mu + sigma * _inv_norm(0.99)),
        # P(latency > t):
        "exceed": (lambda t: 1.0 - _norm_cdf((math.log(t) - mu) / sigma)) if sigma else (lambda t: 0.0),
    }


# =========================================================================== #
# §5  TAIL RISK -- the auction prices tail on CVaR (Expected Shortfall), unified
# =========================================================================== #
# TEXT == CODE (read this before trusting any tail number):
#
#   The fleet's tail-risk price is ONE quantity -- CVaR_alpha (Expected Shortfall),
#   the mean loss CONDITIONAL on breaching the VaR line. There is no second,
#   competing "VaR mode" the auction silently falls back to. VaR appears only as an
#   intermediate (the alpha-quantile threshold CVaR is measured beyond); it is never
#   the priced number. This is a UNIFIED CVaR pricing, not a migration -- do NOT read
#   it as "we switched from VaR to CVaR" (there was never a live VaR budget to switch
#   from; the legacy hot-path `risk_skew = exp(money+security+live)` is a coarse
#   precomputed PROXY, distinct from the CVaR tail-risk layer, and is documented as
#   such -- see §5.3).
#
#   §5.1  WHY CVaR, not VaR.  On the real TROUBLE set (SHADOW_SCORER_FINDINGS_2026-
#         06-03, Insight B) VaR(95%) = 21.8 vs CVaR(95%) = 337  (~15x). Once the VaR
#         line is breached the AVERAGE loss is ~15x the line. LLM hallucination loss
#         is fat-tailed, so the 1% beyond VaR is where the real damage lives. VaR is
#         blind to it; CVaR is the coherent (sub-additive) number that composes
#         correctly across the fleet, so the fleet risk budget and the CDS compute-
#         hedge premium are BOTH priced on CVaR.
#
#   §5.2  ONE definition, two implementations (same shape, no divergence).
#         * `cvar()` below is the simple historical order-statistic estimator
#           (stdlib-only, used for shadow telemetry + the backtest number).
#         * `modelver_cvar_budget` HARDENS the SAME definition into the layer the V5
#           risk-budgeted auction and the V3/V5 compute-hedge economics actually read:
#           Rockafellar-Uryasev historical CVaR (unbiased when alpha*n is non-integer)
#           + a Cornish-Fisher parametric fallback for fat tails + a one-sided upper
#           confidence bound. Both agree on a symmetric sample by construction; the
#           hardened layer is the source of truth for any priced decision.
#
#   §5.3  PROXY vs PRICE (the hot/slow split -- do not conflate).  The live O(1)
#         dispatcher hot path multiplies a precomputed `risk_skew = exp(money+
#         security+live)` severity proxy; it is NOT CVaR and never claims to be. The
#         CVaR tail price is computed in the SLOW loop (the materializer), written
#         into the snapshot, and the auction reads it O(1). Promotion status:
#         cvar_budget is MONEY-classed and SHADOW until calibrated -- it informs the
#         budget gate but, per the V6 gate (D4), may not auto-advance to FULL.
# --------------------------------------------------------------------------- #
def cvar(losses: list[float], alpha: float = 0.95) -> dict:
    """VaR and CVaR (Expected Shortfall) at level alpha -- the §5 tail-risk price.

    VaR  = the alpha-quantile loss (an intermediate threshold only).
    CVaR = the mean loss GIVEN you breached VaR -- the number the auction prices on.

    The fleet risk budget and the CDS compute-hedge premium are priced on CVaR, NOT
    VaR (CVaR is coherent/sub-additive, so it composes across the fleet; VaR is blind
    to the fat tail beyond the quantile). This is the simple stdlib historical
    estimator; `modelver_cvar_budget` hardens the SAME definition (Rockafellar-
    Uryasev + Cornish-Fisher + an upper CI bound) for any priced decision. See §5."""
    xs = sorted(losses)
    if not xs:
        return {"error": "no losses"}
    idx = min(len(xs) - 1, int(math.ceil(alpha * len(xs))) - 1)
    var = xs[idx]
    tail = xs[idx:]
    return {"alpha": alpha, "VaR": var, "CVaR": (sum(tail) / len(tail)) if tail else var}


# --------------------------------------------------------------------------- #
# Non-i.i.d. systemic risk -- factor model (vendor/model correlation)
# Answers the commenter's point: failures are vendor-correlated (shared RLHF/infra),
# NOT independent. Decompose into systematic (shared factor) vs idiosyncratic.
# --------------------------------------------------------------------------- #
def factor_decomposition(observations: list[dict]) -> dict:
    """observations: [{"factor": "<vendor/model>", "loss": float}, ...]
    Splits variance into systematic (between-factor) vs idiosyncratic (within-factor).
    High systematic share => a vendor shock hits many agents at once => concentration risk."""
    if not observations:
        return {"error": "no observations"}
    groups: dict[str, list[float]] = {}
    for o in observations:
        groups.setdefault(str(o.get("factor", "?")), []).append(float(o.get("loss", 0.0)))
    all_losses = [x for g in groups.values() for x in g]
    grand = st.mean(all_losses)
    factor_means = {f: st.mean(v) for f, v in groups.items()}
    n = len(all_losses)
    systematic_var = sum(len(v) * (st.mean(v) - grand) ** 2 for v in groups.values()) / n
    idio_var = sum((x - factor_means[f]) ** 2 for f, v in groups.items() for x in v) / n
    total = systematic_var + idio_var or 1e-9
    return {
        "factor_means": factor_means, "grand_mean": grand,
        "systematic_var": systematic_var, "idiosyncratic_var": idio_var,
        "systematic_share": systematic_var / total,
        "interpretation": ("high systematic_share => failures move together by vendor; "
                           "diversify vendors + apply concentration cap"),
    }


def concentration_check(assignment_vendors: list[str], cap: float = 0.40) -> dict:
    """Cap fleet exposure to any single vendor/architecture on critical work, so one
    vendor's collective degradation cannot take everything down. O(1), boring, high-value."""
    if not assignment_vendors:
        return {"error": "no assignments"}
    n = len(assignment_vendors)
    frac = {}
    for v in assignment_vendors:
        frac[v] = frac.get(v, 0) + 1
    frac = {v: c / n for v, c in frac.items()}
    top = max(frac, key=frac.get)
    return {"fractions": frac, "max_vendor": top, "max_frac": frac[top], "cap": cap,
            "violation": frac[top] > cap,
            "action": (f"throttle {top}: route new critical work to other vendors"
                       if frac[top] > cap else "ok")}


# --------------------------------------------------------------------------- #
# Fragility -- fracture score (CONSOLIDATION candidate; watch double-counting)
# --------------------------------------------------------------------------- #
_FRACTURE_W = {"failed_requeue": 0.15, "critical_file_touch": 0.20, "ci_failure": 0.25,
               "stale_lease_minutes": 0.01, "traceback_noise_kb": 0.02, "cross_domain_deps": 0.10}


def fracture_score(features: dict, weights: dict | None = None) -> float:
    """Single interpretable 'likely to break' score. NOTE: most inputs already exist in
    V3 hazard / tail_risk -- if adopted, NET OUT those terms, do not stack (double-count)."""
    w = weights or _FRACTURE_W
    s = sum(w.get(k, 0.0) * float(features.get(k, 0.0)) for k in w)
    return max(0.0, min(1.0, s))


# --------------------------------------------------------------------------- #
# Grounding / anti-gaming -- the Oracle ladder
# Decides the STRONGEST verification available, and whether a HARD action (kill) is
# allowed. Rule: only deterministic oracles may drive irreversible actions.
# --------------------------------------------------------------------------- #
_TIERS = ["deterministic", "structural", "cross_source", "consensus", "llm_judge", "none"]


def oracle_tier(task: dict) -> dict:
    """task flags: has_tests, has_schema, has_authoritative_source, has_redundant_agents.
    Returns the strongest tier present + whether hard actions (HP/guillotine) are allowed."""
    if task.get("has_tests") or task.get("has_invariant") or task.get("has_compiler"):
        tier = "deterministic"
    elif task.get("has_schema") or task.get("has_numeric_bounds"):
        tier = "structural"
    elif task.get("has_authoritative_source"):
        tier = "cross_source"
    elif task.get("has_redundant_agents"):
        tier = "consensus"
    elif task.get("has_llm_judge"):
        tier = "llm_judge"
    else:
        tier = "none"
    hard_ok = tier in ("deterministic",)             # only deterministic may kill
    may_penalize = tier in ("deterministic", "structural", "cross_source")
    return {"tier": tier, "hard_action_allowed": hard_ok, "may_penalize_score": may_penalize,
            "note": ("no deterministic oracle => no auto-guillotine on 'hallucination'; "
                     "consensus/judge may only RAISE sigma or route to committee")}


# --------------------------------------------------------------------------- #
# Anti-Goodhart -- stop agents gaming the metric (refuse hard tasks / safe garbage)
# --------------------------------------------------------------------------- #
def value_adjustment(base_value: float, difficulty: float, declined: bool,
                     info_content: float) -> float:
    """Reward difficulty/impact; COST refusals and low-information output so 'safe
    garbage' is never a winning strategy."""
    if declined:
        return -abs(base_value) * 0.5                # declining an eligible task costs
    if info_content < 0.2:
        return base_value * 0.1                      # near-empty output earns almost nothing
    return base_value * (1.0 + 0.3 * min(1.0, difficulty / 100.0))


# --------------------------------------------------------------------------- #
# Gamma -- the acceleration of collapse (second difference of mu). LLMs degrade
# non-linearly near a context/logic wall; "falling AND accelerating" warns earlier
# than the level alone. Cheap (one second-difference). SHADOW.
# --------------------------------------------------------------------------- #
def gamma_acceleration(mu_history: list[float], warn_threshold: float = -2.0) -> dict:
    """mu_history newest-last (>=3 points). Negative accel = collapse accelerating."""
    if len(mu_history) < 3:
        return {"accel": 0.0, "warning": False, "note": "need >=3 points"}
    a = mu_history[-1] - 2 * mu_history[-2] + mu_history[-3]
    return {"accel": round(a, 3), "warning": a <= warn_threshold,
            "note": "negative accel => pre-empt (soft-recycle) before the guillotine line"}


# --------------------------------------------------------------------------- #
# FRN floating barrier -- the factor model made ACTIONABLE. The eligibility bar is
# a floating coupon: base + sensitivity * systemic stress (from factor_decomposition's
# systematic_share) + the agent's own spread (sigma). Calm market -> low bar; vendor-wide
# degradation -> bar rises -> high-risk work auto-freezes. SHADOW until calibrated.
# --------------------------------------------------------------------------- #
def frn_barrier(base_bar: float, systematic_share: float, agent_sigma: float = 0.0,
                kappa: float = 0.30, gamma_spread: float = 0.0, cap: float = 0.98) -> dict:
    """Floating eligibility bar = base_bar + kappa*systematic_share + gamma_spread*sigma^2.
    base_bar e.g. the critical-lane min success prob (0.86). Returns the live bar to clear."""
    bar = base_bar + kappa * max(0.0, systematic_share) + gamma_spread * (agent_sigma ** 2)
    return {"bar": round(min(cap, bar), 4), "base": base_bar,
            "systemic_lift": round(kappa * systematic_share, 4),
            "note": "vendor stress (systematic_share) raises the bar -> auto-tighten / freeze"}


# --------------------------------------------------------------------------- #
# gamma_acceleration: concrete second-difference of mu (per the wire-phase spec).
# Distinct from the legacy gamma_acceleration(mu_history, warn_threshold) above which
# is kept for back-compat; this is the registry-anchored second difference that the
# V6 gate consumes for the lambda/oracle-ladder learning-curvature model. SHADOW: it
# only ever raises a flag or nudges sigma; it NEVER guillotines (probabilistic signal).
# --------------------------------------------------------------------------- #
def gamma_acceleration_second_difference(mu_history: list[float]) -> dict:
    """Plain second difference of mu (newest-last, >=3 points): a = mu[-1]-2mu[-2]+mu[-3].

    Deterministic, O(1) on the tail. Negative => learning collapse accelerating; positive
    => recovery accelerating. This is the HOT-path-cheap read; any robust/denoised fit
    (modelver_lambda_controller.d2_mu_accel) belongs in the SLOW loop that writes a
    snapshot. Severity here is capped at 'flag' -- a probabilistic curvature reading may
    only flag / adjust sigma, per the oracle-determinism severity match.
    """
    h = [float(v) for v in mu_history if v is not None]
    if len(h) < 3:
        return {"accel": 0.0, "warning": False, "severity": "none", "note": "need >=3 points"}
    a = h[-1] - 2.0 * h[-2] + h[-3]
    warn = a <= -2.0
    return {
        "accel": round(a, 4),
        "warning": warn,
        "severity": "flag" if warn else "none",  # probabilistic => never above 'flag'
        "note": "second difference of mu; collapse-accel flags soft pre-emption, never a kill",
    }


# =========================================================================== #
# V6 PROMOTION GATE -- the wire phase.
#
# Consumes EVERY V6 model (the 7 modelver_* + fracture + frn_barrier + the concrete
# gamma_acceleration second-difference) and decides, per model, whether it may advance
# along the promotion state machine:
#
#       SHADOW  ->  CANARY  ->  FULL
#
#   SHADOW : computes + logs telemetry; does NOT move live routing/credit. Entry state
#            for everything. The score is collected for replay evidence only.
#   CANARY : the model's signal is allowed to act, but only at its severity ceiling
#            (see oracle-determinism severity match) and only on a limited blast radius.
#   FULL   : the model acts on the live hot path within its severity ceiling.
#
# LOAD-BEARING DISCIPLINE (must hold for every decision):
#   D1 SHADOW-FIRST     : everything enters SHADOW; promotion is one step at a time
#                         (SHADOW->CANARY->FULL); no skipping CANARY.
#   D2 SEVERITY MATCH   : a model's MAX allowed action severity equals its oracle
#                         determinism tier. deterministic -> may 'guillotine'/'hard';
#                         structural/cross_source -> may 'penalize' (adjust mu/sigma);
#                         probabilistic (consensus/judge/none) -> may only 'flag'/sigma.
#                         A probabilistic model NEVER earns a guillotine, at any state.
#   D3 ANTI-GOODHART    : refuse promotion when the shadow metrics look gamed --
#                         degenerate sample, perfect-looking agreement on too few obs,
#                         or info_content collapse. A metric that is too clean to be
#                         true does not promote.
#   D4 MONEY/SECURITY   : a money- or security-classed model may NEVER go SHADOW->FULL,
#                         and may NEVER auto-advance to FULL at all. CANARY is the
#                         automatic ceiling; FULL requires an explicit operator release
#                         (operator_release=True) AND the model already at CANARY.
#   D5 SLOW/HOT SPLIT   : promotion math runs here (slow loop). It emits a decision the
#                         hot path reads O(1); the gate itself does no calculus on the
#                         hot path and calls no model's heavy fit.
# =========================================================================== #
from quant.governance.governance_params import (
    V2_CREDIT as _V2,
    V3_HAZARD as _V3,
    V6_REGIME as _V6R,
)

# Promotion state ladder (strict order; index gives the rank).
PROMOTION_STATES = ("shadow", "canary", "full")

# Action-severity ladder, weakest -> strongest. A model may act only up to the severity
# its oracle-determinism tier permits (D2).
SEVERITY_ORDER = ("none", "flag", "sigma", "penalize", "hard")

# Map an oracle-determinism tier -> the STRONGEST action severity it may ever drive.
#   deterministic ground truth          -> may drive an irreversible hard action.
#   structural / cross_source           -> may penalize (move mu/sigma), never kill.
#   consensus / llm_judge / none (prob.) -> may only flag / raise sigma.
_TIER_MAX_SEVERITY = {
    "deterministic": "hard",
    "structural": "penalize",
    "cross_source": "penalize",
    "consensus": "sigma",
    "llm_judge": "flag",
    "none": "flag",
}

# Per-model registry: oracle-determinism tier + money/security class + the shadow-metric
# thresholds each model must clear to advance. Thresholds are sourced from
# governance_params where one exists (single source of truth), else a conservative local
# default. `min_samples` guards D3 (no promotion on a degenerate sample). `kind` selects
# the readiness predicate.
#
# canary_success_criteria (per-model, mandatory for canary soak):
#   min_soak_obs      : minimum observations that must accumulate while at CANARY before any
#                       exit (promote->full or rollback->shadow) is evaluated.
#   metric            : the metric key to evaluate during canary soak (must be in the
#                       shadow snapshot, e.g. "calibration_error").
#   metric_threshold  : the metric value the model must hold (calibration_error <=
#                       max_calib_error) throughout the soak window. Same as the shadow
#                       threshold so there is no relaxation at canary.
#   rollback_consecutive_windows : if this many CONSECUTIVE soak windows breach
#                       2x metric_threshold, the model is automatically reverted to
#                       SHADOW. This is the explicit rollback trigger (not a "vague" hold).
#   description       : human-readable note for audit logs.
#
# The canary_success_criteria is READ by v6_promote.py's auto-promotion path and enforced
# as the canary exit condition. It is NOT checked by the v6_promote() gate itself
# (which only governs shadow->canary advancement); canary->full advancement requires
# the soak metrics to have been accumulated and passed by the runner.
_MODEL_REGISTRY: dict[str, dict] = {
    # --- the 7 modelver_* ---
    "poisson_failure": {
        "tier": "structural",          # count GOF is a structural/numeric-bounds oracle
        "money_security": False,
        "min_samples": int(_V3["CALIBRATION_MIN_EVENTS"]),
        "min_agreement": 0.80,
        "max_calib_error": 0.10,
        "kind": "calibration",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.10,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.10 must hold; "
                "2 consecutive windows >0.20 (2x threshold) reverts to shadow"
            ),
        },
    },
    "gamma_latency": {
        "tier": "structural",
        "money_security": False,
        "min_samples": 200,
        "min_agreement": 0.80,
        "max_calib_error": 0.12,
        "kind": "calibration",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.12,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.12 must hold; "
                "2 consecutive windows >0.24 reverts to shadow"
            ),
        },
    },
    "cvar_budget": {
        "tier": "deterministic",       # a realized-loss recon is deterministic ground truth
        "money_security": True,        # tail risk budget -> MONEY. D4 applies.
        "min_samples": 300,
        "min_agreement": 0.90,
        "max_calib_error": 0.05,
        "kind": "calibration",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.05,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.05 must hold "
                "(MONEY: canary->full always operator-gated regardless of soak result); "
                "2 consecutive windows >0.10 reverts to shadow"
            ),
        },
    },
    "correlation_regime": {
        "tier": "consensus",           # eigen-share regime read is probabilistic
        "money_security": False,
        "min_samples": int(_V6R["MIN_TOTAL_OBS"]),
        "min_agreement": 0.70,
        "max_calib_error": 0.20,
        "kind": "regime",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.20,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.20 must hold; "
                "2 consecutive windows >0.40 reverts to shadow"
            ),
        },
    },
    "oracle_ladder": {
        "tier": "deterministic",       # the grounding model itself is deterministic
        "money_security": True,        # it gates HP/guillotine eligibility -> security.
        "min_samples": 100,
        "min_agreement": 0.95,
        "max_calib_error": 0.05,
        "kind": "agreement",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.05,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.05 must hold "
                "(SECURITY: canary->full always operator-gated); "
                "2 consecutive windows >0.10 reverts to shadow"
            ),
        },
    },
    "lambda_controller": {
        "tier": "consensus",           # learning-accel curvature is probabilistic
        "money_security": False,
        "min_samples": 50,
        "min_agreement": 0.70,
        "max_calib_error": 0.20,
        "kind": "agreement",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.20,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.20 must hold; "
                "2 consecutive windows >0.40 reverts to shadow"
            ),
        },
    },
    "ucb_sandbox": {
        "tier": "structural",          # bandit bound is a numeric-bounds oracle
        "money_security": False,
        "min_samples": 100,
        "min_agreement": 0.80,
        "max_calib_error": 0.15,
        "kind": "agreement",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.15,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.15 must hold; "
                "2 consecutive windows >0.30 reverts to shadow"
            ),
        },
    },
    # --- fracture + frn_barrier + gamma_acceleration ---
    "fracture": {
        "tier": "structural",          # weighted feature score, numeric bounds
        "money_security": False,
        "min_samples": int(_V3["CALIBRATION_MIN_EVENTS"]),
        "min_agreement": 0.75,
        "max_calib_error": 0.15,
        "kind": "agreement",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.15,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.15 must hold; "
                "2 consecutive windows >0.30 reverts to shadow"
            ),
        },
    },
    "frn_barrier": {
        "tier": "deterministic",       # the bar drives lane eligibility/freeze -> security
        "money_security": True,
        "min_samples": int(_V6R["MIN_TOTAL_OBS"]),
        "min_agreement": 0.90,
        "max_calib_error": 0.05,
        "kind": "agreement",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.05,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.05 must hold "
                "(SECURITY: canary->full always operator-gated); "
                "2 consecutive windows >0.10 reverts to shadow"
            ),
        },
    },
    "gamma_acceleration": {
        "tier": "consensus",           # second-difference curvature is probabilistic
        "money_security": False,
        "min_samples": 30,
        "min_agreement": 0.70,
        "max_calib_error": 0.25,
        "kind": "agreement",
        "canary_success_criteria": {
            "min_soak_obs": 500,
            "metric": "calibration_error",
            "metric_threshold": 0.25,
            "rollback_consecutive_windows": 2,
            "description": (
                "canary soak: >=500 observations; calibration_error<=0.25 must hold; "
                "2 consecutive windows >0.50 reverts to shadow"
            ),
        },
    },
}


def severity_ceiling(model_id: str) -> str:
    """The strongest action severity `model_id` may EVER drive, from its determinism tier.

    This is the oracle-determinism severity match: a probabilistic model is hard-capped at
    'flag'/'sigma' regardless of how good its shadow metrics look -- it can never guillotine.
    """
    spec = _MODEL_REGISTRY.get(model_id)
    if spec is None:
        return "none"
    return _TIER_MAX_SEVERITY.get(spec["tier"], "flag")


def _next_state(state: str) -> str:
    i = PROMOTION_STATES.index(state)
    return PROMOTION_STATES[min(i + 1, len(PROMOTION_STATES) - 1)]


def _anti_goodhart_block(spec: dict, m: dict) -> str | None:
    """D3: return a reason string if the shadow metrics look gamed, else None.

    Gaming patterns refused:
      * degenerate sample (n below the model's calibration floor);
      * 'too clean': perfect/near-perfect agreement on a sample too small to trust
        (>=0.999 agreement with n < 5x the floor -> suspicious, hold);
      * info_content collapse (the model is scoring near-empty inputs, like the
        anti-Goodhart 'safe garbage' guard in value_adjustment / oracle_ladder).
    """
    n = int(m.get("n", 0))
    if n < spec["min_samples"]:
        return f"degenerate sample: n={n} < min_samples={spec['min_samples']}"
    agree = float(m.get("agreement", 0.0))
    if agree >= 0.999 and n < 5 * spec["min_samples"]:
        return (f"suspiciously perfect agreement ({agree:.3f}) on a thin sample "
                f"(n={n} < 5x floor) -- holding for anti-Goodhart")
    info = float(m.get("info_content", 1.0))
    if info < 0.2:
        return f"info_content collapse ({info:.2f}<0.2): scoring near-empty signal"
    return None


def _metrics_ready(spec: dict, m: dict) -> tuple[bool, str]:
    """Whether the shadow metrics clear this model's promotion thresholds (D-readiness)."""
    agree = float(m.get("agreement", 0.0))
    if agree < spec["min_agreement"]:
        return False, f"agreement {agree:.3f} < required {spec['min_agreement']:.2f}"
    if spec["kind"] in ("calibration", "regime"):
        ce = float(m.get("calibration_error", 1.0))
        if ce > spec["max_calib_error"]:
            return False, f"calibration_error {ce:.3f} > max {spec['max_calib_error']:.2f}"
    if not bool(m.get("stable", True)):
        return False, "shadow signal not stable across replay windows"
    return True, "metrics clear thresholds"


def v6_promote(model_id: str, shadow_metrics: dict) -> dict:
    """Decide whether `model_id` may advance one step along SHADOW->CANARY->FULL.

    shadow_metrics keys (all optional; missing -> conservative default):
        current_state    : 'shadow'|'canary'|'full'   (default 'shadow')
        n                : # replay/shadow observations collected (default 0)
        agreement        : agreement vs deterministic ground truth in [0,1] (default 0)
        calibration_error: |empirical - predicted| in [0,1] (default 1.0; calib models)
        info_content     : signal information content in [0,1] (default 1.0)
        stable           : bool, signal stable across replay windows (default True)
        operator_release : bool, explicit operator sign-off for money/security FULL

    Returns a decision dict: {model_id, decision, from_state, to_state, severity_ceiling,
    money_security, reasons, audit}. `decision` in {'promote','hold','reject','blocked'}.

    Pure + deterministic + O(1): NO model fit is called here. The slow loop computes the
    shadow metrics and writes them into a snapshot; this gate only reads that snapshot.
    """
    spec = _MODEL_REGISTRY.get(model_id)
    state = str(shadow_metrics.get("current_state", "shadow")).lower()
    if state not in PROMOTION_STATES:
        state = "shadow"
    ceiling = severity_ceiling(model_id) if spec else "none"
    money_sec = bool(spec and spec["money_security"])

    base = {
        "model_id": model_id,
        "from_state": state,
        "to_state": state,            # default: no movement
        "severity_ceiling": ceiling,
        "money_security": money_sec,
        "decision": "hold",
        "reasons": [],
        "audit": dict(shadow_metrics),
    }

    if spec is None:
        base["decision"] = "reject"
        base["reasons"].append(f"unknown model_id '{model_id}': not in V6 registry")
        return base

    # Already at the top.
    if state == "full":
        base["decision"] = "hold"
        base["reasons"].append("already FULL; no further promotion")
        return base

    target = _next_state(state)

    # D4: money/security guardrails. Evaluate BEFORE readiness so the gate can never be
    # talked into a forbidden jump by good-looking metrics.
    if money_sec:
        # Never SHADOW->FULL (single-step machine already prevents the jump, but assert it).
        if state == "shadow" and target == "full":
            base["decision"] = "blocked"
            base["reasons"].append("money/security model may not go shadow->full")
            return base
        # Auto-advance to FULL is forbidden; FULL needs explicit operator release.
        if target == "full" and not bool(shadow_metrics.get("operator_release", False)):
            base["decision"] = "blocked"
            base["reasons"].append(
                "money/security model requires explicit operator_release for FULL; "
                "CANARY is the automatic ceiling")
            return base

    # D3: anti-Goodhart -- refuse gamed/degenerate metrics.
    gh = _anti_goodhart_block(spec, shadow_metrics)
    if gh is not None:
        base["decision"] = "reject"
        base["reasons"].append(gh)
        return base

    # Readiness against per-model thresholds.
    ready, why = _metrics_ready(spec, shadow_metrics)
    if not ready:
        base["decision"] = "hold"
        base["reasons"].append(why)
        return base

    # All gates clear -> advance exactly one step.
    base["decision"] = "promote"
    base["to_state"] = target
    base["reasons"].append(f"metrics clear thresholds; advance {state}->{target}")
    base["reasons"].append(
        f"action capped at severity '{ceiling}' (oracle tier '{spec['tier']}')")
    if money_sec and target == "canary":
        base["reasons"].append("money/security: CANARY is the auto-ceiling; FULL is operator-gated")
    return base


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("poisson P(>=3 | lmbda=1.2)     :", round(poisson_failure(1.2, 3), 4))
    print("lognormal latency p99          :", round(lognormal_latency([80, 120, 95, 300, 110, 90, 600, 130])["p99"], 1))
    print("cvar vs var (95%)              :", {k: round(v, 1) for k, v in cvar([1, 2, 3, 4, 5, 50, 80], 0.95).items() if k != "alpha"})
    fd = factor_decomposition([
        {"factor": "openai", "loss": 9}, {"factor": "openai", "loss": 8},
        {"factor": "anthropic", "loss": 2}, {"factor": "anthropic", "loss": 1},
        {"factor": "deepseek", "loss": 3}])
    print("factor systematic_share        :", round(fd["systematic_share"], 3))
    print("concentration (cap .4)         :", concentration_check(["openai"]*6 + ["anthropic"]*4))
    print("fracture score                 :", round(fracture_score({"ci_failure": 1, "failed_requeue": 2, "cross_domain_deps": 1}), 3))
    print("oracle (fuzzy, judge only)     :", oracle_tier({"has_llm_judge": True})["hard_action_allowed"])
    print("oracle (has tests)             :", oracle_tier({"has_tests": True})["hard_action_allowed"])
    print("anti-goodhart (declined)       :", value_adjustment(100, 80, True, 0.9))
    print("gamma accel (90,88,82)         :", gamma_acceleration([90, 88, 82]))
    print("frn bar calm (sys=0.03)        :", frn_barrier(0.86, 0.03)["bar"])
    print("frn bar vendor-stress (sys=0.6):", frn_barrier(0.86, 0.6)["bar"])
    print("-- V6 promotion gate --")
    _ok = dict(n=600, agreement=0.85, calibration_error=0.05, info_content=0.9, stable=True)
    print("poisson shadow->canary         :", v6_promote("poisson_failure", _ok)["decision"],
          v6_promote("poisson_failure", _ok)["to_state"])
    print("cvar money shadow->full blocked:", v6_promote("cvar_budget", {**_ok, "n": 400,
          "current_state": "canary"})["decision"])
    print("regime severity ceiling        :", severity_ceiling("correlation_regime"))
    print("oracle severity ceiling        :", severity_ceiling("oracle_ladder"))
    print("gamma 2nd-diff (90,88,82)      :", gamma_acceleration_second_difference([90, 88, 82]))
