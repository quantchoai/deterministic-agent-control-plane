"""risk_core.py -- the V1-V5 deterministic risk / hazard core.

Standard-library only (math, statistics). No numpy/scipy, no Mongo, no LLM.
These are the deterministic risk and hazard primitives the V1-V5 control plane
is built on. Every function here is pure, O(small), and reproducible.

Covers:
  V2/V3 helpers      : poisson_failure, lognormal_latency
  Tail risk (CVaR)   : cvar (Expected Shortfall) -- the auction prices tail on CVaR
  Non-i.i.d. risk    : factor_decomposition, concentration_check   (vendor correlation)
  Fragility          : fracture_score
  Anti-Goodhart      : value_adjustment
  Curvature          : gamma_acceleration, gamma_acceleration_second_difference
  Floating barrier   : frn_barrier
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
# TAIL RISK -- the auction prices tail on CVaR (Expected Shortfall), unified
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
#   such).
#
#   WHY CVaR, not VaR.  Once the VaR line is breached the AVERAGE loss can run an order
#   of magnitude beyond it. LLM hallucination loss is fat-tailed, so the mass beyond VaR
#   is where the real damage lives. VaR is blind to it; CVaR is the coherent
#   (sub-additive) number that composes correctly across the fleet, so the fleet risk
#   budget is priced on CVaR.
#
#   `cvar()` below is the simple historical order-statistic estimator (stdlib-only,
#   used for shadow telemetry + the backtest number). The hardened Rockafellar-Uryasev
#   + Cornish-Fisher estimator used for any priced decision is part of the
#   private/commercial layer and is not bundled in this open package.
# --------------------------------------------------------------------------- #
def cvar(losses: list[float], alpha: float = 0.95) -> dict:
    """VaR and CVaR (Expected Shortfall) at level alpha -- the tail-risk price.

    VaR  = the alpha-quantile loss (an intermediate threshold only).
    CVaR = the mean loss GIVEN you breached VaR -- the number the auction prices on.

    The fleet risk budget is priced on CVaR, NOT VaR (CVaR is coherent/sub-additive,
    so it composes across the fleet; VaR is blind to the fat tail beyond the quantile).
    This is the simple stdlib historical estimator; the hardened Rockafellar-Uryasev +
    Cornish-Fisher estimator is part of the private/commercial layer."""
    xs = sorted(losses)
    if not xs:
        return {"error": "no losses"}
    idx = min(len(xs) - 1, int(math.ceil(alpha * len(xs))) - 1)
    var = xs[idx]
    tail = xs[idx:]
    return {"alpha": alpha, "VaR": var, "CVaR": (sum(tail) / len(tail)) if tail else var}


# --------------------------------------------------------------------------- #
# Non-i.i.d. systemic risk -- factor model (vendor/model correlation)
# Failures are vendor-correlated (shared RLHF/infra), NOT independent. Decompose
# into systematic (shared factor) vs idiosyncratic.
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
# Fragility -- fracture score
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
# than the level alone. Cheap (one second-difference).
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
# degradation -> bar rises -> high-risk work auto-freezes.
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
# gamma_acceleration_second_difference: concrete second-difference of mu.
# Distinct from gamma_acceleration(mu_history, warn_threshold) above (kept for
# back-compat); this is the registry-anchored second difference. Deterministic, O(1).
# It only ever raises a flag or nudges sigma; it NEVER guillotines (probabilistic signal).
# --------------------------------------------------------------------------- #
def gamma_acceleration_second_difference(mu_history: list[float]) -> dict:
    """Plain second difference of mu (newest-last, >=3 points): a = mu[-1]-2mu[-2]+mu[-3].

    Deterministic, O(1) on the tail. Negative => learning collapse accelerating; positive
    => recovery accelerating. Severity here is capped at 'flag' -- a probabilistic curvature
    reading may only flag / adjust sigma, never drive an irreversible action.
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
    print("anti-goodhart (declined)       :", value_adjustment(100, 80, True, 0.9))
    print("gamma accel (90,88,82)         :", gamma_acceleration([90, 88, 82]))
    print("frn bar calm (sys=0.03)        :", frn_barrier(0.86, 0.03)["bar"])
    print("frn bar vendor-stress (sys=0.6):", frn_barrier(0.86, 0.6)["bar"])
    print("gamma 2nd-diff (90,88,82)      :", gamma_acceleration_second_difference([90, 88, 82]))
