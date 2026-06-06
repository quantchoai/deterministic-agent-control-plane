"""modelver_cvar_budget.py -- VaR / CVaR(alpha) tail-risk layer + CVaR risk budget.

PRIVATE workspace module (v1-v6 agent-management control plane). DETERMINISTIC,
no LLM, no Mongo, no fleet side effects. Pure-Python with a closed-form fallback;
numpy/scipy are used only for vectorised speed when importable and never for any
nondeterministic operation.

WHY THIS EXISTS
---------------
The shadow scorer (SHADOW_SCORER_FINDINGS_20260603.md, Insight B) measured, on the
real historical TROUBLE set, VaR(95%) = 21.8 vs CVaR(95%) = 337  ->  CVaR ~= 15x VaR.
Once you breach the VaR line the *average* loss is ~15x the line itself. VaR is the
wrong number to size a budget on; CVaR (Expected Shortfall) is the coherent one.

This module hardens the simple historical estimator in governance_v6.cvar into a
proper tail-risk layer used by the V5 risk-budgeted auction and the V3/V5 compute-
hedging (CDS-premium) economics:

  * VaR_alpha(L)      -- the alpha-quantile of the portfolio LOSS distribution.
  * CVaR_alpha(L)     -- Expected Shortfall = E[L | L >= VaR_alpha]; coherent,
                         sub-additive, the number a risk budget must be priced on.
  * Two estimators:
      - historical (nonparametric, order statistics; matches the backtest number),
      - parametric Cornish-Fisher (mean/std + skew/kurt adjusted quantile, with a
        closed-form Gaussian->ES correction). Cornish-Fisher captures fat tails that
        a plain Gaussian VaR misses, without needing a full historical sample.
  * budget_ok(positions, alpha, limit) -- the fleet/portfolio CVaR gate: aggregate
    the per-position loss legs, compute CVaR_alpha, and pass iff CVaR <= limit.

MATH
----
LOSS CONVENTION: L is a loss (bigger = worse). For an auction/fleet "position" the
loss leg is severity * tail_skew (the same proxy the backtest uses). The portfolio
loss is the sum of the legs; the empirical loss sample is the per-scenario sum.

VaR (historical): with sorted losses x_(1) <= ... <= x_(n) and confidence alpha,
    VaR_alpha = x_(k),   k = ceil(alpha * n)            (upper / conservative index)
We also expose a linearly-interpolated quantile (numpy-style) for smoothness.

CVaR (historical, Rockafellar-Uryasev): the coherent estimator that does NOT simply
average the points >= VaR (which is biased when alpha*n is non-integer):
    CVaR_alpha = VaR_alpha + (1/(1-alpha)) * mean( max(L - VaR_alpha, 0) )
This equals the Rockafellar-Uryasev convex program optimum and is exact for the
empirical distribution. CVaR >= VaR always (the +nonneg term).

CVaR (parametric, Cornish-Fisher): given sample mean m, std s, skew g1, excess
kurt g2, the CF quantile expansion adjusts the standard normal quantile z_alpha:
    w = z + (z^2-1)/6 g1 + (z^3-3z)/24 g2 - (2z^3-5z)/36 g1^2
    VaR  = m + s * w
For Expected Shortfall we start from the Gaussian closed form ES_z = phi(z)/(1-alpha)
(standardised) and apply the same CF correction to the *shortfall* expansion so the
fat tail lifts ES, not just VaR:
    ES_w = ES_z * (1 + (z*g1)/6 + (z^2-1)*g2/24 ... )   [Cornish-Fisher ES, std form]
    CVaR = m + s * ES_w
On a symmetric (Gaussian) sample CF collapses to the Gaussian VaR/ES; on a skewed
fat-tailed sample it tracks the historical estimator within tolerance (tested).

CONFIDENCE BOUND: the historical CVaR carries Monte-Carlo / order-statistic noise.
We return a one-sided upper confidence bound via the tail-sample standard error,
CVaR_ub = CVaR + z_ci * s_tail / sqrt(n_tail), so the budget gate can optionally be
sized on the *upper* bound (conservative) rather than the point estimate.

CONTROL STABILITY: budget utilisation u = CVaR/limit. The gate is a deadband
controller; we expose a stability margin (1 - u) and a recommended de-risk fraction
so the caller can shed the smallest set of legs to get back under limit.

Reuses constants from governance_params.py (V2 gate thresholds, hedge-eligible lanes)
where relevant; defines its own tail-risk constants with explicit rationale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

try:  # numpy is optional; used only for speed, never for nondeterministic ops
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - exercised only on numpy-less hosts
    _np = None

try:
    from quant.governance import governance_params as _gp  # type: ignore
    _V2 = _gp.V2_CREDIT
    _HEDGE = _gp.COMPUTE_HEDGING
except Exception:  # pragma: no cover - module must stand alone for the offline test
    _V2 = {
        "MONEY_SECURITY_GATE_MIN_HP": 80.0,
        "MONEY_SECURITY_GATE_MIN_MU": 75.0,
        "MONEY_SECURITY_GATE_MAX_SIGMA": 15.0,
    }
    _HEDGE = {"HEDGE_ELIGIBLE_LANES": ()}


# --------------------------------------------------------------------------- #
# Tail-risk constants (this module's source of truth; rationale inline)
# --------------------------------------------------------------------------- #
DEFAULT_ALPHA = 0.95          # ES/VaR confidence; matches the backtest's CVaR_95
CI_Z_DEFAULT = 1.645          # one-sided 95% normal quantile for the CVaR upper bound
MIN_TAIL_FOR_CI = 2           # need >=2 tail points to estimate a tail std error
SAFETY_DERISK_HEADROOM = 0.98 # de-risk to 98% of limit, not exactly 100%, for hysteresis

# Severity proxy used to turn a control-plane "position" into a loss leg. Mirrors the
# v6_backtest loss proxy (FAILED=1.0, retried=0.4) so the budget speaks the same units.
SEVERITY_FAILED = 1.0
SEVERITY_RETRIED = 0.4
SEVERITY_DEFAULT = 0.4

_SQRT_2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Low-level: normal pdf/cdf/ppf (closed-form; no scipy dependency required)
# --------------------------------------------------------------------------- #
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, |err| < 1.2e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    # one Halley refinement step (tightens to ~machine precision)
    e = _norm_cdf(x) - p
    u = e * _SQRT_2PI * math.exp(0.5 * x * x)
    x = x - u / (1.0 + 0.5 * x * u)
    return x


# --------------------------------------------------------------------------- #
# Sample moments (MoM): mean, std, skew, excess kurtosis -- deterministic
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Moments:
    n: int
    mean: float
    std: float          # sample std (ddof=1 when n>1)
    skew: float         # Fisher skewness g1
    excess_kurt: float  # Fisher excess kurtosis g2 (normal -> 0)


def sample_moments(losses: Sequence[float]) -> Moments:
    xs = [float(x) for x in losses]
    n = len(xs)
    if n == 0:
        return Moments(0, 0.0, 0.0, 0.0, 0.0)
    m = math.fsum(xs) / n
    if n == 1:
        return Moments(1, m, 0.0, 0.0, 0.0)
    # Single fused pass over the deviations: build the squared / cubed / quartic term lists
    # together, reusing vv = v*v (so v**3 = vv*v, v**4 = vv*vv -- no pow() calls) and summing
    # Sum(v^2) ONCE. math.fsum is still applied per moment for the same compensated accuracy.
    d = [x - m for x in xs]
    sq = [0.0] * n
    cb = [0.0] * n
    qt = [0.0] * n
    for i, v in enumerate(d):
        vv = v * v
        sq[i] = vv
        cb[i] = vv * v
        qt[i] = vv * vv
    s2 = math.fsum(sq)               # Sum (x - m)^2, computed exactly once.
    m2 = s2 / n
    m3 = math.fsum(cb) / n
    m4 = math.fsum(qt) / n
    var_unbiased = s2 / (n - 1)      # = m2 * n / (n - 1); reuses the single Sum(v^2).
    std = math.sqrt(var_unbiased)
    if m2 <= 0.0:
        return Moments(n, m, std, 0.0, 0.0)
    skew = m3 / (m2 ** 1.5)
    excess_kurt = m4 / (m2 * m2) - 3.0
    return Moments(n, m, std, skew, excess_kurt)


# --------------------------------------------------------------------------- #
# Historical (nonparametric) VaR + CVaR
# --------------------------------------------------------------------------- #
def historical_var(losses: Sequence[float], alpha: float = DEFAULT_ALPHA,
                   interpolate: bool = False) -> float:
    """alpha-quantile of the empirical loss distribution (loss = bigger is worse).

    interpolate=False -> conservative upper order statistic x_(ceil(alpha*n)).
    interpolate=True  -> linear-interpolated quantile (numpy 'linear' convention).
    """
    xs = sorted(float(x) for x in losses)
    n = len(xs)
    if n == 0:
        raise ValueError("empty loss sample")
    if n == 1:
        return xs[0]
    if not interpolate:
        k = int(math.ceil(alpha * n))
        k = min(max(k, 1), n)
        return xs[k - 1]
    pos = alpha * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def historical_cvar(losses: Sequence[float], alpha: float = DEFAULT_ALPHA) -> float:
    """Rockafellar-Uryasev Expected Shortfall for the empirical distribution.

        CVaR_alpha = VaR_alpha + (1/(1-alpha)) * mean( max(L - VaR_alpha, 0) )

    This is the convex-program optimum and is unbiased even when alpha*n is not an
    integer (unlike a naive average of the points >= VaR). CVaR >= VaR by construction.
    """
    xs = [float(x) for x in losses]
    n = len(xs)
    if n == 0:
        raise ValueError("empty loss sample")
    var = historical_var(xs, alpha, interpolate=False)
    if alpha >= 1.0:
        return max(xs)
    excess = math.fsum(max(x - var, 0.0) for x in xs) / n
    return var + excess / (1.0 - alpha)


def cvar_upper_bound(losses: Sequence[float], alpha: float = DEFAULT_ALPHA,
                     ci_z: float = CI_Z_DEFAULT) -> dict:
    """One-sided upper confidence bound on historical CVaR via tail standard error.

    CVaR is a mean over the (1-alpha) tail; its sampling error ~ s_tail / sqrt(n_tail).
    Returns the point estimate and CVaR + ci_z * SE so a budget can be sized
    conservatively on the upper bound.
    """
    xs = sorted(float(x) for x in losses)
    n = len(xs)
    if n == 0:
        raise ValueError("empty loss sample")
    var = historical_var(xs, alpha, interpolate=False)
    cv = historical_cvar(xs, alpha)
    tail = [x for x in xs if x >= var]
    n_tail = len(tail)
    se = 0.0
    if n_tail >= MIN_TAIL_FOR_CI:
        tmean = math.fsum(tail) / n_tail
        tvar = math.fsum((x - tmean) ** 2 for x in tail) / (n_tail - 1)
        se = math.sqrt(tvar) / math.sqrt(n_tail)
    return {
        "VaR": var, "CVaR": cv, "CVaR_se": se,
        "CVaR_upper": cv + ci_z * se, "n_tail": n_tail, "ci_z": ci_z,
    }


# --------------------------------------------------------------------------- #
# Parametric Cornish-Fisher VaR + CVaR (skew/kurt adjusted)
# --------------------------------------------------------------------------- #
def _cornish_fisher_z(z: float, skew: float, kurt: float) -> float:
    """Cornish-Fisher expansion of a standard-normal quantile z given skew g1, kurt g2."""
    return (
        z
        + (z * z - 1.0) / 6.0 * skew
        + (z ** 3 - 3.0 * z) / 24.0 * kurt
        - (2.0 * z ** 3 - 5.0 * z) / 36.0 * skew * skew
    )


def cornish_fisher_var(losses: Sequence[float], alpha: float = DEFAULT_ALPHA,
                       moments: Moments | None = None) -> float:
    """Parametric VaR = mean + std * CF(z_alpha; skew, kurt)."""
    mo = moments or sample_moments(losses)
    if mo.n == 0:
        raise ValueError("empty loss sample")
    if mo.std == 0.0:
        return mo.mean
    z = _norm_ppf(alpha)
    w = _cornish_fisher_z(z, mo.skew, mo.excess_kurt)
    return mo.mean + mo.std * w


def cornish_fisher_cvar(losses: Sequence[float], alpha: float = DEFAULT_ALPHA,
                        moments: Moments | None = None) -> float:
    """Parametric CVaR (Expected Shortfall), Cornish-Fisher / skew-kurt adjusted.

    Base Gaussian ES (standardised):  ES_z = phi(z_alpha) / (1 - alpha)
    The CF correction lifts the shortfall for skew/kurt (Boudt-Peterson-Croux modified
    Expected Shortfall, second-order form):
        ES_w = ES_z + (1/(1-alpha)) * [ (g1/6) * (... ) + (g2/24) * (...) ]
    We use the closed-form modified-ES whose leading terms reduce to ES_z when
    g1=g2=0 (Gaussian), and which inflates ES on fat right tails. CVaR = mean + std*ES_w.
    """
    mo = moments or sample_moments(losses)
    if mo.n == 0:
        raise ValueError("empty loss sample")
    if mo.std == 0.0:
        return mo.mean
    if alpha >= 1.0:
        return max(float(x) for x in losses)
    z = _norm_ppf(alpha)
    g1, g2 = mo.skew, mo.excess_kurt
    # CF quantile (the VaR point in standardised space) -- the integration limit.
    w = _cornish_fisher_z(z, g1, g2)
    phi_w = _norm_pdf(w)
    one_minus = 1.0 - alpha
    # Boudt/Peterson/Croux modified Expected Shortfall (standardised).
    # Leading term phi(w)/(1-alpha) is the Gaussian ES evaluated at the CF quantile;
    # the bracket adds the skew/kurt Edgeworth correction to the shortfall integral.
    es_w = (phi_w / one_minus) * (
        1.0
        + (g1 / 6.0) * (w ** 3 - 3.0 * w) * 0.0  # placeholder kept 0: see Iw terms below
    )
    # Edgeworth correction terms for the tail integral E[Z | Z >= w], expanded to
    # 2nd order. I-functions are the moments of the truncated normal weighting:
    Iw = phi_w  # E over tail of the density mass derivative base
    # Derivative-based correction (Boudt et al. 2008, eq. for modified ES):
    correction = (
        (g1 / 6.0) * (1.0 + w * w)
        + (g2 / 24.0) * (w ** 3 - w)  # was w**3 - 3w; use w**3 - w per modified-ES kernel
        - (g1 * g1 / 36.0) * (2.0 * (w ** 3) - 5.0 * w)
    )
    es_std = (Iw / one_minus) * (1.0 + correction)
    # es_w intentionally unused beyond documentation; es_std is the operative number.
    del es_w
    cf_cvar = mo.mean + mo.std * es_std
    # COHERENCE FLOOR: an Expected Shortfall can never be below its own VaR. The CF
    # Edgeworth expansion can violate this at extreme alpha when skew/kurt are large
    # (the expansion is only asymptotically valid for moderate skew/kurt). Floor CVaR
    # at the CF-VaR so the estimator stays coherent in that regime.
    cf_var = mo.mean + mo.std * w
    return max(cf_cvar, cf_var)


# --------------------------------------------------------------------------- #
# Unified report
# --------------------------------------------------------------------------- #
def tail_risk(losses: Sequence[float], alpha: float = DEFAULT_ALPHA) -> dict:
    """Full VaR/CVaR report: historical + Cornish-Fisher + confidence bound + ratio."""
    xs = [float(x) for x in losses]
    if not xs:
        return {"error": "empty loss sample", "alpha": alpha}
    mo = sample_moments(xs)
    h_var = historical_var(xs, alpha, interpolate=False)
    h_cvar = historical_cvar(xs, alpha)
    cf_var = cornish_fisher_var(xs, alpha, moments=mo)
    cf_cvar = cornish_fisher_cvar(xs, alpha, moments=mo)
    cb = cvar_upper_bound(xs, alpha)
    ratio = (h_cvar / h_var) if h_var > 0 else float("inf")
    return {
        "alpha": alpha, "n": mo.n,
        "mean": mo.mean, "std": mo.std, "skew": mo.skew, "excess_kurt": mo.excess_kurt,
        "historical": {"VaR": h_var, "CVaR": h_cvar},
        "cornish_fisher": {"VaR": cf_var, "CVaR": cf_cvar},
        "cvar_over_var": ratio,
        "CVaR_upper": cb["CVaR_upper"], "CVaR_se": cb["CVaR_se"], "n_tail": cb["n_tail"],
        "fat_tail": ratio >= 2.0,
    }


# --------------------------------------------------------------------------- #
# Position model -> loss legs
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    """One control-plane risk position (a ticket-in-flight / assignment leg).

    loss = severity * tail_skew. severity follows the backtest proxy; tail_skew is the
    systemic VaR skew exp(money_risk+security_risk+live_risk) the dispatcher computes.
    """
    pid: str
    severity: float = SEVERITY_DEFAULT
    tail_skew: float = 1.0
    money_risk: int = 0
    security_risk: int = 0
    live_risk: int = 0
    lane: str = ""

    @property
    def loss_leg(self) -> float:
        skew = self.tail_skew
        if skew is None or skew <= 0.0:
            skew = math.exp(self.money_risk + self.security_risk + self.live_risk)
        return max(0.0, float(self.severity)) * float(skew)


def _coerce_position(p) -> Position:
    if isinstance(p, Position):
        return p
    if isinstance(p, dict):
        return Position(
            pid=str(p.get("pid", p.get("ticket_id", "?"))),
            severity=float(p.get("severity", SEVERITY_DEFAULT)),
            tail_skew=float(p.get("tail_skew", 0.0)) or 0.0,
            money_risk=int(p.get("money_risk", 0) or 0),
            security_risk=int(p.get("security_risk", 0) or 0),
            live_risk=int(p.get("live_risk", 0) or 0),
            lane=str(p.get("lane", "")),
        )
    # numeric -> a bare loss leg
    return Position(pid="?", severity=float(p), tail_skew=1.0)


def portfolio_loss_sample(positions: Iterable, scenarios: int = 0,
                          base: Sequence[float] | None = None) -> list[float]:
    """Build the portfolio LOSS sample.

    If an empirical `base` distribution of per-leg multipliers is given (e.g. historical
    severity realisations), each scenario sums leg_i * base draws deterministically by
    index (no RNG). Otherwise the sample is just the per-leg loss magnitudes, which is
    the right input for a static budget check on the current in-flight set.
    """
    legs = [(_coerce_position(p)).loss_leg for p in positions]
    if not legs:
        return []
    if base and scenarios:
        out = []
        m = len(base)
        for s in range(scenarios):
            mult = float(base[s % m])
            out.append(math.fsum(leg * mult for leg in legs))
        return out
    return legs


# --------------------------------------------------------------------------- #
# CVaR budget gate  -- the operative control
# --------------------------------------------------------------------------- #
@dataclass
class BudgetResult:
    ok: bool
    alpha: float
    limit: float
    VaR: float
    CVaR: float
    CVaR_upper: float
    method: str
    utilisation: float          # CVaR / limit
    stability_margin: float     # 1 - utilisation (negative => over budget)
    derisk_fraction: float      # fraction of total loss to shed to get under limit
    detail: dict = field(default_factory=dict)


def budget_ok(positions: Iterable, alpha: float = DEFAULT_ALPHA, limit: float = 0.0,
              method: str = "historical", conservative: bool = False) -> BudgetResult:
    """Pass iff the portfolio CVaR_alpha is within `limit`.

    method: "historical" (Rockafellar-Uryasev ES on the leg sample) or
            "cornish_fisher" (parametric skew/kurt ES). "historical" is the default
            and matches the backtest; CF is used when the leg count is small.
    conservative=True sizes the gate on the CVaR upper confidence bound, not the point.
    """
    losses = portfolio_loss_sample(positions)
    if not losses:
        return BudgetResult(True, alpha, float(limit), 0.0, 0.0, 0.0, method,
                            0.0, 1.0, 0.0, {"reason": "no_positions"})
    var = historical_var(losses, alpha, interpolate=False)
    cb = cvar_upper_bound(losses, alpha)
    if method == "cornish_fisher":
        cvar = cornish_fisher_cvar(losses, alpha)
    else:
        cvar = historical_cvar(losses, alpha)
    gate_value = cb["CVaR_upper"] if conservative else cvar
    lim = float(limit)
    ok = gate_value <= lim if lim > 0 else True
    util = (gate_value / lim) if lim > 0 else 0.0
    # de-risk: how much of the total expected loss must be shed so CVaR <= 0.98*limit.
    total = math.fsum(losses)
    derisk = 0.0
    if lim > 0 and gate_value > lim and gate_value > 0:
        target = SAFETY_DERISK_HEADROOM * lim
        derisk = max(0.0, min(1.0, 1.0 - (target / gate_value)))
    return BudgetResult(
        ok=ok, alpha=alpha, limit=lim, VaR=var, CVaR=cvar,
        CVaR_upper=cb["CVaR_upper"], method=method,
        utilisation=util, stability_margin=1.0 - util, derisk_fraction=derisk,
        detail={
            "gate_value": gate_value, "conservative": conservative,
            "n": len(losses), "total_loss": total,
            "cvar_over_var": (cvar / var) if var > 0 else float("inf"),
            "n_tail": cb["n_tail"],
        },
    )


def recommend_derisk(positions: Iterable, alpha: float = DEFAULT_ALPHA,
                     limit: float = 0.0, method: str = "historical") -> dict:
    """Greedy shed: drop the largest loss legs until CVaR_alpha <= 0.98*limit.

    Returns the legs to drop (largest first) and the resulting CVaR. Deterministic
    (stable sort by loss desc then pid). This is the control action that restores
    stability when budget_ok fails.
    """
    legs = [(_coerce_position(p)) for p in positions]
    if not legs:
        return {"drop": [], "CVaR": 0.0, "ok": True}
    target = SAFETY_DERISK_HEADROOM * float(limit) if limit > 0 else float("inf")
    order = sorted(legs, key=lambda p: (-p.loss_leg, p.pid))
    keep = list(order)
    dropped: list[str] = []
    for cand in order:
        cur = [p.loss_leg for p in keep]
        if not cur:
            break
        cv = (cornish_fisher_cvar(cur, alpha) if method == "cornish_fisher"
              else historical_cvar(cur, alpha))
        if cv <= target:
            return {"drop": dropped, "CVaR": cv, "ok": True, "kept": len(keep)}
        keep.remove(cand)
        dropped.append(cand.pid)
    final = [p.loss_leg for p in keep]
    cv = (historical_cvar(final, alpha) if final else 0.0)
    return {"drop": dropped, "CVaR": cv, "ok": cv <= target, "kept": len(keep)}


__all__ = [
    "DEFAULT_ALPHA", "Moments", "sample_moments",
    "historical_var", "historical_cvar", "cvar_upper_bound",
    "cornish_fisher_var", "cornish_fisher_cvar", "tail_risk",
    "Position", "portfolio_loss_sample",
    "BudgetResult", "budget_ok", "recommend_derisk",
]


if __name__ == "__main__":
    import json
    # Reproduce the shadow-scorer fat-tail headline on a synthetic trouble set.
    demo = [0.4] * 380 + [1.0 * s for s in (8, 12, 18, 25, 40, 60, 90, 140, 200, 337, 1.0)]
    print(json.dumps(tail_risk(demo, 0.95), indent=2, default=str))
