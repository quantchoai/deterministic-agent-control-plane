"""Gamma(k, theta) ticket-latency model for the V1-V6 agent-management control plane.

WHY THIS MODULE EXISTS
----------------------
The V5 risk-budgeted auction (dispatcher.auction_pick) prices a ticket's expected
utility as

    U = P(success) * value - model_cost - tail_penalty - context_penalty - disk_penalty

Two of those terms (model_cost and the SLA/tail penalty) need a *distribution* of
how long a ticket will take, not just a point estimate:

  - cost scales with wall-clock occupancy of a worker slot  -> use the MEAN (k*theta);
  - SLA / starvation / tail risk are p95/p99 phenomena       -> use upper QUANTILES.

Ticket latency (queue-age + execution time) is non-negative, right-skewed, and has
a fatter-than-exponential body when an agent is warm and a long tail when it is cold
or thrashing. The Gamma family Gamma(shape=k, scale=theta) is the canonical
positive, right-skewed, two-parameter fit for service/latency times:

    f(x; k, theta) = x^(k-1) * exp(-x/theta) / (Gamma(k) * theta^k),   x > 0

It degenerates to the Exponential(rate=1/theta) at k=1 (memoryless service), is more
peaked / less tailed for k>1 (warm, consistent workers), and more tail-heavy for
k<1. That single knob (k) is exactly the "warm vs cold / consistent vs thrashing"
axis the V4 propulsion layer cares about, which is why Gamma -- not lognormal -- is
the right primitive here.

WHAT THIS MODULE PROVIDES
-------------------------
  - GammaFit.from_mom(samples)   : closed-form Method-of-Moments fit (fast, deterministic,
                                   the warm-start initial guess).
  - GammaFit.from_mle(samples)   : Newton-Raphson MLE on the digamma score equation
                                   (Choi-Wette), warm-started from MoM. The statistically
                                   efficient fit.
  - fit.quantile(p)              : inverse-CDF latency quantile (p50/p95/p99 for SLA/cost).
  - fit.sla_quantiles()          : the p50/p95/p99 triple used by V5.
  - fit.mean() / .var() / .mode(): moments for the V5 cost term.
  - fit.stderr_*  / .ci_*        : Fisher-information confidence bounds on (k, theta).
  - per_domain_fits(...)         : independent fit per dispatcher domain (Risk, Tax, ...).

DETERMINISM
-----------
No randomness anywhere in the estimator path. Given the same samples the fit, the
quantiles, and the confidence bounds are bit-for-bit reproducible. scipy.special is
used for digamma/polygamma and the regularized-incomplete-gamma inverse when present;
a pure-Python fallback (Lanczos log-Gamma + bisection on the incomplete gamma) gives
the same answers to ~1e-9 when scipy is unavailable, so the control plane never
depends on an optional import for its money/SLA math.

This is a MODEL/ESTIMATION module: it consumes observed latency samples and emits
deterministic statistics. It is intentionally side-effect-free (no Mongo, no fleet,
no LLM) so it can run offline in CI and be unit-pinned against seeded data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional scipy acceleration. The module is fully functional without it; the
# pure-Python fallbacks below match scipy to ~1e-9 on the supported domain.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised via both branches in tests where possible
    from scipy.special import digamma as _sp_digamma  # type: ignore
    from scipy.special import polygamma as _sp_polygamma  # type: ignore
    from scipy.special import gammaincinv as _sp_gammaincinv  # type: ignore
    from scipy.special import gammainc as _sp_gammainc  # type: ignore
    from scipy.special import gammaln as _sp_gammaln  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# Pull domain list + the V5 starvation constant from the single source of truth so
# this model speaks the same vocabulary as the dispatcher / governance registry.
try:
    from quant.governance.governance_params import V5_OPTIMIZATION as _V5
    _LAMBDA_STARVATION = float(_V5.get("LAMBDA_STARVATION", 0.018))
except Exception:  # pragma: no cover - keep the model importable standalone
    _LAMBDA_STARVATION = 0.018

# Canonical dispatcher domains (mirrors dispatcher.ALL_DOMAINS). Kept local so the
# model has no hard import dependency on the dispatcher (which pulls in pymongo).
ALL_DOMAINS: Tuple[str, ...] = (
    "Analytics", "Trading", "Execution", "Pricing", "Risk", "Reporting",
    "Accounting", "Tax", "Ops", "Compliance", "Infra", "FrontOffice",
    "Integration", "DesignDocs", "Committee", "Governance", "PM", "Chaos",
)

# Numerical guards. k and theta are strictly positive; we clamp into a sane band so
# a pathological sample (all-equal, or n=1) cannot produce inf/NaN that would poison
# a downstream V5 cost term.
_K_MIN = 1e-3
_K_MAX = 1e6
_THETA_MIN = 1e-12
_NEWTON_MAX_ITER = 100
_NEWTON_TOL = 1e-12


# ===========================================================================
# Special-function layer (digamma / trigamma / log-Gamma / incomplete-gamma inv)
# ===========================================================================
# Lanczos coefficients (g=7, n=9) -- standard, gives log-Gamma to ~1e-13.
_LANCZOS_G = 7.0
_LANCZOS_C = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def _gammaln_py(x: float) -> float:
    """Natural log of |Gamma(x)| via the Lanczos approximation (x > 0 path)."""
    if x < 0.5:
        # Reflection: Gamma(x)Gamma(1-x) = pi / sin(pi x)
        return math.log(abs(math.pi / math.sin(math.pi * x))) - _gammaln_py(1.0 - x)
    x -= 1.0
    a = _LANCZOS_C[0]
    t = x + _LANCZOS_G + 0.5
    for i in range(1, len(_LANCZOS_C)):
        a += _LANCZOS_C[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def gammaln(x: float) -> float:
    if _HAVE_SCIPY:
        return float(_sp_gammaln(x))
    return _gammaln_py(x)


def digamma(x: float) -> float:
    """psi(x) = d/dx ln Gamma(x). Pure-Python: recurrence up + asymptotic series."""
    if _HAVE_SCIPY:
        return float(_sp_digamma(x))
    # Push x up to >= 6 using psi(x) = psi(x+1) - 1/x, then asymptotic expansion.
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    # psi(x) ~ ln x - 1/(2x) - 1/(12 x^2) + 1/(120 x^4) - 1/(252 x^6) + ...
    result += (
        math.log(x)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)))
    )
    return result


def trigamma(x: float) -> float:
    """psi'(x). Pure-Python: recurrence up + asymptotic series."""
    if _HAVE_SCIPY:
        return float(_sp_polygamma(1, x))
    result = 0.0
    while x < 6.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    # psi'(x) ~ 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - ...
    result += inv * (
        1.0 + inv * (0.5 + inv * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0))))
    )
    return result


def _gammainc_py(k: float, x: float) -> float:
    """Regularized lower incomplete gamma P(k, x) = gamma(k, x) / Gamma(k)."""
    if x <= 0.0:
        return 0.0
    if k <= 0.0:
        return 1.0
    if x < k + 1.0:
        # Series expansion (converges fast for x < k+1).
        ap = k
        total = 1.0 / k
        delta = total
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + k * math.log(x) - gammaln(k))
    # Continued fraction (Lentz) for the upper part, then complement.
    tiny = 1e-300
    b = x + 1.0 - k
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - k)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + k * math.log(x) - gammaln(k)) * h
    return 1.0 - q


def gammainc_reg(k: float, x: float) -> float:
    """Regularized lower incomplete gamma P(k, x)."""
    if _HAVE_SCIPY:
        return float(_sp_gammainc(k, x))
    return _gammainc_py(k, x)


def _norm_ppf_wh(p: float) -> float:
    """Inverse standard-normal CDF (Acklam rational approximation, |err| < 1.2e-9).

    Local helper used ONLY to seed the Wilson-Hilferty cube-root start for the Gamma
    quantile Newton iteration. Deterministic; no RNG.
    """
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    a = (-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239)
    b = (-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1)
    c = (-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def _gammaincinv_py(k: float, p: float) -> float:
    """Pure-Python inverse of P(k, .): y s.t. P(k, y) = p (standardized Gamma quantile).

    Newton-Raphson on g(y) = P(k, y) - p, g'(y) = pdf(y) = y^(k-1) e^{-y} / Gamma(k),
    warm-started from the Wilson-Hilferty cube-root normal approximation
        y0 = k * (1 - 1/(9k) + z*sqrt(1/(9k)))^3 ,  z = Phi^{-1}(p),
    which the module header already names but the old code implemented with a 200-iteration
    bracketing bisection. Newton converges quadratically (a handful of steps) to the SAME
    root the bisection found; a monotone bisection fallback guards the rare overshoot so the
    answer is always the bracketed root.
    """
    # Wilson-Hilferty cube-root start (clamped strictly positive).
    z = _norm_ppf_wh(p)
    inv9k = 1.0 / (9.0 * k)
    base = 1.0 - inv9k + z * math.sqrt(inv9k)
    y = k * (base ** 3) if base > 0.0 else max(1e-8, k)
    if not math.isfinite(y) or y <= 0.0:
        y = max(1e-8, k)
    ln_gamma_k = gammaln(k)
    # Maintain a bracket [lo, hi] so a Newton overshoot degrades to bisection, never diverges.
    lo, hi = 0.0, math.inf
    for _ in range(100):
        cdf = gammainc_reg(k, y)
        if cdf < p:
            lo = y
        else:
            hi = y
        # pdf(y) = exp((k-1)ln y - y - lnGamma(k)); guard y>0.
        log_pdf = (k - 1.0) * math.log(y) - y - ln_gamma_k
        pdf = math.exp(log_pdf) if log_pdf > -740.0 else 0.0
        if pdf > 0.0:
            y_new = y - (cdf - p) / pdf
        else:
            y_new = math.nan
        # Newton step must stay inside the bracket and finite; else take the bisection mid.
        if not (math.isfinite(y_new) and lo < y_new < hi):
            y_new = 0.5 * (lo + hi) if math.isfinite(hi) else (y * 2.0 if cdf < p else 0.5 * (lo + y))
        if abs(y_new - y) <= 1e-14 * max(1.0, y):
            y = y_new
            break
        y = y_new
    return y


def gammaincinv_reg(k: float, p: float) -> float:
    """Inverse of P(k, .): return y s.t. P(k, y) = p.  (standardized Gamma quantile)."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"probability must be in [0,1], got {p}")
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return math.inf
    if _HAVE_SCIPY:
        return float(_sp_gammaincinv(k, p))
    return _gammaincinv_py(k, p)


# ===========================================================================
# The fit object
# ===========================================================================
@dataclass(frozen=True)
class GammaFit:
    """An immutable Gamma(k=shape, theta=scale) fit plus its diagnostics.

    All quantities are deterministic functions of the input samples.
    """

    k: float                      # shape (> 0); k == 1 <=> Exponential
    theta: float                  # scale (> 0); mean = k*theta
    n: int                        # number of samples the fit was computed from
    method: str                   # "mom" or "mle"
    loglik: float = field(default=float("nan"))   # log-likelihood at (k, theta)
    # Fisher-information standard errors (filled in by from_mle; NaN for raw MoM).
    stderr_k: float = field(default=float("nan"))
    stderr_theta: float = field(default=float("nan"))

    # ---- moments -----------------------------------------------------------
    def mean(self) -> float:
        return self.k * self.theta

    def var(self) -> float:
        return self.k * self.theta * self.theta

    def std(self) -> float:
        return math.sqrt(self.var())

    def mode(self) -> float:
        # Mode = (k-1)*theta for k>=1, else 0 (distribution diverges at 0).
        return (self.k - 1.0) * self.theta if self.k >= 1.0 else 0.0

    def skewness(self) -> float:
        return 2.0 / math.sqrt(self.k)

    def cv(self) -> float:
        """Coefficient of variation = std/mean = 1/sqrt(k). The 'consistency' knob."""
        return 1.0 / math.sqrt(self.k)

    # ---- distribution functions -------------------------------------------
    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        return gammainc_reg(self.k, x / self.theta)

    def pdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        return math.exp(
            (self.k - 1.0) * math.log(x)
            - x / self.theta
            - gammaln(self.k)
            - self.k * math.log(self.theta)
        )

    def quantile(self, p: float) -> float:
        """Inverse-CDF latency quantile: smallest x with P(X <= x) = p.

        Monotone non-decreasing in p by construction (gammaincinv is monotone and
        theta > 0). p=0 -> 0, p->1 -> +inf.
        """
        return self.theta * gammaincinv_reg(self.k, p)

    # ---- the V5-facing SLA bundle -----------------------------------------
    def sla_quantiles(self) -> Dict[str, float]:
        """p50 (median, for cost), p95 and p99 (for SLA / tail penalty)."""
        return {
            "p50": self.quantile(0.50),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
        }

    def exceedance_prob(self, sla: float) -> float:
        """P(latency > sla): the SLA-breach probability fed to the V5 tail penalty."""
        return 1.0 - self.cdf(sla)

    # ---- confidence bounds -------------------------------------------------
    def ci_k(self, z: float = 1.959963984540054) -> Tuple[float, float]:
        """Normal-approx (Wald) CI for k from the Fisher information. Default 95%."""
        if math.isnan(self.stderr_k):
            return (float("nan"), float("nan"))
        return (max(_K_MIN, self.k - z * self.stderr_k), self.k + z * self.stderr_k)

    def ci_theta(self, z: float = 1.959963984540054) -> Tuple[float, float]:
        if math.isnan(self.stderr_theta):
            return (float("nan"), float("nan"))
        return (max(_THETA_MIN, self.theta - z * self.stderr_theta),
                self.theta + z * self.stderr_theta)

    def is_exponential(self, tol: float = 1e-6) -> bool:
        """True if the shape is indistinguishable from k=1 (memoryless service)."""
        return abs(self.k - 1.0) <= tol

    # ---- constructors ------------------------------------------------------
    @classmethod
    def from_mom(cls, samples: Sequence[float]) -> "GammaFit":
        """Method-of-Moments fit.

        Matching the first two moments of Gamma(k, theta):
            mean  = k*theta            => theta = var/mean
            var   = k*theta^2          => k     = mean^2 / var
        Closed-form, deterministic, O(n). Used both standalone and as the MLE warm
        start. Uses the (population) sample variance with 1/n for an exact moment
        match; n==1 / zero-variance degenerate cases are clamped.
        """
        xs = _validate(samples)
        n = len(xs)
        # Walk the samples ONCE for the two sufficient statistics the fit + log-likelihood
        # both need: sum_x (for the mean) and sum_ln (for _loglik). _loglik is then evaluated
        # from these sums instead of re-walking xs 2-4 more times.
        sum_x = sum(xs)
        sum_ln = sum(math.log(x) for x in xs)
        mean = sum_x / n
        if n == 1:
            # No spread information: assume exponential (k=1), theta=mean.
            th = max(_THETA_MIN, mean)
            return cls(k=1.0, theta=th, n=n, method="mom",
                       loglik=_loglik_from_sums(n, sum_x, sum_ln, 1.0, th))
        var = sum((x - mean) ** 2 for x in xs) / n
        if var <= 0.0:
            # All identical -> infinitely peaked; clamp to a very large k.
            k = _K_MAX
            theta = mean / k
        else:
            k = mean * mean / var
            theta = var / mean
        k = _clamp(k, _K_MIN, _K_MAX)
        theta = max(_THETA_MIN, theta)
        return cls(k=k, theta=theta, n=n, method="mom",
                   loglik=_loglik_from_sums(n, sum_x, sum_ln, k, theta))

    @classmethod
    def from_mle(
        cls,
        samples: Sequence[float],
        max_iter: int = _NEWTON_MAX_ITER,
        tol: float = _NEWTON_TOL,
    ) -> "GammaFit":
        """Maximum-likelihood fit via Newton-Raphson on the digamma score equation.

        For Gamma(k, theta) the profiled MLE reduces to a 1-D root find in k. With
            s = ln(xbar) - mean(ln x)          (>= 0 by Jensen)
        the score equation is
            g(k) = ln(k) - psi(k) - s = 0
        and theta_hat = xbar / k. We Newton-iterate
            k <- k - g(k) / g'(k),   g'(k) = 1/k - psi'(k)
        warm-started from the closed-form Minka/Choi-Wette approximation
            k0 ~ (3 - s + sqrt((s-3)^2 + 24 s)) / (12 s).
        Quadratic convergence; typically 4-6 iterations to 1e-12.

        Standard errors come from the inverse observed/expected Fisher information:
            I(k, theta) = n * [[ psi'(k),      1/theta      ],
                               [ 1/theta,      k/theta^2    ]]
            det = n^2 * (psi'(k)*k - 1) / theta^2
            Var(k)     = (k/theta^2) / det_per_n ... (see _fisher_se).
        """
        xs = _validate(samples)
        n = len(xs)
        # Single walk for both Gamma sufficient statistics (sum_x, sum_ln x); every
        # subsequent _loglik / score-equation term is derived from these sums, never by
        # re-walking xs (previously sum(ln x) was recomputed up to 2-4 times).
        sum_x = sum(xs)
        sum_ln = sum(math.log(x) for x in xs)
        xbar = sum_x / n
        if n == 1:
            th = max(_THETA_MIN, xbar)
            return cls(k=1.0, theta=th, n=n, method="mle",
                       loglik=_loglik_from_sums(n, sum_x, sum_ln, 1.0, th))

        mean_ln = sum_ln / n
        s = math.log(xbar) - mean_ln  # >= 0; ==0 only if all samples equal

        if s <= 1e-15:
            # Degenerate: zero log-variance -> effectively infinitely peaked.
            k = _K_MAX
            theta = max(_THETA_MIN, xbar / k)
            return cls(k=k, theta=theta, n=n, method="mle",
                       loglik=_loglik_from_sums(n, sum_x, sum_ln, k, theta))

        # Choi-Wette / Minka closed-form warm start.
        k = (3.0 - s + math.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
        k = _clamp(k, _K_MIN, _K_MAX)

        for _ in range(max_iter):
            g = math.log(k) - digamma(k) - s
            gp = 1.0 / k - trigamma(k)  # g'(k); note gp < 0 for all k>0
            if gp == 0.0:
                break
            step = g / gp
            k_new = k - step
            # Guard the iterate positive (Newton can overshoot below 0 for tiny k).
            if k_new <= 0.0 or not math.isfinite(k_new):
                k_new = k / 2.0
            if abs(k_new - k) <= tol * max(1.0, k):
                k = k_new
                break
            k = k_new

        k = _clamp(k, _K_MIN, _K_MAX)
        theta = max(_THETA_MIN, xbar / k)
        se_k, se_theta = _fisher_se(n, k, theta)
        return cls(
            k=k, theta=theta, n=n, method="mle",
            loglik=_loglik_from_sums(n, sum_x, sum_ln, k, theta),
            stderr_k=se_k, stderr_theta=se_theta,
        )


# ===========================================================================
# Helpers
# ===========================================================================
def _validate(samples: Sequence[float]) -> List[float]:
    xs = [float(x) for x in samples]
    if len(xs) == 0:
        raise ValueError("Gamma fit needs at least one sample")
    for x in xs:
        if not math.isfinite(x) or x <= 0.0:
            raise ValueError(
                f"Gamma latency samples must be finite and strictly positive; got {x}"
            )
    return xs


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _loglik_from_sums(n: int, sum_x: float, sum_ln: float, k: float, theta: float) -> float:
    """Total log-likelihood from the Gamma sufficient statistics (n, sum_x, sum_ln x).

    sum_i ln f(x_i; k, theta) depends on the data ONLY through (sum_x, sum_ln x), so a
    caller that already walked the samples once for those sums need not re-walk them.
    """
    return (
        (k - 1.0) * sum_ln
        - sum_x / theta
        - n * gammaln(k)
        - n * k * math.log(theta)
    )


def _loglik(xs: Sequence[float], k: float, theta: float) -> float:
    """Total log-likelihood sum_i ln f(x_i; k, theta) (re-walks xs for the two sums)."""
    return _loglik_from_sums(len(xs), sum(xs), sum(math.log(x) for x in xs), k, theta)


def _fisher_se(n: int, k: float, theta: float) -> Tuple[float, float]:
    """Asymptotic standard errors of (k_hat, theta_hat) from the Fisher information.

    The expected Fisher information for one observation of Gamma(k, theta) is
        I1 = [[ psi'(k),    1/theta    ],
              [ 1/theta,    k/theta^2  ]]
    The full-sample information is n*I1; the covariance is (n*I1)^{-1}.
        det(I1) = psi'(k)*k/theta^2 - 1/theta^2 = (k*psi'(k) - 1)/theta^2
        Var(k)     = (k/theta^2)  / (n*det)
        Var(theta) = (psi'(k))    / (n*det)
    """
    tg = trigamma(k)
    det = (k * tg - 1.0) / (theta * theta)
    if det <= 0.0 or not math.isfinite(det):
        return (float("nan"), float("nan"))
    var_k = (k / (theta * theta)) / (n * det)
    var_theta = tg / (n * det)
    if var_k < 0.0 or var_theta < 0.0:
        return (float("nan"), float("nan"))
    return (math.sqrt(var_k), math.sqrt(var_theta))


def fit_latency(samples: Sequence[float], method: str = "mle") -> GammaFit:
    """Convenience entry point. method in {"mle", "mom"}."""
    m = method.lower()
    if m == "mom":
        return GammaFit.from_mom(samples)
    if m == "mle":
        return GammaFit.from_mle(samples)
    raise ValueError(f"unknown method {method!r}; use 'mle' or 'mom'")


def per_domain_fits(
    samples_by_domain: Dict[str, Sequence[float]],
    method: str = "mle",
    min_samples: int = 2,
) -> Dict[str, GammaFit]:
    """Fit an independent Gamma latency model per dispatcher domain.

    Domains with fewer than ``min_samples`` observations are skipped (an
    under-observed domain must not contribute a spurious tail estimate to the V5
    cost term). Deterministic: iterates domains in canonical ALL_DOMAINS order
    first, then any extra keys sorted, so the returned dict order is stable.
    """
    out: Dict[str, GammaFit] = {}
    ordered = [d for d in ALL_DOMAINS if d in samples_by_domain]
    extras = sorted(k for k in samples_by_domain if k not in ALL_DOMAINS)
    for dom in ordered + extras:
        xs = list(samples_by_domain[dom])
        if len(xs) < min_samples:
            continue
        out[dom] = fit_latency(xs, method=method)
    return out


def starvation_age_for_quantile(fit: GammaFit, p: float = 0.95) -> float:
    """Latency threshold at which the V5 starvation lift exp(LAMBDA*age) is priced.

    Couples the latency model to the V5 starvation constant: returns the p-quantile
    latency (the age past which a ticket is in the tail and should receive priority
    lift). Pure read of the registry LAMBDA is exposed for callers that want the
    multiplier directly.
    """
    q = fit.quantile(p)
    return q


def starvation_multiplier(age: float) -> float:
    """exp(LAMBDA_STARVATION * age) -- the V5 queue-age lift, sourced from the registry."""
    return math.exp(_LAMBDA_STARVATION * age)


__all__ = [
    "GammaFit",
    "fit_latency",
    "per_domain_fits",
    "starvation_age_for_quantile",
    "starvation_multiplier",
    "gammaln",
    "digamma",
    "trigamma",
    "gammainc_reg",
    "gammaincinv_reg",
    "ALL_DOMAINS",
]
