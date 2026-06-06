"""modelver_poisson_failure.py -- Poisson failure-arrival model (estimation + inference).

PRIVATE / SHADOW + OFFLINE only. This is the V6 *estimation* layer that PRODUCES the
rate lambda consumed by the closed-form predictor `governance_v6.poisson_failure(lmbda, k)`.

Where it sits in the V-stack
----------------------------
    V3 hazard         : Poisson lambda is a *prior* on context-failure arrival
                        (a calibrated per-agent/domain base rate, replacing the flat
                        BASE_HAZARD guess in ledger.calculate_hazard).
    V6 gate           : the calibration error of the fitted Poisson (predicted P(>=k)
                        vs realized) is exactly the section-2.2 promotion metric
                        ("calibration error <= 10%"). This module computes that error,
                        the goodness-of-fit chi-square, and the confidence interval on
                        lambda so the promotion review can decide shadow -> canary.

Doctrine honoured (V6 mandate sections 1, 2, 4):
    * Deterministic, pure local math. NO LLM, NO Mongo, NO network. Heavy math lives in
      the slow loop; the hot path only reads the resulting lambda.
    * Failure / hallucination arrivals are DISCRETE COUNT events -> Poisson is the correct
      family (roadmap 1A). We do MLE of lambda from historical per-window failure counts,
      give P(>=1 failure in next k windows) = 1 - e^{-lambda*k}, a chi-square dispersion /
      goodness-of-fit test (a Poisson must have variance == mean; over-dispersion => the
      data is NOT Poisson and the model must NOT be promoted), and Wald + exact (Garwood,
      chi-square) confidence intervals on lambda.
    * Constants (failure budget, calibration-error ceiling, min sample) are sourced from
      governance_params.py so there is one source of truth.

scipy is used when importable (exact chi-square / gamma quantiles); otherwise deterministic
closed-form fallbacks (Wilson-Hilferty for chi-square quantiles, series for the Poisson
CDF) keep the module standard-library-only. Results are bit-identical-stable: no RNG.

Reference targets baked into the test (SHADOW_SCORER_FINDINGS_20260603 / V6 mandate 2.2):
    * recover a known lambda from seeded counts within the MLE standard error;
    * chi-square GoF REJECTS an over-dispersed (non-Poisson) sample;
    * calibration-error gate <= 10% per the V6 promotion threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# Single source of truth for governance constants.
from quant.governance.governance_params import V3_HAZARD, V5_OPTIMIZATION

# Optional acceleration. Everything has a deterministic closed-form fallback.
try:  # pragma: no cover - exercised by whichever branch the host provides
    from scipy import stats as _scipy_stats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Module constants (derived from governance_params where one exists)
# --------------------------------------------------------------------------- #
# Minimum windows before the fit may feed a *gate* (section 2.1 #1 / 2.2). Below this we
# still return an estimate but flag it shadow-only / not promotable.
MIN_WINDOWS_FOR_GATE = int(V3_HAZARD.get("CALIBRATION_MIN_EVENTS", 500))
# V6 section 2.2: Poisson promotion requires calibration error <= 10%.
CALIBRATION_ERROR_CEILING = 0.10
# Lane "low-confidence" probability budget reused as the default per-window failure budget
# for the gate helper (a ticket window whose P(>=1 failure) exceeds this is held).
DEFAULT_FAILURE_BUDGET = 1.0 - float(V5_OPTIMIZATION.get("LOW_CONFIDENCE_MIN_P", 0.72))
# Numerical floor so log/divide never blow up on an all-zero history.
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Closed-form numerics (used when scipy is absent)
# --------------------------------------------------------------------------- #
def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam). Deterministic, ~1e-9 abs error."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
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


def _lower_gamma_regularized(s: float, x: float) -> float:
    """Regularized lower incomplete gamma P(s, x) = gamma(s, x) / Gamma(s).

    Deterministic series (x < s+1) / continued fraction (else), Numerical-Recipes style.
    This is the engine behind both the chi-square CDF and the Poisson CDF fallbacks.
    """
    if x <= 0.0:
        return 0.0
    if s <= 0.0:
        return 1.0
    gln = math.lgamma(s)
    if x < s + 1.0:
        # series representation
        ap = s
        total = 1.0 / s
        delta = total
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + s * math.log(x) - gln)
    # continued fraction for the upper incomplete gamma Q, then 1 - Q
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
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
    q = math.exp(-x + s * math.log(x) - gln) * h
    return 1.0 - q


def _chi2_cdf(x: float, df: float) -> float:
    """CDF of a chi-square with `df` degrees of freedom."""
    if x <= 0.0:
        return 0.0
    if _HAVE_SCIPY:
        return float(_scipy_stats.chi2.cdf(x, df))
    return _lower_gamma_regularized(df / 2.0, x / 2.0)


def _chi2_sf(x: float, df: float) -> float:
    """Survival function (upper-tail p-value) of a chi-square."""
    return max(0.0, min(1.0, 1.0 - _chi2_cdf(x, df)))


def _chi2_ppf(p: float, df: float) -> float:
    """Quantile (inverse CDF) of a chi-square with `df` dof.

    scipy when present; otherwise the Wilson-Hilferty cube-root normal approximation
    refined by one Newton step against the closed-form CDF (deterministic, no RNG).
    """
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    if _HAVE_SCIPY:
        return float(_scipy_stats.chi2.ppf(p, df))
    # Wilson-Hilferty initial guess
    z = _norm_ppf(p)
    t = 1.0 - 2.0 / (9.0 * df)
    x = df * (t + z * math.sqrt(2.0 / (9.0 * df))) ** 3
    x = max(x, 1e-9)
    # one Newton refinement: f(x)=CDF(x)-p, f'(x)=pdf(x)
    for _ in range(60):
        cdf = _chi2_cdf(x, df)
        # chi-square pdf
        log_pdf = (df / 2.0 - 1.0) * math.log(x) - x / 2.0 - (df / 2.0) * math.log(2.0) - math.lgamma(df / 2.0)
        pdf = math.exp(log_pdf)
        if pdf < 1e-300:
            break
        step = (cdf - p) / pdf
        x_new = x - step
        if x_new <= 0.0:
            x_new = x / 2.0
        if abs(x_new - x) < 1e-10 * (1.0 + x):
            x = x_new
            break
        x = x_new
    return x


def _poisson_pmf(lmbda: float, k: int) -> float:
    if lmbda < 0.0 or k < 0:
        return 0.0
    return math.exp(-lmbda + k * math.log(lmbda + _EPS) - math.lgamma(k + 1))


def _poisson_cdf(lmbda: float, k: int) -> float:
    """P(X <= k) for X ~ Poisson(lmbda). Uses the gamma identity (exact, deterministic)."""
    if k < 0:
        return 0.0
    if lmbda <= 0.0:
        return 1.0
    # P(X <= k) = Q(k+1, lambda) = 1 - P(k+1, lambda) (regularized lower incomplete gamma)
    return max(0.0, min(1.0, 1.0 - _lower_gamma_regularized(k + 1.0, lmbda)))


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PoissonFit:
    """Immutable fitted Poisson model + inference for one agent/domain failure stream."""
    lambda_hat: float            # MLE of the rate (failures per window)
    n_windows: int               # number of observed windows
    total_failures: int          # sum of counts
    std_error: float             # SE(lambda_hat) = sqrt(lambda_hat / n)
    ci_low: float                # lower confidence bound on lambda
    ci_high: float               # upper confidence bound on lambda
    ci_method: str               # "exact" (Garwood) or "wald"
    confidence: float            # CI coverage (e.g. 0.95)
    dispersion: float            # variance/mean ratio (Fisher index); 1.0 == ideal Poisson
    gof_stat: float              # chi-square dispersion statistic
    gof_dof: int                 # degrees of freedom for the GoF test
    gof_pvalue: float            # p-value; small => reject Poisson
    is_poisson: bool             # GoF not rejected at `alpha`
    gate_ready: bool             # n_windows >= MIN_WINDOWS_FOR_GATE
    extras: dict = field(default_factory=dict)

    def prob_at_least_one(self, k: int = 1) -> float:
        """P(>=1 failure across the next k windows) = 1 - e^{-lambda*k}."""
        return prob_at_least_one(self.lambda_hat, k)

    def prob_at_least(self, k: int, windows: int = 1) -> float:
        """P(>= k failures over `windows` windows) under Poisson(lambda*windows)."""
        return prob_at_least(self.lambda_hat, k, windows)


# --------------------------------------------------------------------------- #
# Core estimation
# --------------------------------------------------------------------------- #
def mle_lambda(counts: Sequence[int]) -> float:
    """MLE of the Poisson rate: lambda_hat = sum(counts) / n_windows = sample mean.

    For X_i ~ iid Poisson(lambda), the log-likelihood is
        l(lambda) = -n*lambda + (sum x_i) log lambda - sum log(x_i!),
    whose stationary point is the sample mean -- the MLE (also the MoM estimate; for the
    Poisson the two coincide because mean == variance == lambda).
    """
    n = len(counts)
    if n == 0:
        return 0.0
    return sum(int(c) for c in counts) / n


def ewma_lambda(counts: Sequence[int], halflife: float = 10.0) -> float:
    """Exponentially-weighted rate for the V3 hazard prior (recent windows weighted up).

    halflife in windows; decay alpha = 1 - 2^{-1/halflife}. Newest sample weighted most.
    This is the `lambda_hat = EWMA of recent failure rate` the roadmap (1A) specifies for
    feeding the live hazard prior, distinct from the unweighted MLE used for calibration.
    """
    if not counts:
        return 0.0
    alpha = 1.0 - 2.0 ** (-1.0 / max(_EPS, halflife))
    ewma = float(counts[0])
    for c in counts[1:]:
        ewma = alpha * float(c) + (1.0 - alpha) * ewma
    return ewma


def prob_at_least_one(lmbda: float, k: int = 1) -> float:
    """P(>=1 failure in the next k windows) = 1 - e^{-lambda*k}. The headline closed form."""
    if k <= 0 or lmbda <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-lmbda * k)))


def prob_at_least(lmbda: float, k: int, windows: int = 1) -> float:
    """P(>= k failures over `windows` windows): 1 - CDF_{Poisson(lambda*windows)}(k-1)."""
    if k <= 0:
        return 1.0
    rate = max(0.0, lmbda) * max(0, windows)
    if rate <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - _poisson_cdf(rate, k - 1)))


# --------------------------------------------------------------------------- #
# Confidence intervals on lambda
# --------------------------------------------------------------------------- #
def lambda_ci_wald(lmbda: float, n_windows: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wald (normal-approximation) CI: lambda_hat +/- z * sqrt(lambda_hat / n).

    SE(lambda_hat) = sqrt(Var(mean)) = sqrt(lambda/n) because Var(Poisson)=lambda.
    Cheap and symmetric; valid for large n*lambda. Use the exact CI for small samples.
    """
    if n_windows <= 0:
        return (0.0, 0.0)
    z = _norm_ppf(0.5 + confidence / 2.0)
    se = math.sqrt(max(0.0, lmbda) / n_windows)
    return (max(0.0, lmbda - z * se), lmbda + z * se)


def lambda_ci_exact(total_failures: int, n_windows: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) CI for a Poisson rate via the chi-square relationship.

    With S = sum of counts ~ Poisson(n*lambda):
        lower = chi2_ppf(alpha/2,   2S)     / (2n)
        upper = chi2_ppf(1-alpha/2, 2S+2)   / (2n)
    (lower bound is 0 when S == 0). Exact coverage for any sample size -- the correct CI
    for the rare-event / small-count regime the V6 mandate (2.3) flags for Poisson.
    """
    if n_windows <= 0:
        return (0.0, 0.0)
    alpha = 1.0 - confidence
    s = max(0, int(total_failures))
    if s == 0:
        low = 0.0
    else:
        low = _chi2_ppf(alpha / 2.0, 2 * s) / (2.0 * n_windows)
    high = _chi2_ppf(1.0 - alpha / 2.0, 2 * s + 2) / (2.0 * n_windows)
    return (max(0.0, low), high)


# --------------------------------------------------------------------------- #
# Goodness of fit -- dispersion test (variance/mean ratio)
# --------------------------------------------------------------------------- #
def dispersion_gof(counts: Sequence[int], alpha: float = 0.05,
                   xbar: float | None = None) -> dict:
    """Chi-square dispersion (variance) goodness-of-fit test for the Poisson assumption.

    A Poisson distribution has variance == mean. The dispersion statistic
        T = sum_i (x_i - xbar)^2 / xbar   ~  chi-square(n - 1)  under H0: data is Poisson.
    The Fisher index of dispersion (var/mean) should be ~1. Over-dispersion (var >> mean,
    bursty/clustered failures) inflates T and yields a small p-value => REJECT Poisson;
    the model must NOT then be promoted (it would understate clustered-failure risk).

    Returns a TWO-SIDED test (rejects both over- and under-dispersion) plus the components.

    `xbar` may be supplied by a caller that has already computed the MLE (the sample mean)
    so the duplicate `mle_lambda(counts)` pass is skipped -- the GoF is the SAME closed form
    of the sufficient statistic either way (the sample mean IS the MLE).
    """
    n = len(counts)
    if n < 2:
        return {"error": "need >= 2 windows", "is_poisson": False}
    if xbar is None:
        xbar = mle_lambda(counts)
    if xbar <= 0.0:
        # all zeros: degenerate but consistent with Poisson(0); cannot reject.
        return {
            "stat": 0.0, "dof": n - 1, "pvalue": 1.0, "dispersion": 0.0,
            "is_poisson": True, "mean": 0.0, "variance": 0.0,
            "note": "all-zero history; consistent with Poisson(lambda->0)",
        }
    ss = sum((int(c) - xbar) ** 2 for c in counts)
    stat = ss / xbar
    dof = n - 1
    var = ss / n  # population variance
    dispersion = var / xbar
    # two-sided p-value from the chi-square tails
    lower_tail = _chi2_cdf(stat, dof)
    p_two_sided = 2.0 * min(lower_tail, 1.0 - lower_tail)
    p_two_sided = max(0.0, min(1.0, p_two_sided))
    return {
        "stat": stat,
        "dof": dof,
        "pvalue": p_two_sided,
        "dispersion": dispersion,
        "mean": xbar,
        "variance": var,
        "is_poisson": p_two_sided >= alpha,
        "over_dispersed": dispersion > 1.0 and p_two_sided < alpha,
        "under_dispersed": dispersion < 1.0 and p_two_sided < alpha,
    }


def pearson_gof(counts: Sequence[int], alpha: float = 0.05, min_expected: float = 5.0) -> dict:
    """Pearson chi-square binned goodness-of-fit against the FITTED Poisson pmf.

    Bins the observed counts, computes expected = n * pmf(k; lambda_hat), pools the right
    tail so every cell has expected >= `min_expected`, then
        X^2 = sum (O - E)^2 / E   ~  chi-square(bins - 1 - 1)
    (one extra dof lost for estimating lambda). Complementary to the dispersion test:
    this one catches shape mis-fit (e.g. zero-inflation) that variance alone can miss.
    """
    n = len(counts)
    if n < 5:
        return {"error": "need >= 5 windows for a binned chi-square", "is_poisson": True}
    lam = mle_lambda(counts)
    if lam <= 0.0:
        return {"stat": 0.0, "dof": 0, "pvalue": 1.0, "is_poisson": True,
                "note": "degenerate lambda=0"}
    kmax = max(int(c) for c in counts)
    observed: dict[int, int] = {}
    for c in counts:
        observed[int(c)] = observed.get(int(c), 0) + 1
    # build cells 0..kmax with a pooled tail
    cells_o: list[float] = []
    cells_e: list[float] = []
    acc_o = 0.0
    acc_e = 0.0
    cdf_below = 0.0
    for k in range(0, kmax + 1):
        e = n * _poisson_pmf(lam, k)
        o = float(observed.get(k, 0))
        cdf_below += _poisson_pmf(lam, k)
        acc_o += o
        acc_e += e
        if acc_e >= min_expected:
            cells_o.append(acc_o)
            cells_e.append(acc_e)
            acc_o = 0.0
            acc_e = 0.0
    # add the upper tail mass P(X > kmax) into the last cell
    tail_e = n * max(0.0, 1.0 - cdf_below)
    if cells_e:
        cells_o[-1] += acc_o
        cells_e[-1] += acc_e + tail_e
    else:
        cells_o.append(acc_o)
        cells_e.append(acc_e + tail_e)
    # merge a too-small final cell back into its predecessor
    while len(cells_e) >= 2 and cells_e[-1] < min_expected:
        cells_e[-2] += cells_e[-1]
        cells_o[-2] += cells_o[-1]
        cells_e.pop()
        cells_o.pop()
    bins = len(cells_e)
    dof = bins - 1 - 1  # -1 normalization, -1 for estimating lambda
    if dof < 1:
        return {"stat": 0.0, "dof": dof, "pvalue": 1.0, "is_poisson": True,
                "bins": bins, "note": "too few bins after pooling for a powered test"}
    stat = sum((o - e) ** 2 / e for o, e in zip(cells_o, cells_e) if e > 0)
    pvalue = _chi2_sf(stat, dof)
    return {
        "stat": stat, "dof": dof, "pvalue": pvalue, "bins": bins,
        "is_poisson": pvalue >= alpha,
        "cells_observed": cells_o, "cells_expected": cells_e,
    }


# --------------------------------------------------------------------------- #
# Calibration error -- the V6 section 2.2 promotion metric for Poisson
# --------------------------------------------------------------------------- #
def calibration_error(train_counts: Sequence[int], test_counts: Sequence[int], k: int = 1) -> dict:
    """Predicted P(>=k failure per window) on train vs realized frequency on test.

    Fits lambda on `train_counts`, predicts p_hat = P(>=k in 1 window), and compares to the
    realized fraction of test windows with >= k failures. The absolute gap is the V6
    "calibration error" that must be <= 10% (CALIBRATION_ERROR_CEILING) to promote.
    """
    if not test_counts:
        return {"error": "no test windows"}
    lam = mle_lambda(train_counts)
    predicted = prob_at_least(lam, k, windows=1)
    realized = sum(1 for c in test_counts if int(c) >= k) / len(test_counts)
    err = abs(predicted - realized)
    return {
        "lambda_hat": lam,
        "k": k,
        "predicted": predicted,
        "realized": realized,
        "abs_error": err,
        "within_gate": err <= CALIBRATION_ERROR_CEILING,
        "ceiling": CALIBRATION_ERROR_CEILING,
    }


# --------------------------------------------------------------------------- #
# Top-level fit
# --------------------------------------------------------------------------- #
def fit(counts: Sequence[int], confidence: float = 0.95, alpha: float = 0.05,
        ci_method: str = "exact") -> PoissonFit:
    """Fit Poisson(lambda) to a per-window failure-count history and run full inference.

    counts: integer failures observed in each historical window (e.g. per 1000 tickets,
            or per minute) for one agent/domain stream.
    Returns a PoissonFit with the MLE, CI, dispersion goodness-of-fit, and gate readiness.
    """
    clean = [max(0, int(c)) for c in counts]
    n = len(clean)
    total = sum(clean)
    lam = mle_lambda(clean)
    se = math.sqrt(lam / n) if n > 0 else 0.0

    if ci_method == "wald":
        ci_low, ci_high = lambda_ci_wald(lam, n, confidence)
        used = "wald"
    else:
        ci_low, ci_high = lambda_ci_exact(total, n, confidence)
        used = "exact"

    # Pass the already-computed MLE (sample mean) in so dispersion_gof does not re-walk the
    # counts to recompute it -- same number, one fewer pass.
    gof = dispersion_gof(clean, alpha=alpha, xbar=lam)
    return PoissonFit(
        lambda_hat=lam,
        n_windows=n,
        total_failures=total,
        std_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method=used,
        confidence=confidence,
        dispersion=float(gof.get("dispersion", 0.0)),
        gof_stat=float(gof.get("stat", 0.0)),
        gof_dof=int(gof.get("dof", 0)),
        gof_pvalue=float(gof.get("pvalue", 1.0)),
        is_poisson=bool(gof.get("is_poisson", True)),
        gate_ready=n >= MIN_WINDOWS_FOR_GATE,
        extras={"ewma_lambda": ewma_lambda(clean), "scipy": _HAVE_SCIPY},
    )


def _suff_stats(counts: Sequence[int]) -> tuple[int, int, int]:
    """Single-pass Poisson sufficient statistics: (n, S1=sum x, S2=sum x^2).

    For X_i ~ iid Poisson(lambda) the pair (n, S1) is a sufficient statistic for lambda
    (S1 alone, given n); S2 is added so the dispersion / GoF -- which needs sum((x-xbar)^2)
    = S2 - S1^2/n -- is derivable from the SAME pass with no second walk over the data.
    """
    n = 0
    s1 = 0
    s2 = 0
    for c in counts:
        c = max(0, int(c))
        n += 1
        s1 += c
        s2 += c * c
    return n, s1, s2


def _dispersion_gof_from_suff(n: int, s1: int, s2: int, alpha: float = 0.05) -> dict:
    """Dispersion GoF computed from the sufficient stats (n, S1, S2) -- no data pass.

    sum_i (x_i - xbar)^2 = S2 - S1^2/n exactly (the standard sum-of-squares identity),
    so the statistic T = SS / xbar, the dispersion var/mean, and the two-sided chi-square
    p-value are all the SAME closed forms `dispersion_gof` returns, with zero extra passes.
    """
    if n < 2:
        return {"error": "need >= 2 windows", "is_poisson": False}
    xbar = s1 / n
    if xbar <= 0.0:
        return {
            "stat": 0.0, "dof": n - 1, "pvalue": 1.0, "dispersion": 0.0,
            "is_poisson": True, "mean": 0.0, "variance": 0.0,
            "note": "all-zero history; consistent with Poisson(lambda->0)",
        }
    # SS = sum (x - xbar)^2 = S2 - S1^2/n  (algebraically identical to the direct sum).
    ss = s2 - (s1 * s1) / n
    if ss < 0.0:  # guard floating round-off on a degenerate all-equal sample
        ss = 0.0
    stat = ss / xbar
    dof = n - 1
    var = ss / n
    dispersion = var / xbar
    lower_tail = _chi2_cdf(stat, dof)
    p_two_sided = 2.0 * min(lower_tail, 1.0 - lower_tail)
    p_two_sided = max(0.0, min(1.0, p_two_sided))
    return {
        "stat": stat,
        "dof": dof,
        "pvalue": p_two_sided,
        "dispersion": dispersion,
        "mean": xbar,
        "variance": var,
        "is_poisson": p_two_sided >= alpha,
        "over_dispersed": dispersion > 1.0 and p_two_sided < alpha,
        "under_dispersed": dispersion < 1.0 and p_two_sided < alpha,
    }


def fit_fast(counts: Sequence[int], confidence: float = 0.95, alpha: float = 0.05) -> PoissonFit:
    """Streaming Poisson fit: lambda_hat, dispersion, and GoF from a SINGLE pass.

    All of the consumed quantities -- the MLE lambda_hat = S1/n, the standard error
    sqrt(lambda_hat/n), the dispersion index, and the chi-square GoF p-value -- are closed
    forms of the sufficient statistics (n, S1, S2), so one pass over the counts suffices.

    This is byte-for-byte the SAME math as `fit(..., ci_method="wald")` on every field the
    materializer consumes (lambda_hat, dispersion, gof_*, is_poisson, n_windows,
    total_failures, std_error). It deliberately uses the WALD confidence interval (a cheap
    closed form of (lambda_hat, n)) instead of the exact Garwood interval: the exact CI
    needs two chi-square-quantile Newton solves and the materializer never reads ci_low/
    ci_high, so computing it would be pure waste. Callers that need the exact small-count CI
    must use `fit(..., ci_method="exact")`.
    """
    n, s1, s2 = _suff_stats(counts)
    lam = s1 / n if n > 0 else 0.0
    se = math.sqrt(lam / n) if n > 0 else 0.0
    ci_low, ci_high = lambda_ci_wald(lam, n, confidence)
    gof = _dispersion_gof_from_suff(n, s1, s2, alpha=alpha)
    # EWMA needs the ordered stream (it is not a function of the sufficient stats); one pass.
    return PoissonFit(
        lambda_hat=lam,
        n_windows=n,
        total_failures=s1,
        std_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_method="wald",
        confidence=confidence,
        dispersion=float(gof.get("dispersion", 0.0)),
        gof_stat=float(gof.get("stat", 0.0)),
        gof_dof=int(gof.get("dof", 0)),
        gof_pvalue=float(gof.get("pvalue", 1.0)),
        is_poisson=bool(gof.get("is_poisson", True)),
        gate_ready=n >= MIN_WINDOWS_FOR_GATE,
        extras={"ewma_lambda": ewma_lambda([max(0, int(c)) for c in counts]),
                "scipy": _HAVE_SCIPY, "fast": True,
                "over_dispersed": bool(gof.get("over_dispersed", False)),
                "under_dispersed": bool(gof.get("under_dispersed", False))},
    )


# --------------------------------------------------------------------------- #
# V6 gate / V3 hazard-prior feed
# --------------------------------------------------------------------------- #
def hazard_prior(counts: Sequence[int], halflife: float = 10.0) -> float:
    """Per-window failure probability for the V3 hazard prior: 1 - e^{-lambda_ewma}.

    Replaces the flat BASE_HAZARD guess with a calibrated, agent/domain-specific base rate.
    Uses the EWMA rate (recent windows weighted up) so the prior tracks regime shifts.
    """
    lam = ewma_lambda(counts, halflife=halflife)
    return prob_at_least_one(lam, 1)


def gate_decision(counts: Sequence[int], k: int = 1,
                  failure_budget: float | None = None,
                  confidence: float = 0.95) -> dict:
    """V6-style hold/allow decision for the NEXT k windows of work.

    Deterministic: refuse/hold the assignment if the UPPER confidence bound's predicted
    P(>=1 failure in next k) exceeds the lane failure budget. Using the upper CI bound
    (not the point estimate) is the conservative, safety-vetoing choice the V6 mandate
    demands -- a noisy small sample widens the CI and tightens the gate, never loosens it.
    """
    budget = DEFAULT_FAILURE_BUDGET if failure_budget is None else float(failure_budget)
    f = fit(counts, confidence=confidence)
    p_point = prob_at_least_one(f.lambda_hat, k)
    p_upper = prob_at_least_one(f.ci_high, k)
    hold = p_upper > budget
    return {
        "lambda_hat": f.lambda_hat,
        "lambda_ci": (f.ci_low, f.ci_high),
        "k": k,
        "p_fail_point": p_point,
        "p_fail_upper": p_upper,
        "failure_budget": budget,
        "hold": hold,
        "gate_ready": f.gate_ready,
        "is_poisson": f.is_poisson,
        "reason": (
            "upper-CI failure probability exceeds lane budget" if hold
            else "within failure budget"
        ),
    }


__all__ = [
    "PoissonFit", "fit", "fit_fast", "mle_lambda", "ewma_lambda",
    "prob_at_least_one", "prob_at_least",
    "lambda_ci_wald", "lambda_ci_exact",
    "dispersion_gof", "pearson_gof", "calibration_error",
    "hazard_prior", "gate_decision",
    "MIN_WINDOWS_FOR_GATE", "CALIBRATION_ERROR_CEILING", "DEFAULT_FAILURE_BUDGET",
]


if __name__ == "__main__":
    # Deterministic demo (no RNG): a seeded near-Poisson sample.
    demo = [0, 1, 0, 2, 1, 0, 1, 3, 0, 1, 1, 0, 2, 1, 0]
    f = fit(demo)
    print(f"lambda_hat = {f.lambda_hat:.4f}  SE = {f.std_error:.4f}  "
          f"CI{int(f.confidence*100)}% = ({f.ci_low:.4f}, {f.ci_high:.4f}) [{f.ci_method}]")
    print(f"P(>=1 in next 1) = {f.prob_at_least_one(1):.4f}   "
          f"P(>=1 in next 5) = {f.prob_at_least_one(5):.4f}")
    print(f"dispersion = {f.dispersion:.3f}  GoF p = {f.gof_pvalue:.3f}  "
          f"is_poisson = {f.is_poisson}  scipy = {_HAVE_SCIPY}")
    print("gate:", gate_decision(demo, k=3))
