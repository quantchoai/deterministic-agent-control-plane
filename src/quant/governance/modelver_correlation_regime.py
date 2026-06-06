"""V6 -- Systematic-vs-idiosyncratic CORRELATION-REGIME detector (PRIVATE / SHADOW).

The agent-management plane is deterministic; no LLM in the loop. This module answers a
single risk question on the fleet's *trouble matrix*: is the trouble we are seeing
SPREAD across domains (idiosyncratic -- normal, diversifiable) or is it MOVING TOGETHER
(systematic -- a common-factor shock the fleet cannot diversify away)?

It is the agent-management analogue of vendor/model correlation in a portfolio: when
one shared factor (an upstream model regression, an infra brown-out, a poisoned shared
context) hits many domains at once, the V5 risk budget that assumed independence is
wrong by a large multiple. The fix is to *detect the regime* and tighten the eligibility
bar / freeze critical lanes (FRN floating barrier, governance_v6.frn_barrier).

MATH (deterministic, closed-form; numpy used only as a linear-algebra calculator):

  1. Trouble matrix X  (T buckets  x  D domains): X[t,d] = a trouble/return score for
     domain d in time-bucket t (e.g. severity-weighted failure rate). Rows are repeated
     observations of the SAME D-dimensional vector, so co-movement across domains is
     exactly the cross-domain covariance.

  2. One-factor decomposition via the spectrum of the D x D covariance (or correlation)
     matrix  S = cov(X):
         eigvals  l_1 >= l_2 >= ... >= l_D >= 0      (S is PSD, symmetric)
         trace(S) = sum_d l_d = total variance
         systematic_share = l_1 / trace(S)            <-- top-eigenvalue share
     This is the fraction of total cross-domain variance explained by the single
     dominant principal component = the strength of the common factor. =1 means every
     domain is a scalar multiple of one factor (perfect co-movement); ~1/D means the
     domains are mutually uncorrelated (pure idiosyncratic, white).

  3. One-factor MLE / MoM loadings: the top eigenvector b (loadings) and residual
     (idiosyncratic) variances psi_d = S_dd - (l_1 * b_d^2) give the classic one-factor
     model  X_d = b_d * F + e_d. We report b and psi for interpretability and to feed a
     spectral / parallel-analysis floor.

  4. Null floor (Marchenko-Pastur / parallel analysis): under pure noise with T rows and
     D columns the top eigenvalue of the *correlation* matrix is not D-independent; the
     largest MP eigenvalue is  lambda_+ = (1 + sqrt(D/T))^2 . We convert that to an
     expected top-share floor so "high share" is judged against the noise baseline, not
     against 1/D blindly. The alarm uses an ABSOLUTE governance threshold (share crosses
     SYSTEMATIC_ALARM_THRESHOLD) AND requires share to clear the noise floor.

  5. Confidence: deterministic, seeded row-bootstrap CI on systematic_share (resample
     time-buckets with replacement, recompute the top-share). Two-sided at
     V6_REGIME.CONFIDENCE_LEVEL. No RNG state leaks (local Generator, fixed seed).

  6. Control stability: the bar lift fed back into eligibility is  lift = kappa * share
     with kappa = BAR_LIFT_KAPPA < 1, so the map share -> lift -> (next share) is a
     contraction (Lipschitz < 1) and the feedback cannot oscillate/blow up. We assert
     this invariant at import-check time.

BRIDGE TO governance_v6.factor_decomposition: that function computes a between/within
ANOVA share on a flat list of (factor,loss) pairs -- a *grouping* variance split, not a
cross-domain co-movement spectrum. Both are "systematic_share" in spirit; this module
exposes `anova_share()` so the eigenvalue regime view reconciles with the documented
v6_backtest number (0.027 on the real trouble set). The detector headline statistic is
the eigenvalue share (cross-domain co-movement); the ANOVA share is reported alongside.

STATUS: SHADOW / OFFLINE. Reads no Mongo, holds no LLM, mutates nothing. Promote via the
V6 gate only after calibration evidence. Money/severity inputs are deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from quant.governance import governance_params as GP

_V6 = GP.V6_REGIME

# Resolve governance constants once (env-overridable via the registry).
ALARM = float(_V6["SYSTEMATIC_ALARM_THRESHOLD"])
WARN = float(_V6["SYSTEMATIC_WARN_THRESHOLD"])
MIN_FACTORS = int(_V6["MIN_FACTORS"])
MIN_OBS_PER_FACTOR = int(_V6["MIN_OBS_PER_FACTOR"])
MIN_TOTAL_OBS = int(_V6["MIN_TOTAL_OBS"])
BOOTSTRAP_RESAMPLES = int(_V6["BOOTSTRAP_RESAMPLES"])
BOOTSTRAP_SEED = int(_V6["BOOTSTRAP_SEED"])
CONFIDENCE_LEVEL = float(_V6["CONFIDENCE_LEVEL"])
BAR_LIFT_KAPPA = float(_V6["BAR_LIFT_KAPPA"])
BACKTEST_SHARE = float(_V6["REGIME_BACKTEST_SHARE"])

# Control-stability invariant: the share->lift feedback must be a contraction.
assert 0.0 < BAR_LIFT_KAPPA < 1.0, "BAR_LIFT_KAPPA must be in (0,1) for a stable loop"


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegimeResult:
    systematic_share: float            # top-eigenvalue share of total variance (headline)
    regime: str                        # "systematic" | "watch" | "idiosyncratic" | "undetermined"
    alarm: bool                        # share >= ALARM and above noise floor and sample ok
    n_domains: int
    n_buckets: int
    eigenvalues: tuple                 # descending, of the covariance matrix
    loadings: tuple                    # top-eigenvector (one-factor loadings b_d), sign-fixed
    idiosyncratic_var: tuple           # psi_d = S_dd - l1*b_d^2  (per-domain residual var)
    noise_floor_share: float           # MP/parallel-analysis expected top-share under noise
    excess_over_noise: float           # systematic_share - noise_floor_share
    anova_share: float                 # between/within share (governance_v6 bridge), or nan
    ci_low: float                      # bootstrap CI on systematic_share
    ci_high: float
    bar_lift: float                    # kappa*share -> additive eligibility-bar lift
    use_correlation: bool              # standardized (corr) vs raw covariance
    note: str = ""
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Matrix construction from heterogeneous inputs
# --------------------------------------------------------------------------- #
def matrix_from_long(observations: Sequence[dict],
                     domain_key: str = "factor",
                     value_key: str = "loss",
                     bucket_key: str = "bucket") -> tuple:
    """Build a (T buckets x D domains) trouble matrix from long-format rows.

    Each row is {domain_key: <domain>, value_key: <float>, bucket_key?: <hashable>}.
    If no bucket_key is present, rows are slotted round-robin into per-domain sequence
    positions (observation index within a domain) so the d-th domain's k-th observation
    aligns with every other domain's k-th observation -- the standard panel alignment.
    Missing cells are filled with the column (domain) mean (zero-impact on covariance
    centering, the least-information imputation). Deterministic ordering throughout.
    Returns (X ndarray, domains list).
    """
    by_domain: dict[str, list[float]] = {}
    buckets_seen: dict = {}
    has_bucket = any(bucket_key in o for o in observations)

    if has_bucket:
        # explicit panel: index by (bucket, domain)
        cell: dict = {}
        for o in observations:
            d = str(o.get(domain_key, "?"))
            b = o.get(bucket_key, None)
            v = float(o.get(value_key, 0.0))
            buckets_seen.setdefault(b, len(buckets_seen))
            cell.setdefault((b, d), []).append(v)
            by_domain.setdefault(d, [])
        domains = sorted(by_domain.keys())
        bks = sorted(buckets_seen.keys(), key=lambda x: (x is None, x))
        T, D = len(bks), len(domains)
        X = np.full((T, D), np.nan, dtype=float)
        for ti, b in enumerate(bks):
            for di, d in enumerate(domains):
                vs = cell.get((b, d))
                if vs:
                    X[ti, di] = float(np.mean(vs))
    else:
        for o in observations:
            d = str(o.get(domain_key, "?"))
            by_domain.setdefault(d, []).append(float(o.get(value_key, 0.0)))
        domains = sorted(by_domain.keys())
        T = max((len(v) for v in by_domain.values()), default=0)
        D = len(domains)
        X = np.full((T, D), np.nan, dtype=float)
        for di, d in enumerate(domains):
            col = by_domain[d]
            for ti in range(len(col)):
                X[ti, di] = col[ti]

    # impute missing cells with column mean (nan-safe; all-nan column -> 0)
    if X.size:
        col_mean = np.nanmean(np.where(np.isnan(X), np.nan, X), axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    return X, domains


# --------------------------------------------------------------------------- #
# Core one-factor spectral decomposition
# --------------------------------------------------------------------------- #
def _covariance(X: np.ndarray, use_correlation: bool) -> np.ndarray:
    """Sample covariance (or correlation) of columns (domains). T rows, D cols.
    Centered; uses population normalization (ddof=0) for a PSD, trace-stable estimate."""
    T = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / max(1, T)               # D x D, PSD by construction
    if use_correlation:
        d = np.sqrt(np.clip(np.diag(S), 1e-18, None))
        S = S / np.outer(d, d)
        # zero-variance domains -> identity row/col (uncorrelated unit), keep PSD
        zero = np.diag(S) < 1e-12
        if zero.any():
            S[zero, :] = 0.0
            S[:, zero] = 0.0
            S[zero, zero] = 1.0
    return S


def _top_share(X: np.ndarray, use_correlation: bool) -> tuple:
    """Return (systematic_share, eigvals_desc, top_eigvec). Closed-form symmetric eig."""
    S = _covariance(X, use_correlation)
    # eigh: ascending eigenvalues of a symmetric matrix; clip tiny negatives from roundoff
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0.0, None)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    trace = float(w.sum())
    share = float(w[0] / trace) if trace > 1e-18 else float("nan")
    vec = V[:, 0]
    # sign convention: make the dominant loading positive (deterministic)
    if vec[np.argmax(np.abs(vec))] < 0:
        vec = -vec
    return share, w, vec, S


def anova_share(observations: Sequence[dict],
                domain_key: str = "factor", value_key: str = "loss") -> float:
    """Between/within (ANOVA) systematic share -- the governance_v6.factor_decomposition
    statistic, reproduced here so the eigenvalue regime view reconciles with the
    documented v6_backtest number (0.027). Pure stdlib math (no numpy needed)."""
    groups: dict[str, list[float]] = {}
    for o in observations:
        groups.setdefault(str(o.get(domain_key, "?")), []).append(float(o.get(value_key, 0.0)))
    allx = [x for g in groups.values() for x in g]
    if not allx:
        return float("nan")
    n = len(allx)
    grand = sum(allx) / n
    means = {f: (sum(v) / len(v)) for f, v in groups.items()}
    between = sum(len(v) * (means[f] - grand) ** 2 for f, v in groups.items()) / n
    within = sum((x - means[f]) ** 2 for f, v in groups.items() for x in v) / n
    total = between + within
    return between / total if total > 1e-18 else float("nan")


# --------------------------------------------------------------------------- #
# Noise floor -- Marchenko-Pastur top eigenvalue -> expected top-share under pure noise
# --------------------------------------------------------------------------- #
def noise_floor_share(T: int, D: int, use_correlation: bool) -> float:
    """Expected systematic_share if the data were pure independent noise.

    For the CORRELATION matrix of T iid rows x D cols, the bulk of the spectrum follows
    Marchenko-Pastur with ratio q = D/T; the largest bulk eigenvalue is
        lambda_+ = (1 + sqrt(q))^2 .
    trace(corr) = D, so the noise top-share floor is lambda_+ / D. Clamped to [1/D, 1].
    For the covariance branch we fall back to the white-noise share 1/D (scale-free).
    """
    if D <= 0:
        return float("nan")
    base = 1.0 / D
    if not use_correlation or T <= 1:
        return base
    q = D / T
    lam_plus = (1.0 + math.sqrt(q)) ** 2
    return float(min(1.0, max(base, lam_plus / D)))


# --------------------------------------------------------------------------- #
# Deterministic, seeded bootstrap CI on systematic_share
# --------------------------------------------------------------------------- #
def _bootstrap_ci(X: np.ndarray, use_correlation: bool,
                  resamples: int, seed: int, conf: float) -> tuple:
    T = X.shape[0]
    if T < 3 or resamples <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    shares = np.empty(resamples, dtype=float)
    k = 0
    for _ in range(resamples):
        idx = rng.integers(0, T, size=T)
        try:
            s, _w, _v, _S = _top_share(X[idx, :], use_correlation)
        except np.linalg.LinAlgError:
            continue
        if math.isfinite(s):
            shares[k] = s
            k += 1
    if k < 2:
        return (float("nan"), float("nan"))
    shares = np.sort(shares[:k])
    a = (1.0 - conf) / 2.0
    lo = float(np.quantile(shares, a))
    hi = float(np.quantile(shares, 1.0 - a))
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Public detector
# --------------------------------------------------------------------------- #
def detect_regime(observations: Sequence[dict] | None = None,
                  matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
                  domains: Sequence[str] | None = None,
                  domain_key: str = "factor",
                  value_key: str = "loss",
                  bucket_key: str = "bucket",
                  use_correlation: bool = True,
                  with_ci: bool = True) -> RegimeResult:
    """Detect systematic vs idiosyncratic regime on the domain trouble/return matrix.

    Provide EITHER long-format `observations` (list of {factor,loss[,bucket]}) OR a
    pre-built `matrix` (T x D) with `domains`. systematic_share = top-eigenvalue share
    of total variance. Alarm fires when share crosses the governance ALARM threshold,
    clears the noise floor, and the sample is non-degenerate.
    """
    av = float("nan")
    if matrix is not None:
        X = np.asarray(matrix, dtype=float)
        if X.ndim != 2:
            return _undetermined(X.shape, "matrix must be 2-D (buckets x domains)")
        doms = list(domains) if domains is not None else [f"d{i}" for i in range(X.shape[1])]
    elif observations is not None:
        X, doms = matrix_from_long(observations, domain_key, value_key, bucket_key)
        av = anova_share(observations, domain_key, value_key)
    else:
        return _undetermined((0, 0), "no observations or matrix supplied")

    T, D = (X.shape if X.ndim == 2 else (0, 0))

    # --- degeneracy / sample guards ---
    total_obs = int(np.count_nonzero(~np.isnan(X))) if X.size else 0
    if D < MIN_FACTORS:
        return _undetermined((T, D), f"need >= {MIN_FACTORS} domains, have {D}", av)
    if T < MIN_OBS_PER_FACTOR:
        return _undetermined((T, D), f"need >= {MIN_OBS_PER_FACTOR} buckets/domain, have {T}", av)
    if total_obs < MIN_TOTAL_OBS:
        return _undetermined((T, D), f"need >= {MIN_TOTAL_OBS} total obs, have {total_obs}", av)

    share, eigs, vec, S = _top_share(X, use_correlation)
    floor = noise_floor_share(T, D, use_correlation)
    excess = (share - floor) if math.isfinite(share) else float("nan")

    # one-factor residual (idiosyncratic) variances psi_d = S_dd - l1*b_d^2
    l1 = float(eigs[0])
    psi = np.clip(np.diag(S) - l1 * (vec ** 2), 0.0, None)

    ci_lo, ci_hi = (float("nan"), float("nan"))
    if with_ci:
        ci_lo, ci_hi = _bootstrap_ci(X, use_correlation, BOOTSTRAP_RESAMPLES,
                                     BOOTSTRAP_SEED, CONFIDENCE_LEVEL)

    # --- regime classification (deterministic) ---
    above_floor = math.isfinite(excess) and excess > 0.0
    if not math.isfinite(share):
        regime, alarm = "undetermined", False
    elif share >= ALARM and above_floor:
        regime, alarm = "systematic", True
    elif share >= WARN and above_floor:
        regime, alarm = "watch", False
    else:
        regime, alarm = "idiosyncratic", False

    bar_lift = BAR_LIFT_KAPPA * share if math.isfinite(share) else 0.0

    note = (f"share={share:.3f} vs alarm={ALARM:.2f}/floor={floor:.3f} -> {regime}. "
            + ("ALARM: common-factor shock; raise eligibility bar / freeze critical lanes."
               if alarm else "no alarm: trouble is diversified across domains."))
    return RegimeResult(
        systematic_share=share, regime=regime, alarm=alarm,
        n_domains=D, n_buckets=T,
        eigenvalues=tuple(round(float(e), 6) for e in eigs),
        loadings=tuple(round(float(b), 6) for b in vec),
        idiosyncratic_var=tuple(round(float(p), 6) for p in psi),
        noise_floor_share=round(float(floor), 6),
        excess_over_noise=round(float(excess), 6) if math.isfinite(excess) else float("nan"),
        anova_share=round(float(av), 6) if math.isfinite(av) else float("nan"),
        ci_low=round(float(ci_lo), 6) if math.isfinite(ci_lo) else float("nan"),
        ci_high=round(float(ci_hi), 6) if math.isfinite(ci_hi) else float("nan"),
        bar_lift=round(float(bar_lift), 6),
        use_correlation=use_correlation,
        note=note,
        detail={"domains": list(doms), "trace": round(float(eigs.sum()), 6)},
    )


def _undetermined(shape, note: str, anova: float = float("nan")) -> RegimeResult:
    T, D = (shape if len(shape) == 2 else (0, 0))
    return RegimeResult(
        systematic_share=float("nan"), regime="undetermined", alarm=False,
        n_domains=int(D), n_buckets=int(T), eigenvalues=tuple(), loadings=tuple(),
        idiosyncratic_var=tuple(), noise_floor_share=float("nan"),
        excess_over_noise=float("nan"),
        anova_share=round(float(anova), 6) if math.isfinite(anova) else float("nan"),
        ci_low=float("nan"), ci_high=float("nan"), bar_lift=0.0,
        use_correlation=True, note=note,
    )


# --------------------------------------------------------------------------- #
# Convenience: map a regime result to an eligibility-bar lift (V6 FRN bridge)
# --------------------------------------------------------------------------- #
def bar_lift_for(result: RegimeResult) -> float:
    """Additive lift to a critical-lane eligibility bar, = kappa * systematic_share,
    a contraction (kappa<1) so the feedback loop is stable. 0 if undetermined."""
    s = result.systematic_share
    return BAR_LIFT_KAPPA * s if math.isfinite(s) else 0.0


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    # common-factor synthetic demo
    rng = np.random.default_rng(1)
    F = rng.normal(size=40)
    Xc = np.column_stack([1.0 * F + 0.15 * rng.normal(size=40) for _ in range(5)])
    r = detect_regime(matrix=Xc, domains=[f"dom{i}" for i in range(5)])
    print("common-factor  :", r.regime, round(r.systematic_share, 3), "alarm=", r.alarm)
    Xi = rng.normal(size=(40, 5))
    r2 = detect_regime(matrix=Xi)
    print("independent    :", r2.regime, round(r2.systematic_share, 3), "alarm=", r2.alarm)
    print(json.dumps({"common": r.note, "indep": r2.note}, indent=2))
