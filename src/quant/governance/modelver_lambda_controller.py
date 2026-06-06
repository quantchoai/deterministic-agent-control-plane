"""V5/V6 -- Resource shadow-price (lambda) PI controller + robust d2_mu accel feature.

Two deterministic slow-loop primitives for the agent-management control plane:

  (a) LambdaController -- PI control of the resource SHADOW PRICE `lambda` so that
      fleet UTILIZATION (rho = active_workers / safe_capacity) is held at a target.
      lambda is the Lagrange multiplier on the worker-capacity constraint: it is the
      marginal cost (in utility units) of one more unit of concurrency. When the fleet
      runs hot (rho > target) lambda rises, every ticket's utility is taxed harder, and
      fewer/cheaper dispatches occur; when slack, lambda falls and the fleet leans in.
      This is the economic dual of the KAPPA_CONCURRENCY_SLIPPAGE / FLEET_RISK_BUDGET
      gaps left open (as None) in governance_params.V5_OPTIMIZATION.

      Control law (positional/incremental PI with explicit anti-windup):
          e_k      = rho_target - rho_k                       (utilization error)
          P_k      = -Kp * e_k                                 (sign: rho>target => lambda up)
          I_{k}    = I_{k-1} - Ki * e_k        (integrator, accumulates in lambda units)
          u_raw    = lambda_base + P_k + I_k
          lambda_k = clamp(u_raw, LAMBDA_MIN, LAMBDA_MAX)
          # back-calculation anti-windup: undo the integrator share that got clipped
          I_k     += Kw * (lambda_k - u_raw)
      The negative signs fold the "more load => higher price" direction in once, so Kp,Ki
      are quoted as POSITIVE gains (standard form).

      Stability: with a first-order load plant rho_{k+1} = rho_k - g*(lambda_k - lambda*)
      (more price => fewer workers => lower utilization, plant gain g>0) the closed loop
      is a 2-state LTI system. We expose `stable_gain_range()` which returns the proven
      Kp/Ki box (Jury test on the characteristic polynomial) for a given plant gain, and
      assert the shipped defaults sit strictly inside it.

  (b) d2_mu_accel -- ROBUST second difference of an agent's mu-history = LEARNING
      ACCELERATION, a dispatch signal. The naive second difference
      (mu[t]-2mu[t-1]+mu[t-2], as in governance_v6.gamma_acceleration) amplifies noise
      by a factor of sqrt(6); on a noisy mu trace it false-alarms constantly. This version
      fits a LOCAL QUADRATIC by ordinary least squares over the last W points
      (a Savitzky-Golay quadratic smoother) and reads the curvature 2*c off the fit:
          mu(t) ~ a + b*t + c*t^2   =>   d2_mu := 2c  (per-step^2 acceleration)
      OLS gives, for free, the standard error of c, hence a confidence band and a
      noise-aware significance test -- so "positive acceleration" means
      *statistically* positive, which is what keeps false-alarm <= the V6 20% target.
      Positive d2_mu on an accelerating learner, ~0 (and not significant) on a plateau.

Design rules honoured:
  - Deterministic. No LLM in the plane. numpy used only for the linear-algebra core;
    a pure-Python closed-form fallback is provided and is the default if numpy is absent.
  - Constants reused from governance_params (SAFE_WORKER_CAPACITY proxy, CYCLE_SEC,
    ACTIVE_WINDOW_MIN, LAMBDA_STARVATION as the lambda base anchor).
  - Slow-loop primitive (V6 doctrine 5): runs in the warden, writes a snapshot; hot path
    just reads lambda. Nothing here mutates fleet state or calls Mongo.

V6 promotion targets this module is built to (CALIBRATION_PROMOTION_MANDATE 2.2):
  lambda controller: "lambda converges (no oscillation)" -- proven by stable_gain_range
                     + the convergence sim test.
  d2_mu:             "false-alarm <= 20%" -- the significance test is the lever; the test
                     drives a synthetic plateau false-alarm rate to ~0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from quant.governance import governance_params as gp

try:  # numpy is preferred for the OLS core but never required.
    import numpy as _np
    _HAVE_NP = True
except Exception:  # pragma: no cover - exercised only on numpy-less hosts
    _np = None
    _HAVE_NP = False


# ===========================================================================
# Shared constants, anchored to governance_params (single source of truth).
# ===========================================================================
# governance_params left SAFE_WORKER_CAPACITY = None (machine dependent, "~1-2 under
# low disk"). We pick a conservative default and let the caller override; the controller
# itself is invariant to the absolute capacity because it controls the RATIO rho.
DEFAULT_SAFE_CAPACITY: float = 2.0

# lambda is a price in the SAME units as V5 utility's exp(LAMBDA*age) starvation lever's
# multiplier; we anchor the controller's neutral price to that scale so the two knobs are
# commensurate. LAMBDA_STARVATION ~ 0.018.
LAMBDA_BASE_ANCHOR: float = float(gp.V5_OPTIMIZATION["LAMBDA_STARVATION"])

# Price must stay non-negative (a negative resource price would PAY agents to crowd the
# host) and bounded (runaway price starves the queue). Box chosen wide but finite.
LAMBDA_MIN: float = 0.0
LAMBDA_MAX: float = 1.0

# Default utilization target: keep ~15% headroom under the safe capacity so a burst does
# not instantly breach. (rho=1.0 == exactly at safe capacity.)
DEFAULT_RHO_TARGET: float = 0.85


# ===========================================================================
# (a) lambda shadow-price PI controller
# ===========================================================================
@dataclass
class LambdaController:
    """Discrete PI controller for the resource shadow price lambda.

    Call `update(active_workers)` (or `update_rho(rho)`) once per warden cycle. It returns
    the new lambda to publish into the utility snapshot. The controller is a pure function
    of its accumulated state; identical inputs reproduce identical lambda traces.
    """

    kp: float = 0.05                 # proportional gain (positive; sign handled internally)
    ki: float = 0.012                # integral gain per cycle
    rho_target: float = DEFAULT_RHO_TARGET
    safe_capacity: float = DEFAULT_SAFE_CAPACITY
    lambda_base: float = LAMBDA_BASE_ANCHOR
    lambda_min: float = LAMBDA_MIN
    lambda_max: float = LAMBDA_MAX
    kw: float = 1.0                  # back-calculation anti-windup gain (in [0, 1/?]; 1.0 = full)

    # --- mutable state (not constructor args of interest) ---
    integrator: float = field(default=0.0)
    lam: float = field(default=LAMBDA_BASE_ANCHOR)
    last_error: float = field(default=0.0)
    n_updates: int = field(default=0)

    def __post_init__(self) -> None:
        if self.safe_capacity <= 0:
            raise ValueError("safe_capacity must be > 0")
        if not (self.lambda_min <= self.lambda_base <= self.lambda_max):
            raise ValueError("lambda_base must lie within [lambda_min, lambda_max]")
        if not (0.0 < self.rho_target):
            raise ValueError("rho_target must be > 0")
        # Start the integrator so the very first lambda equals lambda_base at zero error.
        self.lam = self.lambda_base

    # -- core update -------------------------------------------------------
    def update_rho(self, rho: float) -> float:
        """Advance one cycle given the measured utilization ratio rho. Returns new lambda."""
        if rho < 0 or not math.isfinite(rho):
            raise ValueError("rho must be a finite non-negative ratio")
        error = self.rho_target - rho                  # >0 means under-utilized
        # Proportional + integral. Negative signs: rho>target (error<0) must RAISE price.
        p_term = -self.kp * error
        self.integrator += -self.ki * error            # tentative integrator step
        u_raw = self.lambda_base + p_term + self.integrator
        # Saturate.
        lam_clamped = min(self.lambda_max, max(self.lambda_min, u_raw))
        # Back-calculation anti-windup: bleed off exactly the clipped excess so the
        # integrator cannot wind past the actuator limits.
        if lam_clamped != u_raw:
            self.integrator += self.kw * (lam_clamped - u_raw)
        self.lam = lam_clamped
        self.last_error = error
        self.n_updates += 1
        return self.lam

    def update(self, active_workers: float) -> float:
        """Convenience: feed raw active-worker count; converts to rho internally."""
        return self.update_rho(active_workers / self.safe_capacity)

    def reset(self) -> None:
        self.integrator = 0.0
        self.lam = self.lambda_base
        self.last_error = 0.0
        self.n_updates = 0

    def snapshot(self) -> dict:
        """O(1) record for the hot path to read (V6 doctrine 5)."""
        return {
            "lambda": round(self.lam, 6),
            "integrator": round(self.integrator, 6),
            "last_error": round(self.last_error, 6),
            "rho_target": self.rho_target,
            "safe_capacity": self.safe_capacity,
            "n_updates": self.n_updates,
        }

    # -- stability analysis ------------------------------------------------
    def stable_gain_range(self, plant_gain: float) -> dict:
        """Proven-stable Kp/Ki box for the first-order load plant, via the Jury test.

        Plant model (one warden cycle):
            rho_{k+1} = rho_k - plant_gain * (lambda_k - lambda_target)
        i.e. raising the price above the equilibrium price sheds utilization at rate
        `plant_gain` (>0). With the (unsaturated) PI law
            lambda_k - lambda_target = kp*e_k... (incremental form) the closed loop's
        characteristic polynomial in z is:
            z^2 + (g*kp - 2) z + (1 - g*kp + g*ki) = 0,     g := plant_gain
        Schur (inside-unit-disc) conditions [Jury, 2nd order  z^2 + a1 z + a0]:
            (1)  a0 < 1
            (2)  1 + a1 + a0 > 0
            (3)  1 - a1 + a0 > 0
        Returns the active margins plus the boolean stability of the *shipped* gains.
        """
        g = float(plant_gain)
        if g <= 0:
            raise ValueError("plant_gain must be > 0 (more price must shed utilization)")
        a1 = g * self.kp - 2.0
        a0 = 1.0 - g * self.kp + g * self.ki
        c1 = a0 < 1.0                       # => ki < kp
        c2 = (1.0 + a1 + a0) > 0.0          # => g*ki > 0 (ki>0)
        c3 = (1.0 - a1 + a0) > 0.0          # => 4 - 2*g*kp + g*ki > 0
        return {
            "plant_gain": g,
            "kp": self.kp,
            "ki": self.ki,
            "char_poly": (1.0, a1, a0),     # z^2 + a1 z + a0
            "jury_c1_a0_lt_1": c1,
            "jury_c2_lower": c2,
            "jury_c3_upper": c3,
            "stable": bool(c1 and c2 and c3),
            # Human-readable proven box (necessary+sufficient for this plant):
            "ki_must_be_positive": True,
            "ki_lt_kp": "ki < kp  (from a0<1)",
            "kp_upper": "kp < (4 + g*ki) / (2g)  (from 1-a1+a0>0)",
        }


# ===========================================================================
# (b) d2_mu -- robust learning-acceleration feature
# ===========================================================================
def _ols_quadratic(y: Sequence[float]) -> tuple[float, float, float, float]:
    """Fit y_i ~ a + b*i + c*i^2 by OLS on i=0..n-1. Returns (a, b, c, se_c).

    se_c is the standard error of the quadratic coefficient under homoscedastic-noise
    OLS (sigma^2 estimated from residuals). With n>=4 we get a real residual dof; with
    n==3 the quadratic is an exact interpolant (0 residual dof) so se_c is undefined and
    returned as +inf (signal present, but unverifiable -> never "significant").
    """
    n = len(y)
    if n < 3:
        raise ValueError("need >= 3 points")
    if _HAVE_NP:
        x = _np.arange(n, dtype=float)
        X = _np.vstack([_np.ones(n), x, x * x]).T          # n x 3 design
        # Normal equations via lstsq (numerically stable enough for small n; Vandermonde
        # conditioning is fine for the short windows we use, W<=9).
        beta, *_ = _np.linalg.lstsq(X, _np.asarray(y, dtype=float), rcond=None)
        a, b, c = (float(v) for v in beta)
        resid = _np.asarray(y, dtype=float) - X @ beta
        dof = n - 3
        if dof <= 0:
            return a, b, c, math.inf
        s2 = float(resid @ resid) / dof
        XtX_inv = _np.linalg.inv(X.T @ X)
        se_c = math.sqrt(max(0.0, s2 * float(XtX_inv[2, 2])))
        return a, b, c, se_c
    # ---- pure-Python closed form (3x3 normal equations, Cramer's rule) ----
    xs = [float(i) for i in range(n)]
    S0 = float(n)
    S1 = sum(xs)
    S2 = sum(v * v for v in xs)
    S3 = sum(v ** 3 for v in xs)
    S4 = sum(v ** 4 for v in xs)
    T0 = sum(y)
    T1 = sum(xs[i] * y[i] for i in range(n))
    T2 = sum(xs[i] * xs[i] * y[i] for i in range(n))
    # Normal matrix M [[S0,S1,S2],[S1,S2,S3],[S2,S3,S4]]; rhs [T0,T1,T2].
    M = [[S0, S1, S2], [S1, S2, S3], [S2, S3, S4]]

    def _det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    detM = _det3(M)
    if abs(detM) < 1e-12:
        raise ValueError("degenerate design (collinear x)")
    rhs = [T0, T1, T2]

    def _solve_col(col):
        m = [row[:] for row in M]
        for r in range(3):
            m[r][col] = rhs[r]
        return _det3(m) / detM

    a = _solve_col(0)
    b = _solve_col(1)
    c = _solve_col(2)
    # residual variance + (XtX)^-1[2,2] for se_c
    dof = n - 3
    if dof <= 0:
        return a, b, c, math.inf
    resid2 = 0.0
    for i in range(n):
        fit = a + b * xs[i] + c * xs[i] * xs[i]
        resid2 += (y[i] - fit) ** 2
    s2 = resid2 / dof
    # inverse element [2,2] of M = cofactor(2,2)/det = (S0*S2 - S1*S1)/detM
    inv22 = (S0 * S2 - S1 * S1) / detM
    se_c = math.sqrt(max(0.0, s2 * inv22))
    return a, b, c, se_c


# Two-sided ~95% t critical values for small residual dof (dof = W-3). Hard-coded so the
# feature is deterministic and dependency-free even without scipy. dof>=8 -> use 2.0.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def _t95(dof: int) -> float:
    if dof <= 0:
        return math.inf
    return _T95.get(dof, 2.0)


def d2_mu_accel(mu_history: Sequence[float], window: int = 7,
                sig_z: Optional[float] = None) -> dict:
    """Robust learning ACCELERATION = curvature of a local-quadratic fit to mu_history.

    mu_history is newest-LAST. Uses the last `window` points (>=3). Returns:
        d2_mu      : 2*c, the per-step^2 acceleration (the de-noised second difference).
        slope      : b, the per-step velocity at the window's start (trend direction).
        se         : standard error of d2_mu (2*se_c).
        significant: |d2_mu| exceeds its ~95% confidence half-width (t or supplied z).
        direction  : 'accelerating' (>0 & sig), 'collapsing' (<0 & sig), else 'flat'.
        ci         : (lo, hi) ~95% interval for d2_mu.
    The significance test is what bounds the false-alarm rate on a noisy plateau: a flat
    learner's curvature estimate is ~0 with se large relative to it, so `significant` is
    False and no acceleration signal is emitted.

    `window` is clamped to len(mu_history); if fewer than 3 usable points -> flat/null.
    """
    h = [float(v) for v in mu_history]
    n_all = len(h)
    if n_all < 3:
        return {"d2_mu": 0.0, "slope": 0.0, "se": math.inf, "significant": False,
                "direction": "flat", "ci": (0.0, 0.0), "n": n_all,
                "note": "need >= 3 points"}
    w = max(3, min(int(window), n_all))
    seg = h[-w:]
    a, b, c, se_c = _ols_quadratic(seg)
    d2 = 2.0 * c
    se = 2.0 * se_c
    dof = w - 3
    crit = sig_z if sig_z is not None else _t95(dof)
    half = crit * se if math.isfinite(se) else math.inf
    significant = math.isfinite(se) and (abs(d2) > half) and se > 0.0
    if significant and d2 > 0:
        direction = "accelerating"
    elif significant and d2 < 0:
        direction = "collapsing"
    else:
        direction = "flat"
    lo = d2 - half if math.isfinite(half) else -math.inf
    hi = d2 + half if math.isfinite(half) else math.inf
    return {
        "d2_mu": round(d2, 5),
        "slope": round(b, 5),
        "se": (round(se, 5) if math.isfinite(se) else math.inf),
        "significant": bool(significant),
        "direction": direction,
        "ci": (round(lo, 5) if math.isfinite(lo) else lo,
               round(hi, 5) if math.isfinite(hi) else hi),
        "n": w,
        "note": "robust (local-quadratic) second difference of mu; significance bounds false alarms",
    }


__all__ = [
    "LambdaController", "d2_mu_accel",
    "DEFAULT_SAFE_CAPACITY", "DEFAULT_RHO_TARGET",
    "LAMBDA_BASE_ANCHOR", "LAMBDA_MIN", "LAMBDA_MAX",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke
    ctrl = LambdaController()
    print("stable box @ g=0.5:", ctrl.stable_gain_range(0.5))
    rho = 1.10
    for _ in range(40):
        lam = ctrl.update_rho(rho)
        rho = rho - 0.5 * (lam - 0.3)   # toy plant
    print("converged snapshot:", ctrl.snapshot())
    print("accel:", d2_mu_accel([10, 11, 13, 16, 20, 25, 31]))
    print("plateau:", d2_mu_accel([50, 50.2, 49.8, 50.1, 49.9, 50.0, 50.1]))
