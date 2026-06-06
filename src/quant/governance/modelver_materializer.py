"""modelver_materializer.py -- the SLOW-LOOP materializer (the warden tick).

PRIVATE / SHADOW + OFFLINE. Deterministic, NO LLM, NO Mongo, NO network, NO RNG state
leak. This is the single place where ALL the heavy V6 math runs: it fits the
distributions, advances the controllers, runs the tail-risk / regime / sandbox
accounting, and bakes every result into ONE precomputed snapshot dict that the O(1)
hot path reads with nothing more than a dictionary lookup.

------------------------------------------------------------------------------------
WHERE THIS SITS  (ADVANCED_QUANT_ROADMAP_PRIVATE.md, "Architecture")
------------------------------------------------------------------------------------
    SLOW LOOP (warden / materializer, every few seconds):
        fit distributions (Poisson lambda, Gamma params)
        update acceleration features, shadow-price lambdas
        run CVaR / correlation-regime / sandbox accounting
        -> write all of it into agent_ledger.snapshot
    HOT PATH (dispatch, per task):
        read precomputed snapshot fields  (O(1))   <-- never calls this module

`materialize(fleet_state)` is a PURE FUNCTION: snapshot = f(fleet_state). Given the same
fleet_state it returns a byte-stable snapshot (idempotent), because every primitive it
calls is itself deterministic (closed-form fits, seeded bootstrap, fixed controller
state) and the function performs NO I/O and reads NO clock.

------------------------------------------------------------------------------------
DOCTRINE HONOURED (load-bearing -- see the prompt + the per-module headers)
------------------------------------------------------------------------------------
* SHADOW: this computes + logs; it never kills. The snapshot carries advisory fields
  (eligible_lanes, frn_barrier, a `shadow` flag) but issues no guillotine. The hot path
  decides actions; severity that matches a *deterministic* oracle is the only thing that
  may ever drive a hard action, and that decision is NOT taken here.
* SEVERITY MATCHES ORACLE DETERMINISM: a probabilistic estimate (lambda_fail, fracture,
  systematic_share) may only adjust sigma / raise a flag / lift the eligibility bar --
  NEVER guillotine. So this module's only feedback into credit is a *sigma widening*
  (uncertainty inflation) when the fitted Poisson is non-Poisson / over-dispersed or the
  fleet is in a systematic-shock regime. mu is never moved here, hp is never cut here.
* HEAVY MATH IN THE SLOW LOOP: all calculus (MLE/Newton, eigh, bootstrap, OLS, CVaR
  integral) lives here. The snapshot stores scalars/tuples the hot path reads in O(1).
* MONEY/SECURITY NEVER GO SHADOW->FULL: the eligible-lanes computation FAILS CLOSED --
  a money/security/live-critical lane is removed from an agent's eligible set unless the
  agent independently clears the V2 hard gate (hp>80, mu>75, sigma<15). The materializer
  can only ever REMOVE a critical lane (tighten), never grant one.

------------------------------------------------------------------------------------
INPUT  (fleet_state)  -- tolerant: every field has a safe default
------------------------------------------------------------------------------------
{
  "agents": [
     {
       "agent_id": str,
       "domain":   str,                 # canonical dispatcher domain (Risk, Tax, ...)
       "mu":       float,               # current V2 credit mean (0..100)
       "sigma":    float,               # current V2 credit spread (3..60)
       "hp":       float,               # current V1 survival HP (0..100)
       "failure_counts": [int, ...],    # per-window failure arrivals  (Poisson fit)
       "latency_samples": [float, ...], # ticket latencies > 0         (Gamma fit)
       "mu_history": [float, ...],      # newest-last mu trace          (d2_mu accel)
       "fracture_features": {failed_requeue, critical_file_touch, ci_failure,
                             stale_lease_minutes, traceback_noise_kb, cross_domain_deps},
       "candidate_lanes": [ {lane|name|domain, money_risk, security_risk, live_risk}
                            | str, ... ],   # lanes this agent might be routed to
     }, ...
  ],
  "fleet": {
     "active_workers":  float,          # for the lambda PI controller (rho = active/safe)
     "safe_capacity":   float,          # denominator for rho (default DEFAULT_SAFE_CAPACITY)
     "lambda_state":  {integrator, lam, last_error, n_updates}?,  # controller carry-over
     "cvar_limit":      float,          # CVaR budget the gate is sized against (optional)
     "regime_observations": [ {factor|domain, loss[, bucket]}, ... ]?,  # for the regime
     "frn_base_bar":    float,          # base eligibility bar (default CRITICAL_CONFIDENCE_MIN_P)
  }
}

------------------------------------------------------------------------------------
OUTPUT  (snapshot)  -- the documented shape (every field always present)
------------------------------------------------------------------------------------
{
  "schema_version": int,
  "agents": {
     "<agent_id>": {
        "mu": float, "sigma": float, "hp": float,      # carried/possibly sigma-widened
        "lambda_fail": float,    # fitted Poisson rate (failures/window), EWMA-tracked
        "latency_q": {p50,p95,p99},   # Gamma SLA quantiles for the V5 cost/tail term
        "d2_mu": float,          # robust learning acceleration (curvature of mu trace)
        "fracture": float,       # 0..1 fragility score
        "eligible_lanes": [str], # lanes this agent may serve (fail-closed on critical)
        "shadow": {...},         # advisory diagnostics (never an action)
     }, ...
  },
  "fleet": {
     "lambdas": {lambda, integrator, last_error, rho, rho_target, n_updates},
     "systematic_share": float,    # top-eigenvalue co-movement share (regime headline)
     "frn_barrier": float,         # floating eligibility bar (rises in a degraded fleet)
     "cvar": {alpha, VaR, CVaR, CVaR_upper, utilisation, ok, ...},
  }
}
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from quant.governance import governance_params as gp

from quant.governance import modelver_poisson_failure as poisson
from quant.governance import modelver_gamma_latency as gamma
from quant.governance import modelver_cvar_budget as cvar
from quant.governance import modelver_lambda_controller as lam_ctrl
from quant.governance import modelver_oracle_ladder as oracle
from quant.governance import modelver_ucb_sandbox as sandbox
from quant.governance import governance_v6 as gv6

try:  # the correlation-regime detector hard-depends on numpy; degrade gracefully.
    from quant.governance import modelver_correlation_regime as regime
    _HAVE_REGIME = True
except Exception:  # pragma: no cover - exercised only on numpy-less hosts
    regime = None
    _HAVE_REGIME = False


SCHEMA_VERSION = 1

# ----------------------------------------------------------------------------------
# Constants pulled from the single source of truth (governance_params).
# ----------------------------------------------------------------------------------
_V2 = gp.V2_CREDIT
_V5 = gp.V5_OPTIMIZATION
_GATE_MIN_HP = float(_V2["MONEY_SECURITY_GATE_MIN_HP"])      # 80
_GATE_MIN_MU = float(_V2["MONEY_SECURITY_GATE_MIN_MU"])      # 75
_GATE_MAX_SIGMA = float(_V2["MONEY_SECURITY_GATE_MAX_SIGMA"])  # 15
_SIGMA_LO, _SIGMA_HI = _V2["SIGMA_CLAMP"]                    # (3, 60)
_MU_LO, _MU_HI = _V2["MU_CLAMP"]                             # (0, 100)
_CRITICAL_RISK = float(gp.V6_MONEY_CRITICAL["EXPLICIT_RISK_CRITICAL_THRESHOLD"])  # 3

# FRN floating-barrier base bar = the critical-lane min success probability (V5).
_FRN_BASE_BAR = float(_V5["CRITICAL_CONFIDENCE_MIN_P"])      # 0.86
_DEFAULT_CVAR_ALPHA = cvar.DEFAULT_ALPHA                     # 0.95

# Maximum sigma WIDENING the materializer may request as an uncertainty flag. Mirrors the
# oracle-ladder shadow cap: a probabilistic signal may nudge sigma up, never tighten.
_MAX_SIGMA_WIDEN = 4.0

# A fleet is "degraded" (FRN bar lifts) when EITHER the regime turns systematic OR the
# realized tail (CVaR) over-runs its budget. This kappa scales the budget-overrun lift;
# kept < 1 with the regime kappa so the combined lift map stays a contraction.
_FRN_CVAR_KAPPA = 0.12

# Non-critical lane shedding budget. The FRN bar is a *critical-lane* success bar; a
# NON-critical lane should not inherit that strict ceiling. We shed a low-risk lane only
# when the agent's forward failure probability blows the V5 low-confidence budget
# (1 - LOW_CONFIDENCE_MIN_P ~ 0.28) -- and we tighten that budget proportionally as the
# fleet degrades (FRN bar lifts above its base), so a degraded fleet sheds sooner.
_LOW_CONF_FAIL_BUDGET = 1.0 - float(_V5["LOW_CONFIDENCE_MIN_P"])  # ~0.28


# ----------------------------------------------------------------------------------
# small, dependency-free coercion helpers (deterministic; never raise)
# ----------------------------------------------------------------------------------
def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _round(x: float, nd: int = 6) -> float:
    """Deterministic rounding that leaves non-finite values intact (NaN/inf pass through
    only where explicitly allowed; here we coerce non-finite to 0.0 to keep snapshots
    JSON-stable and idempotent)."""
    if not math.isfinite(x):
        return 0.0
    return round(float(x), nd)


def _int_counts(seq: Any) -> list[int]:
    if not seq:
        return []
    out = []
    for c in seq:
        try:
            out.append(max(0, int(c)))
        except (TypeError, ValueError):
            out.append(0)
    return out


def _pos_samples(seq: Any) -> list[float]:
    if not seq:
        return []
    out = []
    for x in seq:
        f = _num(x, 0.0)
        if f > 0.0 and math.isfinite(f):
            out.append(f)
    return out


# ----------------------------------------------------------------------------------
# Per-agent feature materialization (the heavy math, all closed-form/deterministic)
# ----------------------------------------------------------------------------------
def _materialize_lambda_fail(failure_counts: list[int]) -> dict:
    """Fit Poisson(lambda) to the per-window failure stream.

    Headline `lambda_fail` is the EWMA rate (recent windows weighted up) so the hot-path
    hazard prior tracks regime shifts -- exactly poisson.hazard_prior's input. We ALSO
    keep the unbiased MLE + the dispersion goodness-of-fit so the snapshot can flag a
    NON-Poisson (over-dispersed, bursty) stream: that flag is the only thing allowed to
    widen sigma (a probabilistic signal -> uncertainty inflation, never a kill)."""
    if not failure_counts:
        return {
            "lambda_fail": 0.0, "lambda_mle": 0.0, "p_fail_next": 0.0,
            "is_poisson": True, "over_dispersed": False, "dispersion": 0.0,
            "n_windows": 0, "gate_ready": False,
        }
    # Streaming fast path: lambda_hat, dispersion, GoF and the over-dispersion flag all
    # come from ONE pass over the counts (sufficient stats n, S1, S2). The materializer
    # never reads ci_low/ci_high here, so the fast path's WALD interval -- instead of the
    # exact Garwood interval's two chi-square-quantile Newton solves -- is the right call:
    # every consumed field (lambda_fail, lambda_mle, p_fail_next, is_poisson, over_dispersed,
    # dispersion, n_windows, gate_ready) is numerically identical to the slow `poisson.fit`.
    fit = poisson.fit_fast(failure_counts)
    ewma = float(fit.extras.get("ewma_lambda", 0.0))
    p_next = poisson.prob_at_least_one(ewma, 1)
    return {
        "lambda_fail": _round(ewma),
        "lambda_mle": _round(fit.lambda_hat),
        "p_fail_next": _round(p_next),
        "is_poisson": bool(fit.is_poisson),
        "over_dispersed": bool(fit.extras.get("over_dispersed", False)),
        "dispersion": _round(fit.dispersion),
        "n_windows": fit.n_windows,
        "gate_ready": bool(fit.gate_ready),
    }


def _materialize_latency(latency_samples: list[float]) -> dict:
    """Fit Gamma(k, theta) to latency samples and read the p50/p95/p99 the V5 auction
    needs (p50 -> cost/occupancy, p95/p99 -> SLA tail penalty). MLE when there is spread,
    MoM warm-start otherwise; degrades cleanly on <2 samples."""
    if len(latency_samples) < 2:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "k": 0.0, "theta": 0.0, "n": len(latency_samples)}
    fit = gamma.fit_latency(latency_samples, method="mle")
    q = fit.sla_quantiles()
    return {
        "p50": _round(q["p50"], 4), "p95": _round(q["p95"], 4), "p99": _round(q["p99"], 4),
        "k": _round(fit.k, 4), "theta": _round(fit.theta, 6), "n": fit.n,
    }


def _materialize_d2mu(mu_history: Sequence[float]) -> dict:
    """Robust learning ACCELERATION = curvature of a local-quadratic fit to the mu trace.
    Significance-gated so a noisy plateau does not false-alarm (bounds false-alarm <=20%)."""
    acc = lam_ctrl.d2_mu_accel(list(mu_history))
    return {
        "d2_mu": _round(acc["d2_mu"], 5),
        "slope": _round(acc["slope"], 5),
        "significant": bool(acc["significant"]),
        "direction": acc["direction"],
    }


def _materialize_fracture(features: Mapping[str, Any] | None) -> float:
    """Single 0..1 fragility score (governance_v6.fracture_score). The roadmap's NET-OUT
    caveat (do not double-count V3 hazard terms) is the hot path's concern; here we just
    materialize the raw, bounded score."""
    if not features:
        return 0.0
    clean = {k: _num(v, 0.0) for k, v in dict(features).items()}
    return _round(gv6.fracture_score(clean), 6)


def _agent_clears_hard_gate(hp: float, mu: float, sigma: float) -> bool:
    """The V2 money/security HARD gate: ALL THREE must hold. Used to decide whether a
    critical lane may remain in an agent's eligible set (fail-closed otherwise)."""
    return hp > _GATE_MIN_HP and mu > _GATE_MIN_MU and sigma < _GATE_MAX_SIGMA


def _lane_name(lane: Any) -> str:
    if isinstance(lane, Mapping):
        return str(lane.get("lane") or lane.get("name") or lane.get("domain") or "")
    return str(lane or "")


def _lane_risk(lane: Any) -> float:
    """Worst-case tail risk of a lane across {money,security,live}; fail-CLOSED (critical)
    for an unlabeled lane so an unclassified lane is treated as money-grade."""
    if not isinstance(lane, Mapping):
        # bare string lane: unknown risk -> fail closed (treat as critical).
        return _CRITICAL_RISK
    dims = [lane.get("money_risk"), lane.get("security_risk"), lane.get("live_risk")]
    vals = [_num(x, -1.0) for x in dims]
    vals = [v for v in vals if v >= 0.0]
    if vals:
        return _clamp(max(vals), 0.0, 5.0)
    for key in ("tail_risk", "risk", "risk_level"):
        if key in lane:
            return _clamp(_num(lane.get(key), _CRITICAL_RISK), 0.0, 5.0)
    return _CRITICAL_RISK  # fail-closed


def _materialize_eligible_lanes(candidate_lanes: Sequence[Any],
                                hp: float, mu: float, sigma: float,
                                frn_barrier: float, p_fail_next: float) -> list[str]:
    """Compute the lanes an agent may serve. FAIL-CLOSED on money/security:

      * A critical lane (risk >= 3) is eligible ONLY IF the agent clears the V2 hard gate.
        The materializer can only REMOVE a critical lane, never grant one (money/security
        never go shadow->full here).
      * A non-critical lane stays eligible unless the agent's fitted forward failure
        probability exceeds the V5 low-confidence failure budget, TIGHTENED in proportion
        to how far the FRN bar has lifted above its base: a degraded fleet (raised bar)
        sheds the riskier lanes first, but a calm fleet keeps low-risk work flowing.

    Deterministic: lanes are de-duplicated by name and returned in sorted order."""
    clears_gate = _agent_clears_hard_gate(hp, mu, sigma)
    # Non-critical budget = the V5 low-confidence budget, scaled DOWN as the fleet degrades.
    # frn_lift in [0,1) measures bar elevation above base over the headroom to the cap.
    frn_lift = _clamp((float(frn_barrier) - _FRN_BASE_BAR) / max(1e-9, 0.98 - _FRN_BASE_BAR),
                      0.0, 1.0)
    fail_budget = max(0.0, _LOW_CONF_FAIL_BUDGET * (1.0 - frn_lift))
    eligible: set[str] = set()
    for lane in candidate_lanes or ():
        name = _lane_name(lane)
        if not name:
            continue
        risk = _lane_risk(lane)
        if risk >= _CRITICAL_RISK:
            # money/security/live-critical: hard gate, fail-closed.
            if clears_gate:
                eligible.add(name)
            continue
        # non-critical: shed if the agent's forward failure prob blows the budget.
        if p_fail_next <= fail_budget:
            eligible.add(name)
    return sorted(eligible)


# ----------------------------------------------------------------------------------
# Fleet-level materialization
# ----------------------------------------------------------------------------------
def _materialize_lambdas(fleet: Mapping[str, Any]) -> dict:
    """Advance the resource shadow-price PI controller ONE warden tick.

    The controller carries state across ticks; the caller threads it via
    fleet['lambda_state']. We reconstruct the controller from that carry-over so the
    materializer stays a pure function of fleet_state (identical state in -> identical
    lambda out -> idempotent)."""
    safe_cap = _num(fleet.get("safe_capacity"), lam_ctrl.DEFAULT_SAFE_CAPACITY) or lam_ctrl.DEFAULT_SAFE_CAPACITY
    active = _num(fleet.get("active_workers"), 0.0)
    ctrl = lam_ctrl.LambdaController(safe_capacity=safe_cap)
    # Restore carried controller state if supplied (pure: no hidden global).
    st = fleet.get("lambda_state") or {}
    if st:
        ctrl.integrator = _num(st.get("integrator"), 0.0)
        ctrl.lam = _num(st.get("lam"), ctrl.lambda_base)
        ctrl.last_error = _num(st.get("last_error"), 0.0)
        try:
            ctrl.n_updates = int(st.get("n_updates", 0) or 0)
        except (TypeError, ValueError):
            ctrl.n_updates = 0
    rho = active / safe_cap
    new_lambda = ctrl.update_rho(rho)
    snap = ctrl.snapshot()
    snap["lambda"] = _round(new_lambda)
    snap["rho"] = _round(rho)
    snap["integrator"] = _round(ctrl.integrator)
    snap["last_error"] = _round(ctrl.last_error)
    return snap


def _materialize_regime(fleet: Mapping[str, Any]) -> dict:
    """Cross-domain co-movement: systematic (common-factor shock) vs idiosyncratic.

    Uses the eigenvalue regime detector when numpy is available and there are enough
    observations; otherwise falls back to the governance_v6 ANOVA between/within share
    (stdlib only). Either way returns systematic_share + an alarm flag. The alarm only
    ever LIFTS the eligibility bar; it never kills (probabilistic signal)."""
    obs = fleet.get("regime_observations") or []
    out = {"systematic_share": 0.0, "alarm": False, "regime": "undetermined",
           "bar_lift": 0.0, "noise_floor_share": 0.0, "method": "none"}
    if not obs:
        return out
    # ANOVA share is always computable (stdlib) and is the documented bridge statistic.
    fd = gv6.factor_decomposition(list(obs))
    anova = _num(fd.get("systematic_share"), 0.0) if "systematic_share" in fd else 0.0
    out["anova_share"] = _round(anova)

    if _HAVE_REGIME:
        try:
            # with_ci=False: the materializer consumes only systematic_share / alarm / regime
            # / bar_lift / noise_floor_share -- never ci_low/ci_high -- so the B-resample
            # bootstrap (each resample a full O(D^3) eigh) on this call path is pure waste.
            r = regime.detect_regime(observations=list(obs), with_ci=False)
            share = r.systematic_share
            if math.isfinite(share):
                out.update({
                    "systematic_share": _round(share),
                    "alarm": bool(r.alarm),
                    "regime": r.regime,
                    "bar_lift": _round(r.bar_lift),
                    "noise_floor_share": _round(r.noise_floor_share),
                    "method": "eigenvalue",
                })
                return out
            # undetermined (degenerate sample): fall through to ANOVA.
        except Exception:
            pass
    # Fallback: ANOVA share with the governance regime thresholds.
    alarm = anova >= float(gp.V6_REGIME["SYSTEMATIC_ALARM_THRESHOLD"])
    watch = anova >= float(gp.V6_REGIME["SYSTEMATIC_WARN_THRESHOLD"])
    out.update({
        "systematic_share": _round(anova),
        "alarm": bool(alarm),
        "regime": "systematic" if alarm else ("watch" if watch else "idiosyncratic"),
        "bar_lift": _round(float(gp.V6_REGIME["BAR_LIFT_KAPPA"]) * anova),
        "method": "anova",
    })
    return out


def _materialize_cvar(agents: Sequence[Mapping[str, Any]], fleet: Mapping[str, Any],
                      lambda_by_agent: Mapping[str, float]) -> dict:
    """Fleet tail-risk: build the per-agent loss legs and price CVaR(alpha) (Expected
    Shortfall, the coherent number), then gate it against the fleet CVaR budget.

    Loss leg per agent = severity * tail_skew where:
        severity  = the agent's forward failure probability (p_fail_next in [0,1]),
                    scaled to the backtest FAILED severity (1.0) so units match.
        tail_skew = exp(money_risk + security_risk + live_risk) over the agent's most
                    critical candidate lane -- the systemic VaR skew the dispatcher uses.
    This makes CVaR the fleet's expected shortfall of *failure-weighted critical exposure*,
    which is exactly what a risk budget must be sized on (CVaR ~ 15x VaR on the real set)."""
    alpha = _num(fleet.get("cvar_alpha"), _DEFAULT_CVAR_ALPHA) or _DEFAULT_CVAR_ALPHA
    limit = _num(fleet.get("cvar_limit"), 0.0)
    positions: list[cvar.Position] = []
    for a in agents:
        aid = str(a.get("agent_id") or "?")
        p_fail = float(lambda_by_agent.get(aid, 0.0))
        # most-critical lane risk dims for this agent
        m = s = l = 0
        for lane in a.get("candidate_lanes") or ():
            if isinstance(lane, Mapping):
                m = max(m, int(_num(lane.get("money_risk"), 0)))
                s = max(s, int(_num(lane.get("security_risk"), 0)))
                l = max(l, int(_num(lane.get("live_risk"), 0)))
        severity = cvar.SEVERITY_FAILED * _clamp(p_fail, 0.0, 1.0)
        # a tiny floor so an all-clean fleet still yields a (near-zero) coherent sample
        severity = max(severity, cvar.SEVERITY_RETRIED * 1e-3)
        positions.append(cvar.Position(pid=aid, severity=severity, tail_skew=0.0,
                                       money_risk=m, security_risk=s, live_risk=l))
    if not positions:
        return {"alpha": alpha, "VaR": 0.0, "CVaR": 0.0, "CVaR_upper": 0.0,
                "limit": limit, "utilisation": 0.0, "ok": True, "n": 0}
    res = cvar.budget_ok(positions, alpha=alpha, limit=limit, conservative=True)
    return {
        "alpha": alpha,
        "VaR": _round(res.VaR, 6),
        "CVaR": _round(res.CVaR, 6),
        "CVaR_upper": _round(res.CVaR_upper, 6),
        "limit": _round(limit, 6),
        "utilisation": _round(res.utilisation, 6),
        "stability_margin": _round(res.stability_margin, 6),
        "derisk_fraction": _round(res.derisk_fraction, 6),
        "ok": bool(res.ok),
        "n": len(positions),
    }


def _materialize_frn_barrier(base_bar: float, regime_snap: Mapping[str, Any],
                             cvar_snap: Mapping[str, Any]) -> dict:
    """The floating eligibility bar (FRN). Rises when the fleet is DEGRADED:
        bar = base_bar + kappa_regime * excess_share + kappa_cvar * max(0, util - 1)
    where excess_share is the systematic share ABOVE the noise floor (Marchenko-Pastur /
    parallel-analysis), clamped at 0. This is the load-bearing fix: the RAW eigenvalue
    share of a short, wide correlation matrix is inflated by sampling noise (it can be 0.6+
    on pure idiosyncratic data), so feeding the raw share would lift the bar on a calm
    fleet. The regime detector judges "systematic" by excess OVER that noise floor and by
    its alarm flag; we drive the lift the same way. A calm, diversified, in-budget fleet
    therefore sits at base_bar; only a genuine common-factor shock and/or a CVaR budget
    over-run lift the bar. Capped at 0.98 so the bar can never demand the impossible (p=1)."""
    sys_share = _num(regime_snap.get("systematic_share"), 0.0)
    floor = _num(regime_snap.get("noise_floor_share"), 0.0)
    # excess co-movement above the pure-noise baseline (>=0); 0 on an idiosyncratic fleet.
    excess = max(0.0, sys_share - floor)
    # only lift on a *genuine* systematic signal: require the detector's alarm OR a
    # meaningful excess over noise. (A sub-floor share is noise, not a shock.)
    effective = excess if (regime_snap.get("alarm") or excess > 0.0) else 0.0
    base = gv6.frn_barrier(base_bar, effective)
    bar = _num(base.get("bar"), base_bar)
    util = _num(cvar_snap.get("utilisation"), 0.0)
    overrun = max(0.0, util - 1.0)
    bar = min(0.98, bar + _FRN_CVAR_KAPPA * overrun)
    return {
        "bar": _round(bar, 4),
        "base": _round(base_bar, 4),
        "systematic_share": _round(sys_share, 4),
        "excess_over_noise": _round(excess, 4),
        "systemic_lift": _round(_num(base.get("systemic_lift"), 0.0), 4),
        "cvar_overrun_lift": _round(_FRN_CVAR_KAPPA * overrun, 4),
        "degraded": bool(regime_snap.get("alarm") or excess > 0.0 or overrun > 0.0),
    }


def _sigma_widen(base_sigma: float, lambda_snap: Mapping[str, Any],
                 regime_alarm: bool) -> tuple[float, float, list[str]]:
    """Probabilistic-signal -> sigma WIDENING (uncertainty inflation) ONLY. Never tightens,
    never moves mu, never cuts hp. This is the materializer's entire credit footprint and
    it respects the doctrine: a non-deterministic signal may at most raise sigma / flag.

    Triggers (each adds a bounded, capped widening):
      * the fitted failure stream is over-dispersed / NON-Poisson (model says: be less
        sure -- clustered failures are under-modelled),
      * the fleet is in a SYSTEMATIC-shock regime (common-factor risk the per-agent
        sigma did not price).
    Returns (new_sigma, widen_applied, reasons)."""
    widen = 0.0
    reasons: list[str] = []
    if lambda_snap.get("over_dispersed") or not lambda_snap.get("is_poisson", True):
        widen += min(_MAX_SIGMA_WIDEN, 1.5)
        reasons.append("non_poisson_failure_stream")
    if regime_alarm:
        widen += min(_MAX_SIGMA_WIDEN, 2.0)
        reasons.append("systematic_regime")
    widen = min(_MAX_SIGMA_WIDEN, widen)
    new_sigma = _clamp(base_sigma + widen, _SIGMA_LO, _SIGMA_HI)
    return new_sigma, _round(new_sigma - base_sigma, 6), reasons


# ----------------------------------------------------------------------------------
# THE public entry point
# ----------------------------------------------------------------------------------
def materialize(fleet_state: Mapping[str, Any] | None) -> dict:
    """Materialize the full slow-loop snapshot from a fleet_state. Pure + deterministic.

    Order of operations (fleet-level context first, because the per-agent eligibility +
    sigma-widening depend on the fleet's FRN barrier and regime):
        1. lambda PI controller tick           -> fleet.lambdas
        2. correlation regime                   -> systematic_share, alarm
        3. per-agent Poisson failure fit        -> lambda_fail (needed by CVaR + lanes)
        4. fleet CVaR on failure-weighted legs  -> fleet.cvar
        5. FRN floating barrier (regime + cvar) -> fleet.frn_barrier
        6. per-agent Gamma latency / d2_mu / fracture / sigma-widen / eligible_lanes
    """
    fleet_state = fleet_state or {}
    agents_in = list(fleet_state.get("agents") or [])
    fleet_in = dict(fleet_state.get("fleet") or {})

    # --- 1. resource shadow-price controller -------------------------------------
    lambdas = _materialize_lambdas(fleet_in)

    # --- 2. correlation regime ---------------------------------------------------
    regime_snap = _materialize_regime(fleet_in)
    regime_alarm = bool(regime_snap.get("alarm"))

    # --- 3. per-agent Poisson fits (first pass; feeds CVaR + lane eligibility) ----
    lambda_features: dict[str, dict] = {}
    p_fail_by_agent: dict[str, float] = {}
    for a in agents_in:
        aid = str(a.get("agent_id") or "?")
        lf = _materialize_lambda_fail(_int_counts(a.get("failure_counts")))
        lambda_features[aid] = lf
        p_fail_by_agent[aid] = lf["p_fail_next"]

    # --- 4. fleet CVaR on failure-weighted critical exposure ---------------------
    cvar_snap = _materialize_cvar(agents_in, fleet_in, p_fail_by_agent)

    # --- 5. FRN floating barrier (degrades up under regime / CVaR stress) --------
    base_bar = _num(fleet_in.get("frn_base_bar"), _FRN_BASE_BAR) or _FRN_BASE_BAR
    frn = _materialize_frn_barrier(base_bar, regime_snap, cvar_snap)
    frn_bar = _num(frn.get("bar"), base_bar)

    # --- 6. per-agent materialization --------------------------------------------
    agent_snaps: dict[str, dict] = {}
    for a in agents_in:
        aid = str(a.get("agent_id") or "?")
        base_mu = _clamp(_num(a.get("mu"), 50.0), _MU_LO, _MU_HI)
        base_sigma = _clamp(_num(a.get("sigma"), 30.0), _SIGMA_LO, _SIGMA_HI)
        hp = _clamp(_num(a.get("hp"), 50.0), 0.0, 100.0)

        lf = lambda_features[aid]
        latency = _materialize_latency(_pos_samples(a.get("latency_samples")))
        d2 = _materialize_d2mu(a.get("mu_history") or [])
        fracture = _materialize_fracture(a.get("fracture_features"))

        # probabilistic-signal sigma widening (uncertainty only; never a kill)
        new_sigma, widen, widen_reasons = _sigma_widen(base_sigma, lf, regime_alarm)

        # eligible lanes (fail-closed on money/security; uses the WIDENED sigma so a
        # less-sure agent is correctly held out of the hard gate)
        eligible = _materialize_eligible_lanes(
            a.get("candidate_lanes") or [], hp, base_mu, new_sigma,
            frn_bar, lf["p_fail_next"],
        )

        agent_snaps[aid] = {
            "mu": _round(base_mu, 6),
            "sigma": _round(new_sigma, 6),
            "hp": _round(hp, 6),
            "lambda_fail": lf["lambda_fail"],
            "latency_q": {"p50": latency["p50"], "p95": latency["p95"], "p99": latency["p99"]},
            "d2_mu": d2["d2_mu"],
            "fracture": fracture,
            "eligible_lanes": eligible,
            "shadow": {
                "lambda_mle": lf["lambda_mle"],
                "p_fail_next": lf["p_fail_next"],
                "is_poisson": lf["is_poisson"],
                "over_dispersed": lf["over_dispersed"],
                "dispersion": lf["dispersion"],
                "gamma_k": latency["k"],
                "gamma_theta": latency["theta"],
                "latency_n": latency["n"],
                "d2_mu_significant": d2["significant"],
                "d2_mu_direction": d2["direction"],
                "mu_slope": d2["slope"],
                "sigma_base": _round(base_sigma, 6),
                "sigma_widen": widen,
                "sigma_widen_reasons": widen_reasons,
                "clears_hard_gate": _agent_clears_hard_gate(hp, base_mu, new_sigma),
                "n_failure_windows": lf["n_windows"],
                "poisson_gate_ready": lf["gate_ready"],
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "agents": agent_snaps,
        "fleet": {
            "lambdas": lambdas,
            "systematic_share": regime_snap["systematic_share"],
            "frn_barrier": frn["bar"],
            "cvar": cvar_snap,
            "regime": regime_snap,
            "frn": frn,
        },
    }


__all__ = ["materialize", "SCHEMA_VERSION"]


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import json
    demo = {
        "agents": [
            {"agent_id": "Risk-R3-1", "domain": "Risk", "mu": 82, "sigma": 12, "hp": 85,
             "failure_counts": [0, 1, 0, 0, 1, 0, 0, 2, 0, 1],
             "latency_samples": [80, 120, 95, 110, 90, 130, 100, 105],
             "mu_history": [70, 73, 76, 78, 80, 81, 82],
             "fracture_features": {"ci_failure": 1, "failed_requeue": 1},
             "candidate_lanes": [{"lane": "Risk", "money_risk": 4}, {"lane": "Reporting", "tail_risk": 1}]},
            {"agent_id": "Docs-R1-1", "domain": "DesignDocs", "mu": 55, "sigma": 28, "hp": 60,
             "failure_counts": [0, 0, 5, 0, 0, 6, 0, 0, 7, 0],  # over-dispersed
             "latency_samples": [200, 50, 600, 90, 300, 80],
             "mu_history": [50, 50.2, 49.8, 50.1, 49.9, 50.0, 50.1],
             "fracture_features": {"cross_domain_deps": 2},
             "candidate_lanes": [{"lane": "DesignDocs", "tail_risk": 1}, {"lane": "Risk", "money_risk": 4}]},
        ],
        "fleet": {"active_workers": 2.3, "safe_capacity": 2.0, "cvar_limit": 0.5},
    }
    snap = materialize(demo)
    print(json.dumps(snap, indent=2, default=str))
