"""V5/V6 -- O(1) HOT-PATH dispatcher: score_and_pick(snapshot, ticket).

THE hot path of the agent-management control plane. It reads ONLY precomputed
fields off a snapshot the SLOW loop (warden / materializer) already wrote, scores
each candidate agent's expected utility, applies the HARD safety gates, and returns
the single highest-utility ELIGIBLE agent (or a blocked verdict). It is deterministic
and O(1) per (agent, ticket) pair -- NO distribution fitting, NO calculus, NO LLM,
NO Mongo, no host probing.

Where the heavy math lives (NOT here -- see ADVANCED_QUANT_ROADMAP_PRIVATE.md sec 0.4
and the "Architecture" block):
    SLOW LOOP (warden, every few cycles) writes into the snapshot:
        per-agent : mu, sigma, hp, p_success (precomputed sigmoid), model_cost,
                    measured_n (the V2 measured-row evidence), vendor
        per-lane  : frn_barrier (the floating eligibility bar from the factor model),
                    systematic_share, regime ('idiosyncratic'|'amber'|'systematic')
        fleet     : lambda (resource shadow price), disk_penalty, disk_frozen
    HOT PATH (here) reads those O(1) and never recomputes them.

The score mirrors the V5 utility already in dispatcher.utility_for, but every term is a
SNAPSHOT LOOKUP rather than a computation:

    U(i,j) = p_success_ij * value_j                       (precomputed p_success)
             - model_cost_i                               (precomputed)
             - tail_penalty_ij                            (precomputed risk_skew * asset_loss)
             - context_penalty_j                          (precomputed: requeue/traceback drag)
             - resource_penalty                           (lambda + disk; precomputed)

Hard safety gates (ALL deterministic; a probabilistic sensor may NEVER fire these --
roadmap discipline 0.3 / oracle ladder 4A):

    G0  disk sentinel freeze            -> block ALL (snapshot.disk_frozen)
    G1  V2 money/security gate          -> money/security ticket requires an agent with
                                           a MEASURED row (measured_n>=1) AND hp>80 AND
                                           mu>75 AND sigma<15. No measured-credit agent
                                           => BLOCKED (safe). (dispatcher._hard_role_eligible)
    G2  FRN floating-barrier freeze     -> on a frozen lane (high frn_barrier, e.g. vendor-
                                           wide degradation) a HIGH-RISK ticket gets no
                                           assignment: the agent's p_success must clear the
                                           lane's floating bar. (governance_v6.frn_barrier)
    G3  oracle-determinism severity     -> a ticket whose strongest oracle is NON-
                                           deterministic may NOT be routed to an irreversible/
                                           live action; such tickets are gated to "safe"
                                           handling. (governance_v6.oracle_tier)
    G4  confidence floor by risk tier   -> risk>=3 needs p_success>=LOW; risk>=4 needs
                                           p_success>=CRITICAL. (dispatcher gates)
    G5  positive-utility acceptance     -> assign only if best U > MIN_POSITIVE_UTILITY.

Determinism contract: identical (snapshot, ticket) -> identical pick, byte for byte.
Ties are broken by a total, stable order (utility desc, then mu desc, then agent_id asc)
so there is never a set-iteration-order dependence.

Standard library only. Reuses constants from governance_params (single source of truth)
and the gate shapes from dispatcher / governance_v6 WITHOUT importing their I/O paths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from quant.governance import governance_params as gp


# ===========================================================================
# Constants -- anchored 1:1 to governance_params (single source of truth).
# These are the ONLY knobs; everything else is read off the snapshot.
# ===========================================================================
_V2 = gp.V2_CREDIT
_V5 = gp.V5_OPTIMIZATION
_V6M = gp.V6_MONEY_CRITICAL

# V2 money/security hard gate (dispatcher._hard_role_eligible).
GATE_MIN_HP: float = float(_V2["MONEY_SECURITY_GATE_MIN_HP"])      # hp > 80
GATE_MIN_MU: float = float(_V2["MONEY_SECURITY_GATE_MIN_MU"])      # mu > 75
GATE_MAX_SIGMA: float = float(_V2["MONEY_SECURITY_GATE_MAX_SIGMA"])  # sigma < 15
MEASURED_ROW_MIN_N: int = int(_V2["MEASURED_ROW_MIN_N"])          # n >= 1 measured row

# V5 acceptance gates.
MIN_POSITIVE_UTILITY: float = float(_V5["MIN_POSITIVE_UTILITY"])
LOW_CONFIDENCE_MIN_P: float = float(_V5["LOW_CONFIDENCE_MIN_P"])        # risk>=3
CRITICAL_CONFIDENCE_MIN_P: float = float(_V5["CRITICAL_CONFIDENCE_MIN_P"])  # risk>=4

# The explicit-risk threshold at which a ticket is money/security critical and a
# "high-risk" lane action (mirrors V6_MONEY_CRITICAL.EXPLICIT_RISK_CRITICAL_THRESHOLD).
EXPLICIT_RISK_CRITICAL: float = float(_V6M["EXPLICIT_RISK_CRITICAL_THRESHOLD"])  # 3.0
HIGH_RISK_TIER: float = 4.0  # risk>=4 == the strict (critical) confidence band

# Oracle tiers that may drive an IRREVERSIBLE / live action (roadmap 4A).
DETERMINISTIC_ORACLE_TIERS = frozenset({"deterministic"})


# ===========================================================================
# Immutable verdict + snapshot views.
# ===========================================================================
@dataclass(frozen=True)
class Pick:
    """Result of a hot-path scoring round. `agent_id is None` => no assignment (blocked)."""
    agent_id: Optional[str]
    utility: float
    p_success: float
    reason: str
    # full ranked, gate-annotated scoreboard (deterministic order) for audit/telemetry:
    scoreboard: tuple = field(default_factory=tuple)

    @property
    def assigned(self) -> bool:
        return self.agent_id is not None


# ===========================================================================
# Pure helpers -- all O(1), all snapshot reads. No fitting, no calculus.
# ===========================================================================
def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _ticket_risk(ticket: Mapping[str, Any]) -> float:
    """The ticket's explicit risk level = max of the money/security/live risk fields.
    Precomputed-classification field `risk_level` (if the slow loop already set it) wins.
    O(1), deterministic, no regex/NLP in the hot path."""
    if "risk_level" in ticket:
        return _num(ticket.get("risk_level"))
    return max(
        _num(ticket.get("money_risk")),
        _num(ticket.get("security_risk")),
        _num(ticket.get("live_risk")),
    )


def _is_money_security(ticket: Mapping[str, Any]) -> bool:
    """Money/security-critical iff the slow loop flagged it OR an explicit risk field
    clears the critical threshold. Classification (the regex/NLP) is the SLOW loop's job
    (governance_v6 / dispatcher.infer_ticket_features); the hot path only reads the flag."""
    if ticket.get("money_security_critical") is True:
        return True
    return _ticket_risk(ticket) >= EXPLICIT_RISK_CRITICAL


def _requires_deterministic_oracle(ticket: Mapping[str, Any]) -> bool:
    """A ticket that performs an IRREVERSIBLE / live action requires a deterministic oracle
    (roadmap 4A: no auto-commit on a noisy sensor)."""
    return bool(ticket.get("irreversible") or ticket.get("live_action")
                or _num(ticket.get("live_risk")) >= EXPLICIT_RISK_CRITICAL)


# ---- the five precomputed utility terms (each a snapshot lookup) ---------- #
def _value(ticket: Mapping[str, Any]) -> float:
    """Precomputed ticket value (business_impact*urgency*downstream*starvation). The slow
    loop materializes `value`; we read it. A bare fallback keeps the hot path total even if
    the materializer has not run yet -- it does NOT recompute the exp(LAMBDA*age) calculus."""
    if "value" in ticket:
        return max(0.0, _num(ticket.get("value")))
    # Minimal, calculus-free fallback (no exp/log): linear surrogate only.
    bi = _num(ticket.get("business_impact"), 1.0)
    urg = _num(ticket.get("urgency"), 1.0)
    return max(0.0, bi * urg)


def _tail_penalty(agent: Mapping[str, Any], ticket: Mapping[str, Any], p_success: float) -> float:
    """Expected-shortfall-style tail term, mirroring dispatcher's
        risk_penalty = (1 - p_success) * risk_skew * asset_loss
    risk_skew = exp(money+security+live) and asset_loss are BOTH precomputed by the slow
    loop and stored on the ticket; the hot path multiplies, it does not exponentiate."""
    risk_skew = _num(ticket.get("risk_skew"), 1.0)
    asset_loss = _num(ticket.get("asset_loss_score"), 12.0)
    return max(0.0, (1.0 - p_success) * risk_skew * asset_loss)


def _context_penalty(ticket: Mapping[str, Any]) -> float:
    """Precomputed context drag (failed requeues / traceback noise). The slow loop already
    folded ALPHA*requeue + BETA*traceback into `context_penalty`; read it. Fallback uses the
    governance_params weights on raw counts (still O(1), no fitting)."""
    if "context_penalty" in ticket:
        return max(0.0, _num(ticket.get("context_penalty")))
    alpha = float(_V5["ALPHA_REQUEUE_PENALTY"])
    beta = float(_V5["BETA_TRACEBACK_PENALTY"])
    return max(0.0, alpha * _num(ticket.get("failed_requeue_count"))
               + beta * _num(ticket.get("traceback_len")))


def _resource_penalty(snapshot: Mapping[str, Any]) -> float:
    """Fleet-level resource tax = disk_penalty + the resource shadow price lambda (scaled).
    BOTH are precomputed by the slow loop (disk by the sentinel, lambda by the
    LambdaController). The hot path reads and adds; it solves no Lagrangian (roadmap 2B)."""
    fleet = snapshot.get("fleet", {}) if isinstance(snapshot, Mapping) else {}
    disk_pen = _num(fleet.get("disk_penalty"), 0.0)
    lam = _num(fleet.get("lambda"), 0.0)
    # lambda is a price per unit utilization; the hot path charges it as a flat resource
    # toll on each dispatch (the materializer already set its magnitude).
    return max(0.0, disk_pen) + max(0.0, lam)


# ===========================================================================
# Gate evaluation -- returns (eligible, reason). All deterministic.
# ===========================================================================
def _lane_view(snapshot: Mapping[str, Any], ticket: Mapping[str, Any]) -> Mapping[str, Any]:
    lanes = snapshot.get("lanes", {}) if isinstance(snapshot, Mapping) else {}
    lane = str(ticket.get("lane") or ticket.get("domain") or "")
    v = lanes.get(lane, {})
    return v if isinstance(v, Mapping) else {}


def _agent_eligible(agent: Mapping[str, Any], ticket: Mapping[str, Any],
                    lane: Mapping[str, Any], p_success: float,
                    money_security: bool, risk: float,
                    needs_det_oracle: bool, det_oracle: bool) -> tuple[bool, str]:
    """All HARD gates for one candidate. Pure, O(1). Order matters: the most safety-
    critical (money-gate) is checked first so its reason wins for an audit trail."""
    hp = _num(agent.get("hp"))
    mu = _num(agent.get("mu"))
    sigma = _num(agent.get("sigma"), math.inf)
    measured_n = _int(agent.get("measured_n"), 0)

    # G1 -- V2 money/security hard gate (measured row + hp/mu/sigma). NEVER shadow:
    # money/security is exactly the lane the roadmap says never goes shadow->full.
    if money_security:
        if measured_n < MEASURED_ROW_MIN_N:
            return False, "BLOCKED_MONEY_GATE_NO_MEASURED_ROW"
        if not (hp > GATE_MIN_HP and mu > GATE_MIN_MU and sigma < GATE_MAX_SIGMA):
            return False, "BLOCKED_MONEY_GATE_THRESHOLDS"

    # G3 -- oracle-determinism severity: an irreversible/live action requires that the
    # AGENT is cleared for deterministic-oracle work (snapshot flag) AND the ticket has a
    # deterministic oracle. A probabilistic sensor may not authorize the irreversible act.
    if needs_det_oracle:
        if not det_oracle:
            return False, "BLOCKED_NO_DETERMINISTIC_ORACLE"
        if not bool(agent.get("deterministic_cleared", True)):
            return False, "BLOCKED_AGENT_NOT_DET_CLEARED"

    # G2 -- FRN floating barrier: on a frozen / stressed lane the floating bar rises; a
    # HIGH-RISK ticket may only go to an agent whose precomputed p_success clears the bar.
    bar = _num(lane.get("frn_barrier"), 0.0)
    if risk >= HIGH_RISK_TIER or money_security:
        if p_success < bar:
            return False, "BLOCKED_FRN_BARRIER_FREEZE"

    # G4 -- confidence floor by risk tier (strictly increasing with risk).
    if risk >= HIGH_RISK_TIER and p_success < CRITICAL_CONFIDENCE_MIN_P:
        return False, "BLOCKED_CONFIDENCE_FLOOR_CRITICAL"
    if risk >= EXPLICIT_RISK_CRITICAL and p_success < LOW_CONFIDENCE_MIN_P:
        return False, "BLOCKED_CONFIDENCE_FLOOR_LOW"

    return True, "ELIGIBLE"


# ===========================================================================
# THE hot path.
# ===========================================================================
def score_agent(snapshot: Mapping[str, Any], ticket: Mapping[str, Any],
                agent: Mapping[str, Any]) -> float:
    """O(1) expected utility for one (agent, ticket). Pure snapshot arithmetic."""
    p = _num(agent.get("p_success"))
    p = min(0.99, max(0.01, p))  # clamp the precomputed probability defensively
    value = _value(ticket)
    model_cost = max(0.0, _num(agent.get("model_cost"), 0.05))
    return (p * value
            - model_cost
            - _tail_penalty(agent, ticket, p)
            - _context_penalty(ticket)
            - _resource_penalty(snapshot))


def score_and_pick(snapshot: Mapping[str, Any], ticket: Mapping[str, Any]) -> Pick:
    """Pick the highest-utility ELIGIBLE agent for `ticket`, or block.

    snapshot = {
        "agents": [ {agent_id, hp, mu, sigma, measured_n, p_success, model_cost,
                     deterministic_cleared, vendor, ...}, ... ],   # precomputed
        "lanes":  { "<lane>": {frn_barrier, systematic_share, regime}, ... },
        "fleet":  { "lambda", "disk_penalty", "disk_frozen" },
    }
    ticket = {risk fields, value, risk_skew, asset_loss_score, context_penalty, lane, ...}

    Deterministic, O(#agents), no fitting / calculus / LLM / I/O.
    """
    fleet = snapshot.get("fleet", {}) if isinstance(snapshot, Mapping) else {}

    # G0 -- disk sentinel freeze blocks the entire fleet (overrides ambition).
    if bool(fleet.get("disk_frozen")):
        return Pick(None, -math.inf, 0.0, "BLOCKED_DISK_SENTINEL_FREEZE", ())

    agents: Sequence[Mapping[str, Any]] = snapshot.get("agents", []) or []
    lane = _lane_view(snapshot, ticket)
    money_security = _is_money_security(ticket)
    risk = _ticket_risk(ticket)
    needs_det_oracle = _requires_deterministic_oracle(ticket)
    det_oracle = bool(ticket.get("has_deterministic_oracle")
                      or ticket.get("has_tests") or ticket.get("has_invariant")
                      or ticket.get("has_compiler"))

    board: list[tuple] = []
    for agent in agents:
        aid = str(agent.get("agent_id") or "")
        p = min(0.99, max(0.01, _num(agent.get("p_success"))))
        eligible, reason = _agent_eligible(
            agent, ticket, lane, p, money_security, risk, needs_det_oracle, det_oracle)
        u = score_agent(snapshot, ticket, agent) if eligible else -math.inf
        board.append((aid, round(u, 9), round(p, 6), eligible, reason,
                      _num(agent.get("mu"))))

    # Deterministic total order: eligible first, then utility desc, then mu desc, then
    # agent_id asc. This is what guarantees identical snapshot+ticket -> identical pick
    # regardless of input list order or any set/dict iteration order.
    board_sorted = sorted(
        board,
        key=lambda r: (0 if r[3] else 1, -r[1] if r[3] else 0.0, -r[5], r[0]),
    )
    scoreboard = tuple(
        {"agent_id": r[0], "utility": r[1], "p_success": r[2],
         "eligible": r[3], "reason": r[4], "mu": r[5]}
        for r in board_sorted
    )

    # Best eligible.
    for r in board_sorted:
        aid, u, p, eligible, reason, _mu = r
        if not eligible:
            break  # eligibles sort first; first non-eligible means none remain
        if u > MIN_POSITIVE_UTILITY:
            return Pick(aid, u, p, "ASSIGNED", scoreboard)
        # Highest-U eligible still below the acceptance floor -> hold (safe, no negative-
        # value dispatch). The reason names the binding constraint.
        return Pick(None, u, p, "HELD_UTILITY_BELOW_THRESHOLD", scoreboard)

    # No eligible agent at all.
    block_reason = scoreboard[0]["reason"] if scoreboard else "BLOCKED_NO_AGENTS"
    return Pick(None, -math.inf, 0.0, block_reason, scoreboard)


__all__ = [
    "Pick", "score_and_pick", "score_agent",
    "GATE_MIN_HP", "GATE_MIN_MU", "GATE_MAX_SIGMA", "MEASURED_ROW_MIN_N",
    "MIN_POSITIVE_UTILITY", "LOW_CONFIDENCE_MIN_P", "CRITICAL_CONFIDENCE_MIN_P",
    "EXPLICIT_RISK_CRITICAL", "HIGH_RISK_TIER",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke
    snap = {
        "agents": [
            {"agent_id": "A_elite", "hp": 92, "mu": 84, "sigma": 9, "measured_n": 7,
             "p_success": 0.91, "model_cost": 0.15},
            {"agent_id": "B_mid", "hp": 70, "mu": 66, "sigma": 20, "measured_n": 3,
             "p_success": 0.80, "model_cost": 0.055},
            {"agent_id": "C_unproven", "hp": 95, "mu": 90, "sigma": 5, "measured_n": 0,
             "p_success": 0.88, "model_cost": 0.012},
        ],
        "lanes": {"Risk": {"frn_barrier": 0.86, "regime": "idiosyncratic"}},
        "fleet": {"lambda": 0.02, "disk_penalty": 0.0, "disk_frozen": False},
    }
    money_ticket = {"lane": "Risk", "money_risk": 4.0, "value": 500.0,
                    "risk_skew": 7.4, "asset_loss_score": 500.0}
    print("money ticket pick:", score_and_pick(snap, money_ticket).agent_id, "(expect A_elite)")
    cheap = {"lane": "Analytics", "value": 40.0}
    print("cheap ticket pick:", score_and_pick(snap, cheap).agent_id)
