"""v6_runner.py -- the CONTROL-PLANE RUNNER (the operational heart).

PRIVATE / SHADOW + OFFLINE. Deterministic, NO LLM, NO Mongo, NO network, NO RNG.

This is the module that makes the V1-V6 math actually RUN. Everything underneath it
(governance_params, ledger, dispatcher, governance_v6, the eleven modelver_* engines) is
COMPLETE and unit-tested; this file wires them into a single deterministic control loop:

    tick(events) ->
        1. materializer.materialize(fleet_state)          -> ONE precomputed snapshot
        2. for each pending ticket:
              hot_path.score_and_pick(snapshot, ticket)    -> choose an agent (auction)
              fifo_pick(snapshot, ticket)                  -> the naive baseline
        3. record a SHADOW comparison row per ticket:
              {ticket, auction_pick, fifo_pick, expected_utility,
               hard_gate_blocked, reason_components}
        4. apply V1 survival/HP (ledger event deltas, pure/in-memory) and V3 hazard
           updates on the realized outcomes the event stream carries.

It is SHADOW: it computes, scores, compares, and logs. It never kills a live agent and
never mutates Mongo -- the V1 HP and V3 hazard math run against an in-memory ledger so the
loop is a pure function of (fleet_state, events). The auction's pick is recorded next to the
FIFO pick so the shadow rows are the evidence the V6 gate promotes (or rejects) on.

------------------------------------------------------------------------------------
THE SLOW/HOT SPLIT (honoured exactly; see modelver_materializer + modelver_hot_path headers)
------------------------------------------------------------------------------------
* SLOW loop (once per tick): materializer.materialize fits every distribution, advances the
  controllers, prices CVaR/regime, and bakes the result into the snapshot.
* HOT path (per ticket): hot_path.score_and_pick reads the snapshot O(1) and picks.

The materializer snapshot keys `agents` by id and exposes mu/sigma/hp/eligible_lanes/etc.;
the hot path expects an agent LIST with a precomputed `p_success`, `measured_n`, `model_cost`
plus a `lanes` map carrying the FRN barrier and a `fleet.lambda` resource price. The bridge
`_hot_snapshot_for_ticket` adapts the slow snapshot into the hot-path view WITHOUT redoing any
heavy math: `p_success` is the same sigmoid the dispatcher uses (mu/sigma vs ticket difficulty),
the FRN barrier and lambda are read straight off the slow snapshot, and eligibility honours the
materializer's fail-closed `eligible_lanes`.

------------------------------------------------------------------------------------
DETERMINISM CONTRACT
------------------------------------------------------------------------------------
Identical (events) -> identical picks + identical shadow rows, byte for byte. The runner reads
no clock and uses no RNG; the synthetic generator is a pure function of its integer seed.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Optional, Sequence

from quant.governance import governance_params as gp
from quant.governance import modelver_materializer as materializer
from quant.governance import modelver_hot_path as hot_path
from quant.governance import governance_v6 as gv6
from quant.governance import ledger as L
from quant.governance import modelver_poisson_failure as poisson


# ===========================================================================
# Constants (single source of truth = governance_params / the engines).
# ===========================================================================
_V2 = gp.V2_CREDIT
_V5 = gp.V5_OPTIMIZATION
_CRITICAL_RISK = float(gp.V6_MONEY_CRITICAL["EXPLICIT_RISK_CRITICAL_THRESHOLD"])  # 3.0

# V1 HP event deltas come from ledger.EVENTS (single source of truth). We map a realized
# outcome -> the canonical ledger event so the in-memory HP walk matches the live ledger.
_OUTCOME_EVENT = {
    "success": "ci_qa_pass",          # +5 HP : a clean delivery
    "merged": "pr_merged_review_branch",  # +10 HP : merged work
    "fail": "committee_reject",       # -15 HP : own-work failure (also resets warm streak)
    "p0": "p0_online",                # -50 HP : a production incident
    "qa_missed": "qa_missed_bug",     # -40 HP
    "idle": "idle_decay",             # -2 HP
}

# V3 hazard contribution we layer onto an agent's failure-window history on a realized
# failure: one failure arrival in the newest window. Successes append a 0-failure window.
# This is exactly what poisson.hazard_prior / ewma_lambda consume.
_FAILURE_OUTCOMES = frozenset({"fail", "p0", "qa_missed"})


# ===========================================================================
# Pure helpers (deterministic; never raise).
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


def _ticket_difficulty(ticket: Mapping[str, Any]) -> float:
    """The difficulty the p_success sigmoid is measured against. A precomputed `difficulty`
    wins; otherwise we derive it from risk (higher risk == harder) the same way the dispatcher
    grades money/security work up. Bounded [10, 100] like ledger._ticket_difficulty."""
    if "difficulty" in ticket:
        return max(10.0, min(100.0, _num(ticket.get("difficulty"), 45.0)))
    risk = max(_num(ticket.get("money_risk")), _num(ticket.get("security_risk")),
               _num(ticket.get("live_risk")), _num(ticket.get("risk_level")))
    # risk 0 -> ~40 (routine), risk 5 -> ~90 (critical). Linear, clamped.
    return max(10.0, min(100.0, 40.0 + 10.0 * risk))


def _p_success(mu: float, sigma: float, difficulty: float) -> float:
    """The dispatcher's success sigmoid (dispatcher._p_success), reused verbatim so the
    shadow auction prices probability identically to the live engine."""
    z = (mu - difficulty) / max(8.0, sigma)
    p = 1.0 / (1.0 + math.exp(-0.717 * z))
    return max(0.01, min(0.99, p))


def _ticket_lane(ticket: Mapping[str, Any]) -> str:
    return str(ticket.get("lane") or ticket.get("domain") or ticket.get("target_domain") or "")


def _ticket_risk(ticket: Mapping[str, Any]) -> float:
    if "risk_level" in ticket:
        return _num(ticket.get("risk_level"))
    return max(_num(ticket.get("money_risk")), _num(ticket.get("security_risk")),
               _num(ticket.get("live_risk")))


# ===========================================================================
# In-memory V1 ledger / V3 hazard state (pure; no Mongo).
# ===========================================================================
@dataclass
class AgentState:
    """The mutable per-agent carry the runner walks across ticks. Mirrors the ledger fields
    the engines read, but lives entirely in memory so the loop is a pure function."""
    agent_id: str
    domain: str = ""
    mu: float = 50.0
    sigma: float = 30.0
    hp: float = L.DEFAULT_HP
    measured_n: int = 0
    model_cost: float = 0.05
    failure_counts: list = field(default_factory=list)   # per-window failure arrivals (V3)
    latency_samples: list = field(default_factory=list)
    mu_history: list = field(default_factory=list)
    candidate_lanes: list = field(default_factory=list)
    fracture_features: dict = field(default_factory=dict)
    successful_ticket_streak: int = 0
    # running telemetry
    hazard_rate: float = 0.0
    n_assigned: int = 0
    n_outcomes: int = 0
    guillotined: bool = False

    def to_fleet_agent(self) -> dict:
        """Project into the materializer's per-agent input shape."""
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "mu": self.mu,
            "sigma": self.sigma,
            "hp": self.hp,
            "failure_counts": list(self.failure_counts),
            "latency_samples": list(self.latency_samples),
            "mu_history": list(self.mu_history),
            "fracture_features": dict(self.fracture_features),
            "candidate_lanes": list(self.candidate_lanes),
        }


@dataclass(frozen=True)
class ShadowRow:
    """One recorded shadow comparison for a single ticket."""
    ticket: str
    auction_pick: Optional[str]
    fifo_pick: Optional[str]
    expected_utility: float          # the auction pick's expected utility (-inf if blocked)
    hard_gate_blocked: bool
    reason_components: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        # -inf is not JSON-stable; serialise as a string sentinel.
        if d["expected_utility"] == -math.inf:
            d["expected_utility"] = "-inf"
        return d


# ===========================================================================
# The slow -> hot snapshot bridge.
# ===========================================================================
def _build_fleet_state(states: Sequence[AgentState], fleet: Mapping[str, Any]) -> dict:
    return {
        "agents": [s.to_fleet_agent() for s in states if not s.guillotined],
        "fleet": dict(fleet or {}),
    }


def _hot_snapshot_for_ticket(slow: Mapping[str, Any],
                             states_by_id: Mapping[str, AgentState],
                             ticket: Mapping[str, Any]) -> dict:
    """Adapt the SLOW materializer snapshot into the hot_path snapshot view for ONE ticket.

    No heavy math: p_success is the dispatcher sigmoid on the snapshot's mu/sigma vs the
    ticket difficulty; the FRN barrier + lambda are read straight off the slow snapshot;
    eligibility honours the materializer's fail-closed `eligible_lanes`.
    """
    difficulty = _ticket_difficulty(ticket)
    lane = _ticket_lane(ticket)
    agents_slow = slow.get("agents", {}) or {}
    fleet_slow = slow.get("fleet", {}) or {}
    frn_bar = _num(fleet_slow.get("frn_barrier"), 0.0)

    hot_agents: list[dict] = []
    for aid, asnap in agents_slow.items():
        st = states_by_id.get(aid)
        mu = _num(asnap.get("mu"), 50.0)
        sigma = _num(asnap.get("sigma"), 30.0)
        hp = _num(asnap.get("hp"), 50.0)
        p_succ = _p_success(mu, sigma, difficulty)
        eligible_lanes = asnap.get("eligible_lanes") or []
        # Fail-closed: a critical-lane ticket whose lane is NOT in the agent's materialized
        # eligible set is hard-excluded here by zeroing measured_n so G1 blocks it. The
        # materializer already removed critical lanes from agents that miss the V2 gate.
        lane_eligible = (lane in eligible_lanes) if lane else True
        measured_n = st.measured_n if st is not None else 0
        if _ticket_risk(ticket) >= _CRITICAL_RISK and not lane_eligible:
            measured_n = 0  # force the money-gate to block (no measured row visible)
        hot_agents.append({
            "agent_id": aid,
            "hp": hp,
            "mu": mu,
            "sigma": sigma,
            "measured_n": measured_n,
            "p_success": p_succ,
            "model_cost": (st.model_cost if st is not None else 0.05),
            "deterministic_cleared": True,
            "_eligible_lane": lane_eligible,
        })

    return {
        "agents": hot_agents,
        "lanes": {lane: {"frn_barrier": frn_bar}} if lane else {},
        "fleet": {
            "lambda": _num((fleet_slow.get("lambdas") or {}).get("lambda"), 0.0),
            "disk_penalty": _num(fleet_slow.get("disk_penalty"), 0.0),
            "disk_frozen": bool(fleet_slow.get("disk_frozen", False)),
        },
    }


# ===========================================================================
# The FIFO baseline (what the auction must beat).
# ===========================================================================
def fifo_pick(hot_snap: Mapping[str, Any], ticket: Mapping[str, Any]) -> Optional[str]:
    """The naive baseline: pick the FIRST agent (stable input order) that is not hard-gated,
    ignoring expected utility entirely. This is the 'just give it to whoever is next' policy
    the auction is measured against. Deterministic: input order is the agent list order.

    It still respects the HARD safety gates (a baseline that hands money work to an unqualified
    agent would be a strictly unsafe straw man, not a fair comparison) -- so FIFO differs from
    the auction ONLY in that it does not RANK by utility, not in that it ignores safety."""
    agents = hot_snap.get("agents", []) or []
    fleet = hot_snap.get("fleet", {}) or {}
    if bool(fleet.get("disk_frozen")):
        return None
    lane = hot_path._lane_view(hot_snap, ticket)
    money_security = hot_path._is_money_security(ticket)
    risk = hot_path._ticket_risk(ticket)
    needs_det = hot_path._requires_deterministic_oracle(ticket)
    det_oracle = bool(ticket.get("has_deterministic_oracle") or ticket.get("has_tests")
                      or ticket.get("has_invariant") or ticket.get("has_compiler"))
    for agent in agents:
        p = min(0.99, max(0.01, _num(agent.get("p_success"))))
        eligible, _reason = hot_path._agent_eligible(
            agent, ticket, lane, p, money_security, risk, needs_det, det_oracle)
        if eligible:
            return str(agent.get("agent_id"))
    return None


# ===========================================================================
# V1 / V3 in-memory outcome application.
# ===========================================================================
def _apply_outcome(state: AgentState, outcome: str, latency: Optional[float]) -> None:
    """Apply ONE realized outcome to an agent's in-memory V1 HP + V3 hazard state.

    V1 survival/HP : add the canonical ledger event delta (ledger.EVENTS), clamp [0,100],
                     guillotine below ledger.GUILLOTINE_HP. Pure arithmetic -- the live
                     ledger.apply_delta does the same walk against Mongo; we mirror it.
    V3 hazard      : append a failure-arrival window (1 on a failure, 0 on a success) and
                     recompute the EWMA hazard prior (poisson.hazard_prior) so the next
                     tick's materializer sees the updated failure stream.
    Warm streak    : a genuine own-work failure resets the streak (ledger.STREAK_RESET_EVENTS).
    """
    event = _OUTCOME_EVENT.get(outcome, "idle_decay")
    delta = float(L.EVENTS.get(event, 0.0))
    state.hp = max(0.0, min(L.MAX_HP, state.hp + delta))
    if state.hp < L.GUILLOTINE_HP:
        state.guillotined = True

    is_failure = outcome in _FAILURE_OUTCOMES
    state.failure_counts.append(1 if is_failure else 0)
    # keep the window history bounded but long enough for the EWMA + GoF
    if len(state.failure_counts) > 60:
        state.failure_counts = state.failure_counts[-60:]
    state.hazard_rate = round(poisson.hazard_prior(state.failure_counts), 6)

    # V2 credit drift (Elo-style nudge) so a proven agent earns a measured row + mu lift.
    expected = 0.5
    obs = 0.0 if is_failure else 1.0
    k = max(2.0, min(12.0, state.sigma / 2.7))
    state.mu = max(0.0, min(100.0, state.mu + k * (obs - expected)))
    if is_failure:
        state.sigma = min(60.0, state.sigma * 1.08 + 1.5)
        state.successful_ticket_streak = 0
        if event in L.STREAK_RESET_EVENTS:
            state.successful_ticket_streak = 0
    else:
        state.sigma = max(5.0, state.sigma * 0.95)
        state.successful_ticket_streak += 1
    state.measured_n += 1
    state.mu_history.append(round(state.mu, 4))
    if len(state.mu_history) > 30:
        state.mu_history = state.mu_history[-30:]
    if latency is not None and latency > 0:
        state.latency_samples.append(float(latency))
        if len(state.latency_samples) > 60:
            state.latency_samples = state.latency_samples[-60:]
    state.n_outcomes += 1


# ===========================================================================
# Shadow-row persistence: per-model accumulation -> the snapshot the promotion gate reads.
# ===========================================================================
def persist_shadow_rows(rows: Sequence[ShadowRow], snapshot: dict) -> dict:
    """Accumulate shadow comparison rows into the promotion snapshot.

    For EACH shadow row the runner produces, this function derives agreement + calibration
    evidence per model and merges it into a per-model entry in ``snapshot["models"]``.  The
    snapshot is the persistent object the V6 promotion gate (v6_promote.run) reads later.

    Per-model accumulation schema (snapshot["models"][model_id]):
        n                  : total shadow observations accumulated so far.
        agreement          : rolling agreement fraction (auction_pick == fifo_pick, or both
                             unassigned -> unanimous gate decision).
        calibration_error  : rolling mean |expected_utility| / (100 + |expected_utility|);
                             proxy for how well the gate's expected utility is calibrated.
                             Blocked rows (utility == -inf) contribute 1.0 (maximum error).
        info_content       : fraction of rows that are NOT hard-gate-blocked.
        stable             : True when the rolling agreement has not moved by more than 0.05
                             over the last 50 observations.
        _recent_agree_window : internal: last-50 observations for stability check.
        _prev_window_agreement : internal: previous window agreement for drift check.

    The mapping from a ShadowRow to model-level metrics follows the V6 architecture:
      - Every shadow row reflects the ENTIRE governance pipeline for that tick; each model's
        signal contribution is not independently observable at the row level. We therefore
        use the aggregate auction-vs-FIFO comparison as a PROXY for model agreement:
          agreement++  when auction_pick == fifo_pick (both converge on the same agent)
                       OR both unassigned (both blocked -> unanimous gate decision)
          disagreement when picks diverge (a model's signal changed routing)
      - calibration_error = |eu| / (100 + |eu|), normalised to [0, 1] per row.
        Blocked rows (eu == -inf) contribute 1.0 (maximum calibration error).
      - info_content = 1 - blocked_fraction.

    This is DETERMINISTIC: identical (rows, snapshot) -> identical output, byte for byte.
    The snapshot dict is mutated in place AND returned (callers may chain).
    """
    if not rows:
        return snapshot

    models_snap: dict[str, dict] = snapshot.setdefault("models", {})

    # Aggregate row-level stats for this batch (same for all models -- fleet-level evidence).
    n_rows = len(rows)
    n_agree = 0
    n_blocked = 0
    calib_sum = 0.0
    recent_agree: list[int] = []

    for row in rows:
        # Agreement: auction == fifo, or both unassigned.
        if row.auction_pick == row.fifo_pick:
            n_agree += 1
            recent_agree.append(1)
        elif row.auction_pick is None and row.fifo_pick is None:
            n_agree += 1
            recent_agree.append(1)
        else:
            recent_agree.append(0)

        # Calibration proxy.
        eu = row.expected_utility
        if eu != -math.inf and math.isfinite(eu):
            calib_sum += abs(eu) / (100.0 + abs(eu))
        else:
            calib_sum += 1.0

        if row.hard_gate_blocked:
            n_blocked += 1

    agreement_batch = n_agree / n_rows
    calibration_error_batch = calib_sum / n_rows
    info_content_batch = 1.0 - (n_blocked / n_rows)

    # Merge into each registered model's snapshot entry (running weighted average).
    for model_id in list(gv6._MODEL_REGISTRY.keys()):
        entry = models_snap.setdefault(model_id, {
            "n": 0,
            "agreement": 0.0,
            "calibration_error": 1.0,
            "info_content": 1.0,
            "stable": False,
            "_recent_agree_window": [],
            "_prev_window_agreement": None,
        })

        prev_n = int(entry.get("n", 0))
        new_n = prev_n + n_rows

        if prev_n == 0:
            entry["agreement"] = agreement_batch
            entry["calibration_error"] = calibration_error_batch
            entry["info_content"] = info_content_batch
        else:
            w_prev = prev_n / new_n
            w_new = n_rows / new_n
            entry["agreement"] = (
                w_prev * float(entry["agreement"]) + w_new * agreement_batch
            )
            entry["calibration_error"] = (
                w_prev * float(entry["calibration_error"]) + w_new * calibration_error_batch
            )
            entry["info_content"] = (
                w_prev * float(entry["info_content"]) + w_new * info_content_batch
            )

        entry["n"] = new_n

        # Stability: compare last-50 window agreement to the previous last-50 window.
        prev_window: list = list(entry.get("_recent_agree_window") or [])
        merged_window = (prev_window + recent_agree)[-50:]
        entry["_recent_agree_window"] = merged_window

        if len(merged_window) >= 50:
            curr_win_agree = sum(merged_window) / len(merged_window)
            prev_agree = entry.get("_prev_window_agreement")
            if prev_agree is not None:
                drift = abs(curr_win_agree - float(prev_agree))
                entry["stable"] = drift <= 0.05
            else:
                entry["stable"] = False
            entry["_prev_window_agreement"] = curr_win_agree
        else:
            entry["stable"] = False

    return snapshot


# ===========================================================================
# THE RUNNER.
# ===========================================================================
class V6Runner:
    """Deterministic control-plane runner. Holds the in-memory fleet state and walks it
    tick by tick. SHADOW: it scores + compares + logs; it never kills a live agent."""

    def __init__(self, agents: Sequence[Mapping[str, Any]] | None = None,
                 fleet: Mapping[str, Any] | None = None,
                 promotion_snapshot: dict | None = None):
        self.states: dict[str, AgentState] = {}
        for a in agents or ():
            st = AgentState(
                agent_id=str(a.get("agent_id")),
                domain=str(a.get("domain", "")),
                mu=_num(a.get("mu"), 50.0),
                sigma=_num(a.get("sigma"), 30.0),
                hp=_num(a.get("hp"), L.DEFAULT_HP),
                measured_n=_int(a.get("measured_n"), 0),
                model_cost=_num(a.get("model_cost"), 0.05),
                failure_counts=list(a.get("failure_counts") or []),
                latency_samples=list(a.get("latency_samples") or []),
                mu_history=list(a.get("mu_history") or []),
                candidate_lanes=list(a.get("candidate_lanes") or []),
                fracture_features=dict(a.get("fracture_features") or {}),
            )
            self.states[st.agent_id] = st
        # carried fleet context (lambda controller state threads across ticks)
        self.fleet: dict = dict(fleet or {})
        self.shadow_rows: list[ShadowRow] = []
        self.tick_index = 0
        # persistent promotion snapshot: accumulated shadow metrics for the gate.
        # If the caller supplies an existing snapshot we use it DIRECTLY (no copy) so
        # the caller's reference sees accumulation in real time. When None, we create a
        # fresh empty dict. Either way, self.promotion_snapshot is the live accumulator.
        self.promotion_snapshot: dict = promotion_snapshot if promotion_snapshot is not None else {}

    # -- the core loop -------------------------------------------------------
    def tick(self, events: Mapping[str, Any] | None = None) -> dict:
        """Run ONE control-plane tick.

        events = {
            "pending":  [ {ticket_id, lane/domain, money_risk/.../risk_level, value, ...}, ... ],
            "outcomes": [ {agent_id, outcome, latency?}, ... ],   # realized last-tick results
        }

        Returns a per-tick report carrying the snapshot headline, the shadow rows produced
        this tick, and the utility totals used to judge auction-vs-FIFO.
        """
        events = events or {}
        self.tick_index += 1

        # --- (1) SLOW loop: build the snapshot from the live in-memory fleet ----
        fleet_state = _build_fleet_state(list(self.states.values()), self.fleet)
        slow = materializer.materialize(fleet_state)
        # thread the lambda controller state forward (idempotent, pure carry-over)
        self.fleet["lambda_state"] = dict(slow["fleet"]["lambdas"])

        # --- (2)+(3) HOT path per pending ticket + record shadow rows ----------
        rows: list[ShadowRow] = []
        auction_util_total = 0.0
        fifo_util_total = 0.0
        for ticket in events.get("pending") or ():
            tid = str(ticket.get("ticket_id") or ticket.get("ticket") or "?")
            hot_snap = _hot_snapshot_for_ticket(slow, self.states, ticket)

            pick = hot_path.score_and_pick(hot_snap, ticket)
            fifo = fifo_pick(hot_snap, ticket)

            hard_blocked = (not pick.assigned) and pick.reason.startswith("BLOCKED")
            # reason_components: the gate verdict + the binding-constraint breakdown.
            reason_components = {
                "verdict": pick.reason,
                "auction_assigned": pick.assigned,
                "fifo_assigned": fifo is not None,
                "scoreboard": [
                    {"agent_id": r["agent_id"], "utility": r["utility"],
                     "eligible": r["eligible"], "reason": r["reason"]}
                    for r in pick.scoreboard
                ],
            }
            row = ShadowRow(
                ticket=tid,
                auction_pick=pick.agent_id,
                fifo_pick=fifo,
                expected_utility=pick.utility,
                hard_gate_blocked=hard_blocked,
                reason_components=reason_components,
            )
            rows.append(row)
            self.shadow_rows.append(row)

            # utility totals (only finite, assigned utilities count toward the comparison)
            if pick.assigned and math.isfinite(pick.utility):
                auction_util_total += pick.utility
            if fifo is not None:
                fifo_util_total += hot_path.score_agent(
                    hot_snap, ticket, _agent_by_id(hot_snap, fifo))
            if pick.assigned:
                self.states[pick.agent_id].n_assigned += 1

        # --- (4) apply V1 survival/HP + V3 hazard updates on realized outcomes --
        for oc in events.get("outcomes") or ():
            aid = str(oc.get("agent_id") or "")
            st = self.states.get(aid)
            if st is None or st.guillotined:
                continue
            _apply_outcome(st, str(oc.get("outcome") or "idle"),
                           oc.get("latency"))

        # --- (5) persist shadow rows into the promotion snapshot ---------------
        # This is the fix for the dead-promotion-pipeline bug: without this step, the
        # promotion gate's snapshot never accumulates evidence and every model stays at n=0.
        if rows:
            persist_shadow_rows(rows, self.promotion_snapshot)

        return {
            "tick": self.tick_index,
            "snapshot_fleet": {
                "frn_barrier": slow["fleet"]["frn_barrier"],
                "systematic_share": slow["fleet"]["systematic_share"],
                "lambda": slow["fleet"]["lambdas"].get("lambda"),
                "cvar": slow["fleet"]["cvar"].get("CVaR"),
                "cvar_ok": slow["fleet"]["cvar"].get("ok"),
            },
            "shadow_rows": [r.as_dict() for r in rows],
            "auction_utility_total": round(auction_util_total, 6),
            "fifo_utility_total": round(fifo_util_total, 6),
            "auction_beats_fifo": auction_util_total >= fifo_util_total,
        }

    # -- offline replay ------------------------------------------------------
    def run_offline(self, events_stream: Sequence[Mapping[str, Any]]) -> dict:
        """Replay a SYNTHETIC / recorded event stream (a list of per-tick `events` dicts).
        NO live fleet, NO LLM. Returns the aggregated report + every shadow row, deterministic.

        The returned dict now includes `promotion_snapshot`: the accumulated per-model evidence
        the V6 promotion gate reads.  This is the fix for the dead promotion pipeline.
        """
        tick_reports = [self.tick(ev) for ev in events_stream]
        auction_total = sum(t["auction_utility_total"] for t in tick_reports)
        fifo_total = sum(t["fifo_utility_total"] for t in tick_reports)
        n_blocked = sum(1 for r in self.shadow_rows if r.hard_gate_blocked)
        return {
            "ticks": len(tick_reports),
            "shadow_rows": [r.as_dict() for r in self.shadow_rows],
            "auction_utility_total": round(auction_total, 6),
            "fifo_utility_total": round(fifo_total, 6),
            "auction_beats_fifo": auction_total >= fifo_total,
            "hard_gate_blocked_count": n_blocked,
            "final_fleet": {
                aid: {"hp": round(s.hp, 3), "mu": round(s.mu, 3), "sigma": round(s.sigma, 3),
                      "hazard_rate": s.hazard_rate, "measured_n": s.measured_n,
                      "n_assigned": s.n_assigned, "guillotined": s.guillotined}
                for aid, s in self.states.items()
            },
            "tick_reports": tick_reports,
            # The accumulated promotion evidence snapshot (shadow->canary evidence base).
            "promotion_snapshot": dict(self.promotion_snapshot),
        }


def _agent_by_id(hot_snap: Mapping[str, Any], aid: str) -> dict:
    for a in hot_snap.get("agents", []) or ():
        if str(a.get("agent_id")) == aid:
            return a
    return {"agent_id": aid, "p_success": 0.0}


# ===========================================================================
# Synthetic event-stream generator (pure function of a seed; NO RNG module).
# ===========================================================================
def _lcg(seed: int):
    """A tiny deterministic linear-congruential generator (Numerical Recipes constants).
    Pure + reproducible; avoids the `random` module so determinism is airtight."""
    state = seed & 0xFFFFFFFF
    while True:
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        yield state / 0xFFFFFFFF


def synthetic_fleet() -> list[dict]:
    """A small, fixed synthetic fleet: one elite (clears the money gate), one mid, one weak."""
    return [
        {"agent_id": "Risk-R3-1", "domain": "Risk", "mu": 84, "sigma": 9, "hp": 92,
         "measured_n": 7, "model_cost": 0.15,
         "failure_counts": [0, 0, 1, 0, 0, 0, 1, 0],
         "latency_samples": [80, 120, 95, 110, 90, 130, 100, 105],
         "mu_history": [78, 80, 81, 82, 83, 84],
         "candidate_lanes": [{"lane": "Risk", "money_risk": 4},
                             {"lane": "Reporting", "tail_risk": 1}]},
        {"agent_id": "Analytics-R2-1", "domain": "Analytics", "mu": 66, "sigma": 20, "hp": 70,
         "measured_n": 3, "model_cost": 0.055,
         "failure_counts": [0, 1, 0, 1, 0, 0, 1, 0],
         "latency_samples": [200, 150, 180, 160, 210, 170],
         "mu_history": [60, 62, 64, 65, 66, 66],
         "candidate_lanes": [{"lane": "Analytics", "tail_risk": 1},
                             {"lane": "Reporting", "tail_risk": 1}]},
        {"agent_id": "Docs-R1-1", "domain": "DesignDocs", "mu": 48, "sigma": 32, "hp": 55,
         "measured_n": 0, "model_cost": 0.012,
         "failure_counts": [0, 2, 0, 3, 0, 1, 0, 2],
         "latency_samples": [300, 90, 500, 80, 250],
         "mu_history": [52, 51, 50, 49, 48, 48],
         "candidate_lanes": [{"lane": "DesignDocs", "tail_risk": 1}]},
    ]


def synthetic_events(n: int, seed: int = 12345) -> list[dict]:
    """Generate a deterministic stream of `n` ticks. Each tick has a few pending tickets and
    a few realized outcomes for the previously-assigned agents. Pure function of (n, seed)."""
    rng = _lcg(seed)
    fleet_ids = [a["agent_id"] for a in synthetic_fleet()]
    lanes = ["Risk", "Analytics", "Reporting", "DesignDocs"]
    stream: list[dict] = []
    for t in range(n):
        pending = []
        # 2-3 tickets per tick, deterministically chosen.
        k = 2 + int(next(rng) * 2)  # 2 or 3
        for j in range(k):
            r = next(rng)
            lane = lanes[int(next(rng) * len(lanes))]
            # ~30% money/security critical (risk 4), else routine (risk 0-1)
            if r < 0.30:
                money_risk = 4.0
                value = 400.0 + 200.0 * next(rng)
                risk_skew = 7.4
                asset_loss = 500.0
                lane = "Risk"  # critical work routes to the Risk lane
            else:
                money_risk = float(int(next(rng) * 2))  # 0 or 1
                value = 30.0 + 60.0 * next(rng)
                risk_skew = 1.2
                asset_loss = 12.0
            pending.append({
                "ticket_id": f"T{t:03d}-{j}",
                "lane": lane,
                "money_risk": money_risk,
                "value": round(value, 2),
                "risk_skew": risk_skew,
                "asset_loss_score": asset_loss,
                "has_tests": True,  # deterministic oracle available
            })
        outcomes = []
        # realized outcomes for the fleet: mostly success, occasional failure (deterministic)
        for aid in fleet_ids:
            r = next(rng)
            if r < 0.15:
                outcome = "fail"
            elif r < 0.20:
                outcome = "idle"
            else:
                outcome = "success"
            outcomes.append({"agent_id": aid, "outcome": outcome,
                             "latency": round(80.0 + 200.0 * next(rng), 1)})
        stream.append({"pending": pending, "outcomes": outcomes})
    return stream


def run_offline(events_stream: Sequence[Mapping[str, Any]],
                agents: Sequence[Mapping[str, Any]] | None = None,
                fleet: Mapping[str, Any] | None = None) -> dict:
    """Module-level convenience: build a runner over `agents` (default the synthetic fleet)
    and replay `events_stream`. NO live fleet, NO LLM. Deterministic."""
    runner = V6Runner(agents if agents is not None else synthetic_fleet(), fleet=fleet)
    return runner.run_offline(events_stream)


# ===========================================================================
# CLI.
# ===========================================================================
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V6 control-plane runner (deterministic, shadow).")
    ap.add_argument("--synthetic", type=int, metavar="N",
                    help="replay an N-tick synthetic event stream and print the report")
    ap.add_argument("--seed", type=int, default=12345, help="synthetic stream seed (default 12345)")
    ap.add_argument("--full", action="store_true", help="print every shadow row (verbose)")
    args = ap.parse_args(argv)

    if args.synthetic is not None:
        stream = synthetic_events(args.synthetic, seed=args.seed)
        report = run_offline(stream)
        if not args.full:
            # compact: drop the per-row scoreboards for readability
            slim = dict(report)
            slim["shadow_rows"] = [
                {k: v for k, v in r.items() if k != "reason_components"}
                for r in report["shadow_rows"]
            ]
            slim.pop("tick_reports", None)
            print(json.dumps(slim, indent=2, default=str))
        else:
            print(json.dumps(report, indent=2, default=str))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
