"""runner.py -- the V1-V5 deterministic control-plane runner.

OFFLINE / SHADOW. Deterministic, NO LLM, NO Mongo, NO network, NO RNG.

This module makes the V1-V5 math actually RUN. Everything underneath it
(governance_params, ledger, dispatcher, risk_core) is complete and unit-tested;
this file wires them into a single deterministic control loop:

    tick(events) ->
        1. for each pending ticket:
              auction_pick(snapshot, ticket)   -> choose an agent (V5 auction)
              fifo_pick(snapshot, ticket)      -> the naive baseline
        2. record a comparison row per ticket:
              {ticket, auction_pick, fifo_pick, expected_utility, hard_gate_blocked}
        3. apply V1 survival/HP (ledger event deltas, pure/in-memory) and V3 hazard
           updates on the realized outcomes the event stream carries.

It is SHADOW: it computes, scores, compares, and logs. It never kills a live agent
and never mutates Mongo -- the V1 HP and V3 hazard math run against an in-memory
ledger so the loop is a pure function of (fleet_state, events). The auction's pick
is recorded next to the FIFO pick so the rows are the evidence the offering produces.

------------------------------------------------------------------------------------
DETERMINISM CONTRACT
------------------------------------------------------------------------------------
Identical (events) -> identical picks + identical rows, byte for byte. The runner
reads no clock and uses no RNG; the synthetic generator is a pure function of its seed.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Optional, Sequence

from quant.governance import dispatcher as _dispatcher
from quant.governance import ledger as L
from quant.governance import risk_core as rc


# ===========================================================================
# V1 HP event deltas come from ledger.EVENTS (single source of truth). We map a
# realized outcome -> the canonical ledger event so the in-memory HP walk matches
# the live ledger.
# ===========================================================================
_OUTCOME_EVENT = {
    "success": "ci_qa_pass",              # +5 HP : a clean delivery
    "merged": "pr_merged_review_branch",  # +10 HP : merged work
    "fail": "committee_reject",           # -15 HP : own-work failure (resets warm streak)
    "p0": "p0_online",                    # -50 HP : a production incident
    "qa_missed": "qa_missed_bug",         # -40 HP
    "idle": "idle_decay",                 # -2 HP
    # `plausibility_fail` and any other ledger.EVENTS key may be passed through directly.
}

# V3 hazard: a realized failure is one failure-arrival in the newest window. Successes
# append a 0-failure window.
_FAILURE_OUTCOMES = frozenset({"fail", "p0", "qa_missed", "plausibility_fail", "committee_reject"})

# Critical-risk threshold (V2 money/security hard gate).
_CRITICAL_RISK = 3.0

# Disk-free GB we price the offline auction on (healthy box, so the disk penalty is
# negligible and we isolate the GOVERNANCE decision). Deterministic; no host read.
_OFFLINE_DISK_FREE_GB = 50.0


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


def _ticket_lane(ticket: Mapping[str, Any]) -> str:
    return str(ticket.get("lane") or ticket.get("domain") or ticket.get("target_domain") or "")


def _ticket_risk(ticket: Mapping[str, Any]) -> float:
    if "risk_level" in ticket:
        return _num(ticket.get("risk_level"))
    return max(_num(ticket.get("money_risk")), _num(ticket.get("security_risk")),
               _num(ticket.get("live_risk")))


def _ticket_for_dispatcher(ticket: Mapping[str, Any]) -> dict:
    """Project a runner ticket (lane-keyed) into the dispatcher's ticket shape
    (target_domain-keyed), preserving the explicit risk/value/difficulty fields."""
    t = dict(ticket)
    lane = _ticket_lane(ticket)
    t.setdefault("target_domain", lane)
    t.setdefault("ticket_id", str(ticket.get("ticket_id") or ticket.get("ticket") or "?"))
    return t


# ===========================================================================
# In-memory V1 ledger / V3 hazard state (pure; no Mongo).
# ===========================================================================
@dataclass
class AgentState:
    """The mutable per-agent carry the runner walks across ticks. Mirrors the ledger
    fields the engines read, but lives entirely in memory so the loop is a pure function."""
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

    def eligible_lanes(self) -> set:
        """Lanes this agent may take critical work on (a candidate lane with money_risk>=3
        is a critical lane the agent has declared eligibility for)."""
        lanes = set()
        for cl in self.candidate_lanes:
            if isinstance(cl, dict) and cl.get("lane"):
                lanes.add(str(cl["lane"]))
        return lanes

    def to_dispatcher_agent(self) -> dict:
        """Project into the dispatcher's agent descriptor: lift mu/sigma into a measured
        domain_skills row (n = measured_n) so the V2 hard gate honors the track record."""
        row = {"mu": self.mu, "sigma": self.sigma, "n": self.measured_n}
        skill = _dispatcher._skill_name(self.domain)
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "hp": self.hp,
            "mu": self.mu,
            "sigma": self.sigma,
            "model_tier_cost": self.model_cost,
            "domain_skills": {skill: dict(row), self.domain: dict(row)},
        }


@dataclass(frozen=True)
class ShadowRow:
    """One recorded comparison for a single ticket."""
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
# The V5 auction + the FIFO baseline (what the auction must beat).
# ===========================================================================
def _score_all(states: Sequence[AgentState], ticket: Mapping[str, Any]) -> list[dict]:
    """Score every non-guillotined agent on the V5 auction objective for one ticket.

    Returns a per-agent list of {agent_id, utility, eligible, reason}. Pure: the offline
    disk-free figure is fixed and no DB is touched (features inferred from the ticket alone).
    """
    t = _ticket_for_dispatcher(ticket)
    features = _dispatcher.infer_ticket_features(t, db=None)
    lane = _ticket_lane(ticket)
    risk = _ticket_risk(ticket)
    rows: list[dict] = []
    for st in states:
        if st.guillotined:
            continue
        # Lane eligibility (fail-closed on critical work): an agent serves the lanes it
        # declares in candidate_lanes. A CRITICAL ticket (risk>=3) may only be taken by an
        # agent that has declared the lane -- so an off-lane agent (even a strong one) can
        # never absorb critical work for a lane it does not cover. Non-critical work may
        # fall through to a generalist (an agent with no declared lanes).
        declared = st.eligible_lanes()
        if lane and risk >= _CRITICAL_RISK and lane not in declared:
            rows.append({"agent_id": st.agent_id, "utility": -math.inf, "eligible": False,
                         "reason": "OFF_LANE_FOR_CRITICAL"})
            continue
        if lane and declared and lane not in declared:
            rows.append({"agent_id": st.agent_id, "utility": -math.inf, "eligible": False,
                         "reason": "OFF_LANE"})
            continue
        agent = st.to_dispatcher_agent()
        ur = _dispatcher.utility_for(t, agent, disk_free=_OFFLINE_DISK_FREE_GB, db=None,
                                     features=features)
        rows.append({
            "agent_id": st.agent_id,
            "utility": ur.utility,
            "eligible": math.isfinite(ur.utility),
            "reason": ur.reason,
        })
    return rows


def auction_pick(states: Sequence[AgentState], ticket: Mapping[str, Any]) -> dict:
    """V5 auction: the eligible agent with the maximum expected utility, or a BLOCKED
    verdict when nothing is eligible (fail-closed).

    Returns {agent_id, utility, assigned, reason, scoreboard}.
    """
    scoreboard = _score_all(states, ticket)
    eligible = [r for r in scoreboard if r["eligible"]]
    if not eligible:
        # Surface the dominant blocking reason so the row records WHY nothing won.
        block_reason = scoreboard[0]["reason"] if scoreboard else "NO_ELIGIBLE_AGENT"
        return {"agent_id": None, "utility": -math.inf, "assigned": False,
                "reason": f"BLOCKED:{block_reason}", "scoreboard": scoreboard}
    winner = max(eligible, key=lambda r: r["utility"])
    return {"agent_id": winner["agent_id"], "utility": winner["utility"], "assigned": True,
            "reason": winner["reason"], "scoreboard": scoreboard}


def fifo_pick(states: Sequence[AgentState], ticket: Mapping[str, Any]) -> Optional[str]:
    """The naive baseline: pick the FIRST agent (stable input order) that is not hard-gated,
    ignoring expected utility entirely. This is the 'give it to whoever is next' policy the
    auction is measured against. It still respects the HARD safety gates -- a baseline that
    hands money work to an unqualified agent would be a strictly unsafe straw man, not a fair
    comparison -- so FIFO differs from the auction ONLY in that it does not RANK by utility."""
    for row in _score_all(states, ticket):
        if row["eligible"]:
            return row["agent_id"]
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
                     recompute the hazard via the real ledger.calculate_hazard so the next
                     tick prices on the updated failure stream.
    Warm streak    : a genuine own-work failure resets the streak (ledger.STREAK_RESET_EVENTS).
    """
    # A caller may pass a raw ledger.EVENTS key directly; otherwise map the coarse outcome.
    event = outcome if outcome in L.EVENTS else _OUTCOME_EVENT.get(outcome, "idle_decay")
    delta = float(L.EVENTS.get(event, 0.0))
    state.hp = max(0.0, min(L.MAX_HP, state.hp + delta))
    if state.hp < L.GUILLOTINE_HP:
        state.guillotined = True

    is_failure = (outcome in _FAILURE_OUTCOMES) or (event in _FAILURE_OUTCOMES) or (delta < 0.0)
    state.failure_counts.append(1 if is_failure else 0)
    if len(state.failure_counts) > 60:
        state.failure_counts = state.failure_counts[-60:]

    # V3 hazard via the real ledger model on the in-memory telemetry snapshot.
    streak_failed = state.successful_ticket_streak  # placeholder updated below
    snap = {
        "failed_requeue_count": sum(1 for x in state.failure_counts[-10:] if x),
        "streak_failed": 0,
        "mu_history": list(state.mu_history),
    }
    if is_failure:
        snap["streak_failed"] = 1
    hazard = L.calculate_hazard(snap)
    state.hazard_rate = round(float(hazard["hazard_rate"]), 6)

    # V2 credit drift (Elo-style nudge) so a proven agent earns a measured row + mu lift.
    expected = 0.5
    obs = 0.0 if is_failure else 1.0
    k = max(2.0, min(12.0, state.sigma / 2.7))
    state.mu = max(0.0, min(100.0, state.mu + k * (obs - expected)))
    if is_failure:
        state.sigma = min(60.0, state.sigma * 1.08 + 1.5)
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
# THE RUNNER.
# ===========================================================================
class Runner:
    """Deterministic V1-V5 control-plane runner. Holds the in-memory fleet state and walks
    it tick by tick. SHADOW: it scores + compares + logs; it never kills a live agent."""

    def __init__(self, agents: Sequence[Mapping[str, Any]] | None = None,
                 fleet: Mapping[str, Any] | None = None):
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
        self.fleet: dict = dict(fleet or {})
        self.shadow_rows: list[ShadowRow] = []
        self.tick_index = 0

    # -- the core loop -------------------------------------------------------
    def tick(self, events: Mapping[str, Any] | None = None) -> dict:
        """Run ONE control-plane tick.

        events = {
            "pending":  [ {ticket_id, lane/domain, money_risk/.../risk_level, value, ...}, ... ],
            "outcomes": [ {agent_id, outcome, latency?}, ... ],   # realized last-tick results
        }

        Returns a per-tick report carrying the rows produced this tick and the utility totals
        used to judge auction-vs-FIFO.
        """
        events = events or {}
        self.tick_index += 1
        states = list(self.states.values())

        rows: list[ShadowRow] = []
        auction_util_total = 0.0
        fifo_util_total = 0.0
        for ticket in events.get("pending") or ():
            tid = str(ticket.get("ticket_id") or ticket.get("ticket") or "?")
            pick = auction_pick(states, ticket)
            fifo = fifo_pick(states, ticket)

            hard_blocked = (not pick["assigned"]) and str(pick["reason"]).startswith("BLOCKED")
            reason_components = {
                "verdict": pick["reason"],
                "auction_assigned": pick["assigned"],
                "fifo_assigned": fifo is not None,
                "scoreboard": pick["scoreboard"],
            }
            row = ShadowRow(
                ticket=tid,
                auction_pick=pick["agent_id"],
                fifo_pick=fifo,
                expected_utility=pick["utility"],
                hard_gate_blocked=hard_blocked,
                reason_components=reason_components,
            )
            rows.append(row)
            self.shadow_rows.append(row)

            if pick["assigned"] and math.isfinite(pick["utility"]):
                auction_util_total += pick["utility"]
            if fifo is not None:
                # the FIFO pick's utility for the SAME ticket (the baseline's realized value)
                fifo_score = next((r["utility"] for r in pick["scoreboard"]
                                   if r["agent_id"] == fifo and math.isfinite(r["utility"])), 0.0)
                fifo_util_total += fifo_score
            if pick["assigned"]:
                self.states[pick["agent_id"]].n_assigned += 1

        # --- apply V1 survival/HP + V3 hazard updates on realized outcomes -----
        for oc in events.get("outcomes") or ():
            aid = str(oc.get("agent_id") or "")
            st = self.states.get(aid)
            if st is None or st.guillotined:
                continue
            _apply_outcome(st, str(oc.get("outcome") or "idle"), oc.get("latency"))

        return {
            "tick": self.tick_index,
            "shadow_rows": [r.as_dict() for r in rows],
            "auction_utility_total": round(auction_util_total, 6),
            "fifo_utility_total": round(fifo_util_total, 6),
            "auction_beats_fifo": auction_util_total >= fifo_util_total,
        }

    # -- offline replay ------------------------------------------------------
    def run_offline(self, events_stream: Sequence[Mapping[str, Any]]) -> dict:
        """Replay a SYNTHETIC / recorded event stream (a list of per-tick `events` dicts).
        NO live fleet, NO LLM. Returns the aggregated report + every row, deterministic."""
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
        }


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
        k = 2 + int(next(rng) * 2)  # 2 or 3
        for j in range(k):
            r = next(rng)
            lane = lanes[int(next(rng) * len(lanes))]
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
    runner = Runner(agents if agents is not None else synthetic_fleet(), fleet=fleet)
    return runner.run_offline(events_stream)


# ===========================================================================
# CLI.
# ===========================================================================
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V1-V5 control-plane runner (deterministic, shadow).")
    ap.add_argument("--synthetic", type=int, metavar="N",
                    help="replay an N-tick synthetic event stream and print the report")
    ap.add_argument("--seed", type=int, default=12345, help="synthetic stream seed (default 12345)")
    ap.add_argument("--full", action="store_true", help="print every row (verbose)")
    args = ap.parse_args(argv)

    if args.synthetic is not None:
        stream = synthetic_events(args.synthetic, seed=args.seed)
        report = run_offline(stream)
        if not args.full:
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
