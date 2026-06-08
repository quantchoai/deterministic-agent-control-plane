"""quantchoai-governor — 60-second quickstart.

Wires a toy 3-agent fleet through the REAL quant.governance control loop
(runner.Runner / dispatcher.utility_for) and prints what the Governor actually
decides: who earns each task, who gets blocked from money work, and who is
decaying. Every number below comes from the real engine — no stubs, no Mongo,
no LLM, no network.

    pip install quantchoai-governor mcp
    python -m quant.governance.kit.quickstart
"""
from __future__ import annotations

# ── Step 1: import the real governance modules ────────────────────────────────
from quant.governance.runner import Runner               # core control loop
from quant.governance.dispatcher import utility_for      # auction scorer
from quant.governance import risk_core as rc             # risk analytics

# ── Step 2: a tiny fleet, in arbitrary arrival order ──────────────────────────
# Real fleets aren't pre-sorted by skill. FIFO ("give it to whoever's next")
# routes to the first eligible agent in this order; the auction routes to the
# best one. The gap between them is the Governor earning its keep.
AGENTS = [
    # Cheap doc agent — NO measured track record (n=0), low HP. Listed first,
    # so FIFO keeps handing it work the auction would never give it.
    {"agent_id": "Docs-Cheap", "domain": "DesignDocs", "mu": 50.0, "sigma": 28.0,
     "hp": 55.0, "measured_n": 0, "model_cost": 0.012,
     "failure_counts": [0, 0, 1, 0, 2], "latency_samples": [300, 250, 400],
     "mu_history": [52, 51, 50],
     "candidate_lanes": [{"lane": "DesignDocs", "tail_risk": 1}]},
    # Mid-tier analytics agent — a real but unspectacular track record.
    {"agent_id": "Analytics-Mid", "domain": "Analytics", "mu": 65.0, "sigma": 19.0,
     "hp": 72.0, "measured_n": 3, "model_cost": 0.055,
     "failure_counts": [0, 1, 0, 0, 1], "latency_samples": [160, 175, 150, 180],
     "mu_history": [62, 63, 64, 65],
     "candidate_lanes": [{"lane": "Analytics", "tail_risk": 1},
                         {"lane": "Reporting", "tail_risk": 1}]},
    # Elite risk analyst — clears the V2 money gate (hp>80, mu>75, sigma<15, n>=1).
    {"agent_id": "Risk-Elite", "domain": "Risk", "mu": 82.0, "sigma": 11.0,
     "hp": 88.0, "measured_n": 5, "model_cost": 0.15,
     "failure_counts": [0, 0, 0, 1, 0], "latency_samples": [90, 95, 88, 102, 97],
     "mu_history": [78, 79, 80, 81, 82],
     "candidate_lanes": [{"lane": "Risk", "money_risk": 4},
                         {"lane": "Analytics", "tail_risk": 1}]},
]

# ── Step 3: a tiny event stream (tasks in, outcomes back) ─────────────────────
EVENT_STREAM = [
    {"pending": [
        # Routine analytics work: auction picks Risk-Elite, FIFO settles for Docs-Cheap.
        {"ticket_id": "T001", "lane": "Analytics", "money_risk": 0.0, "value": 60.0,
         "risk_skew": 1.2, "asset_loss_score": 12.0, "has_tests": True},
        # Money-critical Risk work: only the proven elite clears the V2 hard gate.
        {"ticket_id": "T002", "lane": "Risk", "money_risk": 4.0, "value": 450.0,
         "difficulty": 50.0, "has_tests": True},
        # Money-critical DesignDocs work: ONLY Docs-Cheap covers the lane — but it
        # has no track record (n=0), so the money gate refuses it. FAIL-CLOSED:
        # the task is blocked rather than handed to an unproven agent.
        {"ticket_id": "T003", "lane": "DesignDocs", "money_risk": 4.0, "value": 200.0,
         "difficulty": 40.0, "has_tests": True},
    ], "outcomes": []},
    {"pending": [
        {"ticket_id": "T004", "lane": "Analytics", "money_risk": 0.0, "value": 80.0,
         "risk_skew": 1.2, "asset_loss_score": 12.0, "has_tests": True},
    ], "outcomes": [
        {"agent_id": "Risk-Elite", "outcome": "success", "latency": 95.0},
        {"agent_id": "Docs-Cheap", "outcome": "fail", "latency": None},
    ]},
    {"pending": [
        {"ticket_id": "T005", "lane": "Reporting", "money_risk": 1.0, "value": 40.0,
         "risk_skew": 1.2, "asset_loss_score": 12.0, "has_tests": True},
    ], "outcomes": [
        {"agent_id": "Docs-Cheap", "outcome": "fail", "latency": None},
        {"agent_id": "Docs-Cheap", "outcome": "plausibility_fail", "latency": None},
    ]},
]


def run() -> dict:
    return Runner(agents=AGENTS).run_offline(EVENT_STREAM)


def main() -> None:
    r = run()
    a_tot, f_tot = r["auction_utility_total"], r["fifo_utility_total"]
    lift = (a_tot / f_tot - 1.0) * 100.0 if f_tot else 0.0
    blocked = r["hard_gate_blocked_count"]
    decaying = [aid for aid, st in r["final_fleet"].items()
                if st["hazard_rate"] >= 0.20 or st["hp"] < 40]

    print("=" * 64)
    print("  quantchoai-governor — what the Governor decided")
    print("=" * 64)
    print(f"  Auction utility {a_tot:8.1f}   vs FIFO {f_tot:8.1f}    (+{lift:.1f}%, deterministic)")
    print(f"  Money tasks refused to unproven agents : {blocked}   (fail-closed)")
    print(f"  Agents flagged decaying                : {', '.join(decaying) or 'none'}")

    print("\n--- Per-task routing (auction vs the naive FIFO baseline) ---")
    for row in r["shadow_rows"]:
        pick = row["auction_pick"] or "BLOCKED"
        fifo = row["fifo_pick"] or "BLOCKED"
        flag = "  <- fail-closed money gate" if row["hard_gate_blocked"] else (
               "  <- auction routed better" if pick != fifo else "")
        print(f"  {row['ticket']}:  auction={pick:<14} fifo={fifo:<14}{flag}")

    print("\n--- Final fleet state (V1 survival + V2 credit + V3 hazard) ---")
    for aid, st in r["final_fleet"].items():
        g = "  GUILLOTINED" if st["guillotined"] else (
            "  <- decaying" if st["hazard_rate"] >= 0.20 or st["hp"] < 40 else "")
        print(f"  {aid:<14} hp={st['hp']:5.1f}  mu={st['mu']:5.1f}  "
              f"sigma={st['sigma']:5.1f}  hazard={st['hazard_rate']:.3f}{g}")

    print("\n--- Pure-math risk analytics (no Mongo, no network) ---")
    losses = [1.0] * 99 + [7.389, 20.09]
    cv = rc.cvar(losses, alpha=0.95)
    print(f"  CVaR(0.95) on the loss tail : VaR={cv['VaR']:.2f}  CVaR={cv['CVaR']:.2f}")
    conc = rc.concentration_check(
        ["Risk-Elite", "Risk-Elite", "Analytics-Mid", "Risk-Elite"], cap=0.40)
    print(f"  vendor concentration        : {conc['max_vendor']} holds "
          f"{conc['max_frac']*100:.0f}% of critical work  ->  "
          f"{'VIOLATION flagged' if conc['violation'] else 'within cap'}")

    # A single standalone score, the way you'd call it inline:
    res = utility_for(
        {"ticket_id": "X", "target_domain": "Analytics", "money_risk": 0.0,
         "value": 80.0, "risk_skew": 1.2, "asset_loss_score": 12.0},
        {"agent_id": "Analytics-Mid", "domain": "Analytics",
         "mu": 65.0, "sigma": 19.0, "hp": 72.0}, disk_free=50.0)
    print(f"\n  utility_for(one task, one agent) -> utility={res.utility:.3f} "
          f"p_success={res.p_success:.3f} reason={res.reason}")

    print("\n[done — every number above came from the real engine, no stubs]")


if __name__ == "__main__":
    main()
