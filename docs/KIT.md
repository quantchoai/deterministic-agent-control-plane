# V1-V5 Governance Kit — Drop-In Deploy Guide

This kit packages the QuantChoAI V1-V5 agent-management control plane as a
self-contained drop-in for any Python project that needs risk-budgeted,
survival-gated, auction-ranked agent dispatch.

---

## 5-Step Portability Recipe (from the whitepaper)

### Step 1 — Define the Event Table (V1)

The event table is your outcome vocabulary: what each agent action is worth in
HP (Health Points). The default table lives in `ledger.EVENTS`. Map your
project's outcomes to one of the canonical events:

| Canonical event           | HP delta | When to fire                               |
|---------------------------|----------|--------------------------------------------|
| `ci_qa_pass`              | +5       | clean delivery, test suite green           |
| `pr_merged_review_branch` | +10      | work merged and reviewed                   |
| `committee_reject`        | -15      | own-work failure, output rejected          |
| `p0_online`               | -50      | production incident                        |
| `qa_missed_bug`           | -40      | QA caught what agent missed                |
| `idle_decay`              | -2       | agent idle a full cycle                    |

Feed your realized outcomes as `{"agent_id": ..., "outcome": "<event_name>",
"latency": <ms>}` dicts in the `outcomes` key of each tick's event dict.

### Step 2 — Define the Outcome Signal (V2)

V2 is the Bayesian mu/sigma credit model. Each agent carries:

- `mu` (float, 0–100): expected success rate on a difficulty-50 task.
- `sigma` (float, 3–60): uncertainty. Wide sigma = not much evidence yet.
- `measured_n` (int): how many measured outcomes back this agent's credit.

Cold-start defaults by agent rank (dispatcher `_default_mu_sigma`):

| Rank | mu   | sigma |
|------|------|-------|
| R3   | 78.0 | 14.0  |
| R2   | 62.0 | 24.0  |
| R1   | 48.0 | 32.0  |

The runner walks mu/sigma forward automatically on each realized outcome via
`_apply_outcome`. You supply the starting values; the kit learns from there.

### Step 3 — Define the Risk Legs (V5)

Each ticket carries up to three risk fields (all floats, 0–5):

- `money_risk`: financial mutation / valuation risk
- `security_risk`: auth / permission / access risk
- `live_risk`: real-money / live-broker execution risk

The auction prices the Expected-Shortfall tail term `CVaR(loss_vector, 0.95)`
where the loss vector has a calm body of 99 unit draws plus `exp(risk_i)` for
each active leg. A ticket with all risks at zero carries a unit tail weight
(benign). Higher risk raises the tail penalty and triggers eligibility gates:

- `risk >= 3`: agent must have p_success >= 0.72
- `risk >= 4`: agent must have p_success >= 0.86
- `risk >= 3` (money/security): agent must have a MEASURED row (n >= 1) with
  hp > 80, mu > 75, sigma < 15 (the V2 hard gate)

### Step 4 — Pick the Actors

Each agent dict must carry at minimum:

```python
{
    "agent_id": str,           # unique key
    "domain": str,             # e.g. "Risk", "Analytics", "DesignDocs"
    "mu": float,               # Bayesian mean skill (0–100)
    "sigma": float,            # uncertainty (3–60)
    "hp": float,               # survival health (0–100; spawn at 50)
    "measured_n": int,         # measured outcome count (0 = cold start)
    "model_cost": float,       # cost tier: 0.15 (elite) / 0.055 (mid) / 0.012 (cheap)
    "failure_counts": list,    # per-window failure arrivals (V3 hazard; [] OK cold start)
    "latency_samples": list,   # latency history in ms ([] OK)
    "mu_history": list,        # trailing mu values ([] OK)
    "candidate_lanes": list,   # [{"lane": "<name>", "money_risk": 0-5}]
}
```

The `candidate_lanes` list tells the runner which lanes this agent is
eligible for. A critical (risk >= 3) ticket may only be taken by an agent that
has declared that lane AND clears the V2 money/security hard gate; an off-lane
agent — even a strong one — can never absorb critical work for a lane it does
not cover (fail-closed).

### Step 5 — Wire In; the Rest Is Unchanged

Once you have the event table, outcome signal, risk legs, and actors, the
entire V1-V5 math is self-contained. Import the runner and call it:

```python
from quant.governance.runner import Runner, synthetic_events

runner = Runner(agents=my_agents)
report = runner.run_offline(my_event_stream)
```

Or use the module-level convenience:

```python
from quant.governance.runner import run_offline
report = run_offline(event_stream, agents=my_agents)
```

No Mongo, no LLM, no network required for the offline/shadow control loop.

---

## Public Modules to Import

| Module | Purpose |
|--------|---------|
| `quant.governance.runner` | **Core runner.** `Runner`, `run_offline`, `synthetic_fleet`, `synthetic_events` |
| `quant.governance.dispatcher` | **Auction.** `utility_for` — scores one (ticket, agent) pair; `infer_ticket_features` |
| `quant.governance.ledger` | **V1 survival + V3 hazard.** `EVENTS`, `DEFAULT_HP`, `MAX_HP`, `GUILLOTINE_HP`, `calculate_hazard` |
| `quant.governance.risk_core` | **Risk primitives.** `cvar`, `concentration_check`, `factor_decomposition`, `value_adjustment`, `poisson_failure`, `lognormal_latency`, `fracture_score`, `frn_barrier`, `gamma_acceleration` |
| `quant.governance.governance_params` | **Parameter registry.** `V1_SURVIVAL`, `V2_CREDIT`, `V3_HAZARD`, `V4_PROPULSION`, `V5_OPTIMIZATION`, `REGIME_DETECTION`, `MONEY_CRITICAL` |
| `quant.governance.model_router` | **Model-tier routing.** Data-driven V1-V5 governed model selection (opus/sonnet/haiku) |

> The hardened CVaR estimator (Rockafellar-Uryasev + Cornish-Fisher), the slow-loop
> snapshot materializer, and the fitted calibration overlay are part of the
> private/commercial layer and are not bundled in this open package.

---

## Config Knobs

All knobs are env-overridable floats/ints. The key ones:

| Env var | Default | Role |
|---------|---------|------|
| `QUANTCHO_AUCTION_LAMBDA` | 0.018 | Starvation lift rate (V5 value = … × exp(λ × age_h)) |
| `QUANTCHO_AUCTION_GAMMA` | 0.025 | σ² cost spread (model_cost × (1 + γ × σ²)) |
| `QUANTCHO_AUCTION_ALPHA` | 3.0 | Context penalty per failed-requeue count |
| `QUANTCHO_AUCTION_BETA` | 0.0015 | Context penalty per traceback character |
| `QUANTCHO_AUCTION_THETA` | 18.0 | Disk penalty magnitude (θ / (free_gb − critical)) |
| `QUANTCHO_AUCTION_MIN_RISK_P` | 0.72 | Min p_success for risk >= 3 tickets |
| `QUANTCHO_AUCTION_CRITICAL_RISK_P` | 0.86 | Min p_success for risk >= 4 tickets |
| `QUANTCHO_AUCTION_CVAR_ALPHA` | 0.95 | Confidence level for Expected Shortfall |
| `QUANTCHO_DISK_YELLOW_GB` | 10.0 | Yellow disk alert threshold |
| `QUANTCHO_DISK_CRITICAL_GB` | 5.0 | Hard dispatch freeze below this free GB |
| `QUANTCHO_MARGINAL_CVAR_ENABLED` | TRUE | Marginal-CVaR tail pricing + vendor-concentration enforcement |
| `QUANTCHO_VENDOR_CONCENTRATION_CAP` | 0.40 | Max fleet share of critical work on one vendor |

See `quant.governance.governance_params` for the full V1-V5 parameter registries.

---

## Quickstart

See `quickstart.py` in this directory for a self-contained, runnable end-to-end example.
