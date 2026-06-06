# quantchoai-governor

**Deterministic, auditable governance for your agent fleet — no LLM in the routing loop.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-native-7C3AED.svg)](https://modelcontextprotocol.io)
[![Telemetry: none](https://img.shields.io/badge/telemetry-none-success.svg)](#privacy)

> Orchestrators run your agents. **The Governor decides which one earns the task, which one is degrading, and which one may touch money** — with quantitative-finance risk math, deterministically, writing every decision to an auditable ledger row.

It borrows the machinery institutions use to govern *traders* — Bayesian skill credit, coherent tail-risk (CVaR), survival/hazard analysis, and a risk-budgeted auction — and applies it to governing *agents*. No model sits in the routing loop, so every decision is reproducible and explainable after the fact.

---

## 60-second proof

```bash
pip install quantchoai-governor mcp
python -m quant.governance.kit.quickstart      # no API key, no network, no database
```

```text
  Auction utility    563.3   vs FIFO    476.3    (+18.3%, deterministic)
  Money tasks refused to unproven agents : 1   (fail-closed)
  Agents flagged decaying                : Docs-Cheap

--- Per-task routing (auction vs the naive FIFO baseline) ---
  T001:  auction=Risk-Elite   fifo=Docs-Cheap     <- auction routed better
  T002:  auction=Risk-Elite   fifo=Risk-Elite
  T003:  auction=BLOCKED      fifo=BLOCKED         <- fail-closed money gate
  T004:  auction=Risk-Elite   fifo=Docs-Cheap      <- auction routed better

--- Final fleet state (V1 survival + V2 credit + V3 hazard) ---
  Docs-Cheap     hp=23.0  mu=44.9  sigma=34.0  hazard=0.242   <- decaying, flagged before it fails
  Analytics-Mid  hp=72.0  mu=65.0  sigma=19.0  hazard=0.000
  Risk-Elite     hp=93.0  mu=84.0  sigma=10.4  hazard=0.057
```

Real engine, real numbers, no stubs. The auction routes **+18% more utility** than naive FIFO, **refuses** the money task to the unproven agent, and **flags the decaying agent before it fails** — and the whole run is deterministic: run it twice, get the same answer.

---

## Why

LLM-orchestrated fleets break in five predictable ways. The Governor closes each one:

| Failure mode | The Governor's answer |
|---|---|
| **Cost compounds** — a meta-LLM picks the next LLM, every hop | A pure-math auction, **zero model calls** to route |
| **Non-determinism** — identical tasks route differently; failures won't reproduce | Deterministic objective — same inputs, same decision, every time |
| **Unauditable** — "the model decided" is not an explanation | Every decision is a structured **ledger row**, not an attention matrix |
| **No retirement gate** — a degrading agent causes the incident | A **survival ledger + hazard rate** retire an agent *before* it fails |
| **Vendor concentration** — all critical work on one provider | A concentration check **flags the hidden single-point-of-failure** |

---

## The V1–V6 control plane

Six layers, each a small, well-understood piece of math:

| | Layer | What it governs | The math |
|---|---|---|---|
| **V1** | **Survival** | a health reservoir + absolute-HP guillotine line | event-summed ledger |
| **V2** | **Credit** | a Bayesian skill posterior + the *money gate* (proven skill **and** low uncertainty before touching money) | μ / σ posterior, fail-closed gate |
| **V3** | **Hazard** | failure risk *before* it completes | Poisson base rate + acceleration |
| **V4** | **Propulsion** | reward rising agents, not just safe ones | momentum term |
| **V5** | **Auction** | who earns the task, under a risk budget | `U = p·value − cost − CVaR_tail − …` |
| **V6** | **Promotion** | shadow → canary → full, with one immutable invariant | calibration gate |

> **The invariant that never bends:** a probabilistic model can *advise* but can **never** earn deterministic-kill authority. High hazard alone never auto-kills — a deterministic oracle must back the verdict, or it routes to a human/committee.

---

## The six MCP tools

Exposed to any MCP client — Claude Desktop, Claude Code, Cursor, Cline, or any SDK caller:

| Tool | What it decides |
|---|---|
| **`govern_route`** | Auction a task across N agents on the risk-adjusted utility objective; return the winner **and** the full score table. |
| **`record_outcome`** | Apply one observed outcome; update Bayesian credit (μ/σ), the HP ledger, and the Poisson-hazard rate. You own persistence. |
| **`survival_check`** | Verdict for one agent: `SURVIVE` / `SOFT_RECYCLE` / `HOLD` (high hazard → committee) / `GUILLOTINE`. |
| **`model_route`** | Pick the model tier (opus/sonnet/haiku) via the same auction; enforce the money/security hard-gate (`money_risk ≥ 3 → opus only`). |
| **`fleet_snapshot`** | HP / hazard distribution, survival-verdict counts, at-risk list, vendor-concentration flag. |
| **`deployment_info`** | Report whether you're on the open local baseline or a hosted deployment. |

Every tool is **stateless at call time** — prior state arrives as plain JSON, so they unit-test deterministically, offline.

---

## Wire it into your fleet (5 steps)

1. **Hold state per agent** — `{id, vendor, hp, mu, sigma, n}`. Plain JSON in your store.
2. **Before dispatch → `govern_route`** (or `model_route`). Send the task + candidates; act on the returned winner.
3. **After each task → `record_outcome`.** Feed back success/failure; persist the returned new state.
4. **On a timer → `survival_check` + `fleet_snapshot`.** Retire `GUILLOTINE` agents; route `HOLD` to a human.
5. **Audit anytime.** Every decision is a ledger row — replay it, diff it, explain it.

Prefer no transport? Call the logic directly:

```python
from quant.governance.mcp.server import govern_route_logic, survival_check_logic

win = govern_route_logic(
    task={"ticket_id": "T1", "target_domain": "Analytics", "money_risk": 0.0, "value": 60.0},
    agents=[{"agent_id": "a", "mu": 72.0, "sigma": 18.0, "hp": 80.0},
            {"agent_id": "b", "mu": 55.0, "sigma": 28.0, "hp": 60.0}],
)
print(win["winner"])                                                # -> 'a'  (higher-credit, deterministic)

# Survival verdicts escalate with concern — but a hard kill is never automatic:
print(survival_check_logic({"agent_id": "x", "hp": 90})["verdict"])  # -> 'SURVIVE'
print(survival_check_logic({"agent_id": "x", "hp": 22})["verdict"])  # -> 'SOFT_RECYCLE'
print(survival_check_logic({"agent_id": "x", "hp": 12})["verdict"])  # -> 'HOLD'  (low HP alone routes to a human, never an auto-kill)
```

Full MCP-client config (local stdio + remote/hosted) is in [`examples/`](examples/).

---

## Where it fits

The Governor is **not** an orchestrator and doesn't replace one. Frameworks like LangGraph, CrewAI, or a Claude-Code-style setup **run** your agents; the Governor sits underneath and decides **who's trusted, who's decaying, and who may touch money**. It's the risk desk for your fleet — not the trading floor.

---

## Open core, honestly

This repository is the **complete, working open baseline** — the real V1–V6 math, MIT-licensed, no crippled stubs. It runs fully on its own.

A **hosted deployment** can add a server-side overlay (fitted calibration + a vendor-correlation tail refinement) that sharpens the same decisions on real outcome history. That overlay is **not** in this repo and is **never shipped to a client** — the open server exposes only the integration seam, and `deployment_info` reports `local-open-baseline`. What you read here is exactly what runs; the proprietary fit stays server-side. That is the entire boundary.

<a name="privacy"></a>
**Privacy.** This package phones home to **no one**. No telemetry, no analytics, no network call you didn't make. Run it air-gapped.

---

## The math

The objective function, the coherent CVaR derivation (Rockafellar–Uryasev / Cornish–Fisher), the Poisson–Gamma conjugacy, the controller stability proof, and the calibration/promotion protocol are written up in [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) — constants as **symbols**, not fitted values.

---

## Cite · License

MIT — free to use, modify, redistribute. If it helps your work or research, a citation is appreciated: [`CITATION.cff`](CITATION.cff).

Built by **QuantChoAI**. This is the canonical reference implementation of the V1–V6 deterministic agent-governance framework.


<!-- mcp-name: io.github.quantchoai/quantchoai-governor -->
