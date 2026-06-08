# Why your agent fleet needs deterministic governance

*Published by the QuantChoAI team · June 2026*

*Caveat: all validation numbers cited below were produced under shadow/replay
conditions. This product is pre-revenue. We are not claiming production-scale
deployment figures we do not have.*

---

## The routing tax you are paying

Every multi-agent system in production today has a hidden cost line that does
not appear in your LLM spend dashboard: the **meta-routing call**.

When you use an LLM to decide which agent or model should handle the next task,
you are paying:

- One extra generation per task for the routing decision itself
- Non-determinism: the same task routes differently on different runs, which
  means you cannot reproduce a failure to diagnose it
- Zero audit trail: when a risk-sensitive task routes to the wrong agent and
  produces a bad outcome, the decision record is an attention weight — not a
  structured log entry you can query, diff, or hand to a compliance reviewer

For a small fleet doing low-stakes tasks, this is tolerable. For a fleet
handling tasks with financial, legal, or security exposure — or for any team
that has to explain decisions after the fact — it is not.

## The v1-v5 governance model

`quantchoai-governor` is our answer to this. It is a governance engine, not an
orchestrator. It does not run agents. It does not call APIs. It answers one
question, deterministically, for every task: **which agent is the
utility-maximising choice, given current fleet state and task risk profile?**

The engine is structured as five layers, each solving a distinct failure mode:

**V1 — HP ledger.** Every agent carries an absolute health-point balance. Each
outcome (success, partial, fail, critical fail) applies a deterministic delta.
When HP falls below a threshold, the agent moves toward retirement. This
replaces "the orchestrator stopped giving it tasks" with a structured,
observable state machine.

**V2 — Bayesian skill credit.** Each agent maintains a (mu, sigma) Gaussian
belief over its success probability in each domain. Outcomes are Bayesian
updates, not running averages. The uncertainty term is priced into the routing
auction, so an untested agent is not mis-priced as zero-probability or as
equivalent to a proven one.

**V3 — Poisson-hazard survival rate.** HP and Bayesian credit are
backward-looking. V3 adds a forward-looking hazard rate: a Poisson-process
estimate of mean-time-to-next-failure, fitted on the agent's recent outcome
history. High hazard alone does not kill an agent — see the severity-matched
authority invariant below — but it does reduce its probability of winning
auctions and raises the survival-check flag.

**V4 — Warm-propulsion.** Symmetric scoring ignores momentum. V4 rewards agents
whose skill trajectory is accelerating upward (the positive mirror of V3's
collapse-acceleration signal), so capital concentrates on *rising* agents earlier
than a level-only policy would. The money/security **hard gate** — for tasks with
a money-risk or security-risk flag above a defined threshold, eligibility is locked
to a proven, low-uncertainty agent (and the highest model tier) regardless of
auction outcome — is an invariant enforced before the auction runs.

**V5 — Risk-adjusted utility auction.** The routing decision is the argmax of:

```
U = p_success × value
  − model_cost
  − CVaR_tail_term
  − context_penalty
  − disk_penalty
  + starvation_lift
```

The CVaR tail term is coherent Expected Shortfall under the
Rockafellar-Uryasev / Cornish-Fisher expansion, computed over the task's money,
security, and live-execution risk legs jointly. This is not a hand-tuned penalty
coefficient; it is a mathematically principled risk measure with a derivation
you can verify. The full derivation is in the whitepaper.

**Severity-matched authority (the safety invariant).** The guillotine verdict — a
hard retirement — is only issued when a deterministic oracle signal (consecutive
hard failures, HP below the floor, or a manual override flag) backs the decision.
High hazard alone triggers HOLD, which routes the retirement decision to a human
committee. This prevents the system from auto-killing an agent on statistical noise,
and it holds across all five layers regardless of how good a probabilistic signal looks.

Together, these five layers give you: deterministic routing, Bayesian skill
tracking, forward-looking health monitoring, hard risk gates, coherent tail-risk
pricing, and a human-in-the-loop on irreversible decisions. No LLM in the
routing path. No fitted constants in the open baseline — fitted calibration and
advanced risk tiers are a separate private/commercial layer.

## Five MCP tools, one install line

The entire control plane is exposed as five MCP tools:

- `govern_route` — run the V5 auction
- `record_outcome` — apply an observation to agent state
- `survival_check` — get a survival verdict for one agent
- `model_route` — tier selection with the money/security hard gate
- `fleet_snapshot` — full fleet health summary

All five tools are **stateless at call time**: prior state arrives as plain JSON
in the request, results return as plain JSON. There is no database embedded in
the tool logic. This means every tool invocation is a pure function you can
unit-test offline with no infrastructure.

Install and connect to Claude Desktop:

```bash
pip install quantchoai-governor mcp
```

Then add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "quantchoai-governor": {
      "command": "python",
      "args": ["-m", "quant.governance.mcp.server"],
      "cwd": "/path/to/quantchoai-governor/backend",
      "env": { "QCHO_MCP_TRANSPORT": "stdio" }
    }
  }
}
```

The five tools appear in Claude Desktop immediately. No API key, no network
calls from the tool logic, no database.

## Open-core model

The open baseline — this repository — ships the complete v1-v5 mathematics
under MIT. Nothing is stripped or stubbed. You can read the CVaR derivation,
the severity-matched authority rule, and the V3 hazard estimator in the source.

The hosted tier adds two things the open baseline does not include:

1. A **server-side calibration overlay** — parameters fitted on shadow/replay
   data from realistic fleet simulations. The shadow-validation process is what
   produces the concrete performance numbers in our materials; those numbers
   carry an explicit "shadow-validated, pre-revenue" caveat because that is what
   they are. The overlay is loaded on operator infrastructure only; it is never
   shipped to client machines.

2. **Managed infrastructure + compliance export** — a hosted MCP endpoint with
   authentication, a structured audit log, and a report API for compliance use
   cases.

The moat is the calibration overlay. The math is MIT.

## Who this is for

If you are building or running a multi-agent system where:

- task routing decisions need to be reproducible and auditable
- some tasks carry financial, legal, or security exposure
- you want agent retirement to be a structured state machine rather than
  implicit neglect
- you need to hand compliance reviewers a structured decision log

— then quantchoai-governor is designed for you. It is particularly relevant for
institutional deployments where "the model decided" is not a sufficient answer.

## Where to go from here

- **Repo and quickstart:** [https://github.com/quantchoai/quantchoai-governor]
- **Runnable demo (no Mongo, no network):**
  `cd backend && python -m quant.governance.kit.quickstart`
- **Whitepaper** (CVaR derivation, severity-matched authority, open/closed split
  architecture): available on request → hello@quantchoai.com
- **Hosted tier access:** hello@quantchoai.com

We are pre-revenue. We built this to solve a problem we kept running into
ourselves, and we are releasing the math open because we believe the field
should have it. If the V5 objective function is wrong, we want to know — open a
GitHub issue or email us.

---

*QuantChoAI · quantchoai.com · hello@quantchoai.com*
