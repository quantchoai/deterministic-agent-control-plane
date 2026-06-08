# quantchoai-governor

Open-core v1-v5 agent-governance thin client for Python.

Exposes the 5-step Governor API over the open baseline locally, or delegates
to a hosted premium endpoint when you have a subscription.

## Install

```bash
pip install quantchoai-governor
```

No third-party dependencies — stdlib only (`math`, `statistics`, `urllib`).

## 5-Step Deploy

```python
from solo_fleet_api import Governor

# --- Local open baseline (no network required) ---
g = Governor()

# 1. Route a task to the best agent
result = g.route(
    "compute portfolio VaR for the risk desk",
    ["agent-alpha", "agent-beta", "agent-gamma"],
)
print(result["winner"])       # e.g. "agent-alpha"
print(result["winner_score"]) # expected utility
print(result["mode"])         # "local"

# 2. Record the outcome (returns updated Bayesian mu/sigma — caller persists)
update = g.record(
    {"name": "agent-alpha", "mu": 72.0, "sigma": 14.0},
    success=True,
    difficulty=65.0,
)
print(update["mu"], update["sigma"])   # updated credit row

# 3. Check survival gate (HP + hazard)
sv = g.survival(
    {"name": "agent-alpha", "hp": 35.0},
    streak_failures=2,
    requeue_count=1,
)
print(sv["action"])  # "ok" | "soft_recycle" | "guillotine"

# 4. Fleet health snapshot
snap = g.snapshot([
    {"name": "agent-alpha", "hp": 80.0},
    {"name": "agent-beta",  "hp": 45.0},
    {"name": "agent-gamma", "hp": 18.0},
])
print(snap["zone_counts"])  # {"elite": 1, "standard": 1, ...}
print(snap["fleet_cvar"])   # expected-shortfall of HP deficits

# 5. Model tier selection
mr = g.model_route("write a risk pricing model", difficulty=70.0)
print(mr["model"])   # "opus" | "sonnet" | "haiku"
print(mr["reason"])  # why this tier won
```

## Open (local) vs Premium (hosted)

| Feature | Open / local | Premium / hosted |
|---|---|---|
| Survival gate (V1 HP rules) | Generic thresholds | Calibrated from real fleet HP history |
| Credit routing (V2 Bayesian mu/sigma) | Grounded priors | Learned from real dispatch outcomes |
| Hazard model (V3 Poisson) | Flat baseline | Calibrated per-agent lambda + NB2 overdispersion |
| CVaR tail pricing | Historical estimator | Rockafellar-Uryasev + Cornish-Fisher + CI bound |
| Marginal-CVaR vendor concentrations | Not included | Euler/marginal allocation over vendor book |
| Fitted constants | NOT in this package | Server-side, operator-calibrated |

The open package ships the **frame + generic math only**.  Fitted calibration,
the marginal-CVaR vendor concentrations, and premium governance models live
server-side and are never bundled in this package.

## Pointing at a Hosted Endpoint

```python
g = Governor(remote_url="https://governance.example.com/v1")

# Every method now delegates its JSON payload to the hosted service.
# The premium engine runs server-side; this package is a thin transport wrapper.
result = g.route("compute VaR", ["agent-alpha", "agent-beta"])
print(result["mode"])  # "hosted"
```

Each method POSTs to `{remote_url}/{method_name}` with the same parameters.
Set the `SOLO_FLEET_REMOTE_URL` environment variable to avoid passing the URL
in code.

## MCP Server

The package ships a stdio MCP server entry-point:

```bash
quantchoai-governor-mcp
```

It reads newline-delimited JSON requests on stdin and writes responses on stdout:

```json
{"id": "1", "method": "route",  "params": {"task": "…", "agents": ["a", "b"]}}
{"id": "1", "result": {"winner": "a", "winner_score": 0.42, "mode": "local"}}
```

Set `SOLO_FLEET_REMOTE_URL` to delegate to a hosted endpoint from the MCP server.

## License

MIT
