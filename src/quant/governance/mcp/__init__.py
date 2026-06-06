"""v1-v6 governance MCP server package.

Exposes the real `quant.governance` engine (dispatcher auction, HP ledger, the V6
oracle/tail-risk layer, and the data-driven model-tier router) as Model Context
Protocol tools so an operator-hosted deployment can drive the SAME governance the
live fleet runs on -- without re-implementing any of it.

The tool *logic* lives in plain importable functions in `server.py`
(`govern_route_logic`, `record_outcome_logic`, `survival_check_logic`,
`model_route_logic`, `fleet_snapshot_logic`) so it unit-tests deterministically
without the MCP transport. The FastMCP wiring is import-guarded: when the official
`mcp` SDK is absent the logic functions still import and run.
"""
from __future__ import annotations

from .server import (  # noqa: F401
    PREMIUM_AVAILABLE,
    fleet_snapshot_logic,
    govern_route_logic,
    model_route_logic,
    record_outcome_logic,
    survival_check_logic,
)

__all__ = [
    "PREMIUM_AVAILABLE",
    "govern_route_logic",
    "record_outcome_logic",
    "survival_check_logic",
    "model_route_logic",
    "fleet_snapshot_logic",
]
