"""V1-V5 deterministic agent-governance control plane (open baseline).

This package is the open, MIT-licensed implementation of the V1-V5 deterministic
governance frame: a health/survival ledger (V1), a Bayesian skill-credit model and
money/security hard gate (V2), a Poisson failure-hazard model (V3), warm-propulsion
(V4), and a risk-budgeted Expected-Utility auction with coherent CVaR tail pricing
(V5). No LLM sits in the routing loop, so every decision is deterministic, reproducible,
and auditable.

``runner`` is the deterministic V1-V5 control loop (offline/shadow: it scores, compares
against a FIFO baseline, and logs; it never kills a live agent or mutates a database).
``dispatcher`` is the V5 auction, ``ledger`` the V1 survival + V3 hazard math, and
``risk_core`` the stdlib-only V1-V5 risk/hazard primitives (CVaR, Poisson failure,
log-normal latency, factor decomposition, concentration check, ...).

Fitted calibration overlays and advanced risk tiers are a separate private/commercial
layer and are not bundled in this package.
"""
from __future__ import annotations

__all__ = ["runner", "dispatcher", "ledger", "risk_core", "model_router", "governance_params"]
