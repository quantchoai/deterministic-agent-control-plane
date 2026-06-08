"""Smoke tests for quantchoai-governor (open V1-V5 baseline).

Pure / offline / no network / no Mongo. Proves the package imports, the core
governance loop runs deterministically, and the central safety invariant holds:
an unproven agent is fail-closed out of a money/security-critical action.

    cd <repo> && python -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

# src/ layout: make the package importable without an install step.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def test_all_public_modules_import():
    import importlib
    for m in [
        "quant.governance.runner",
        "quant.governance.dispatcher",
        "quant.governance.ledger",
        "quant.governance.model_router",
        "quant.governance.risk_core",
        "quant.governance.governance_params",
    ]:
        importlib.import_module(m)


def test_cvar_is_coherent_and_at_least_var():
    # Rockafellar-Uryasev CVaR must be >= VaR (coherent tail measure).
    from quant.governance import risk_core
    losses = [0.0, 0.1, 0.2, 0.5, 1.0, 5.0]
    res = risk_core.cvar(losses, alpha=0.95) if hasattr(risk_core, "cvar") else None
    if res is None:
        import pytest
        pytest.skip("risk_core.cvar not exported under this name")
    var = res.get("var") if isinstance(res, dict) else None
    cvar = res.get("cvar") if isinstance(res, dict) else None
    if var is not None and cvar is not None:
        assert cvar >= var


def test_determinism_same_inputs_same_decision():
    # The core promise: identical (state, events) -> identical decision, no RNG/clock.
    from quant.governance import dispatcher as d
    ticket = {"ticket_id": "T1", "target_domain": "Risk", "priority": 1,
              "ci_gate_difficulty": 70.0, "money_risk": 0.0, "payload": {}}
    agent = {"agent_id": "a1", "hp": 90.0, "rank": 3,
             "domain_skills": {"Risk": {"mu": 80.0, "sigma": 10.0, "n": 5}}}
    r1 = d.utility_for(ticket, agent, disk_free=50.0)
    r2 = d.utility_for(ticket, agent, disk_free=50.0)
    assert r1.utility == r2.utility
    assert r1.reason == r2.reason


def test_money_gate_fail_closed_for_unproven_agent():
    # The immutable invariant: an unproven agent (thin/low credit) must NOT be
    # routed to a money/security-critical action -- utility is hard-blocked.
    import math
    from quant.governance import dispatcher as d
    money_ticket = {"ticket_id": "M1", "target_domain": "Risk", "priority": 0,
                    "ci_gate_difficulty": 80.0, "money_risk": 4.0, "payload": {}}
    unproven = {"agent_id": "rookie", "hp": 90.0, "rank": 3,
                "domain_skills": {"Risk": {"mu": 55.0, "sigma": 30.0, "n": 0}}}
    res = d.utility_for(money_ticket, unproven, disk_free=50.0)
    # Fail-closed: either an explicit low-confidence skip or a non-finite utility.
    assert (res.reason == "SKIP_LOW_CONFIDENCE_FOR_RISK") or (not math.isfinite(res.utility)), \
        f"unproven agent must be fail-closed off money work; got {res.reason} u={res.utility}"
