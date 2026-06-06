"""V1-V6 agent-governance control plane (PRIVATE MOAT).

This package is the ported control plane that already existed and was
unit-tested under ``_cto2_v1v6_private``; it is moved here verbatim (math
UNCHANGED) and its top-level imports are rewritten to be package-relative
so it runs inside the backend as ``quant.governance.*``.

PRIVATE / SHADOW ONLY. None of these symbols may be re-exported on any
public-eligible surface. ``v6_runner`` is the deterministic control loop;
``live_adapter`` adapts REAL Mongo roster/work into the runner and persists
SHADOW decision snapshots into ``db.governance_shadow`` (it NEVER kills or
mutates any live agent/collection).
"""
from __future__ import annotations

__all__ = ["v6_runner", "live_adapter", "efficiency"]
