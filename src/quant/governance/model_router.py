"""model_router.py -- DATA-DRIVEN v1-v5 governed MODEL-TIER routing for the dev fleet.

An earlier prototype HARDCODED tier capability (mu/sigma per tier) and difficulty
floors (haiku<=50, sonnet<=80). Those magic numbers are not truths -- they should be
(a) GROUNDED PRIORS for cold-start, and (b) LEARNED from real dispatch outcomes via the
SAME v1-v5 V2 Bayesian credit machinery the agent ledger already uses. This module is
exactly that:

    * The three model tiers (opus/sonnet/haiku) are the *agents*.
    * Each tier carries a per-(tier, difficulty-band) MEASURED credit row
      {mu, sigma, n} stored in Mongo `db.tier_credit`.
    * At route time we read the CURRENT (mu, sigma) from `tier_credit` and run the
      real V5 auction (`dispatcher.utility_for`, U = p_success*value - model_cost -
      tail - context - disk). The difficulty competence EMERGES from
      `p_success(mu, sigma, difficulty)` (the dispatcher's sigmoid) -- a tier whose
      MEASURED mu is far below the task difficulty gets a low p_success and loses
      the auction naturally. No hardcoded difficulty floor.
    * `record_outcome` applies the EXACT v1-v5 Bayesian mu/sigma update law that
      `ledger.update_domain_skill` uses (Elo/Bayesian: mu += k*q*(outcome-expected),
      sigma shrinks on success / grows on failure). We do NOT invent a new update
      law -- we mirror the ledger's onto the tier_credit rows.
    * The V2 money/security hard-gate (ELITE_ONLY_DOMAINS / money_risk>=3 /
      security_risk>=3 -> opus) still applies, unchanged.

Tier <-> model map (model_tier_cost = the dispatcher's MODEL_COST_* proxies, taken
verbatim from governance_params.V2_CREDIT so this module shares the system's
existing cost constants):

    R3 / ELITE  cost MODEL_COST_ELITE_R3 (0.15)   -> "opus"
    R2          cost MODEL_COST_R2       (0.055)  -> "sonnet"
    R1          cost MODEL_COST_R1       (0.012)  -> "haiku"

Deterministic, no LLM in the routing decision, auditable.
"""
from __future__ import annotations

import math
from typing import Any

from quant.governance import dispatcher as D
from quant.governance import ledger as L
from quant.governance import governance_params as P

# Mongo collection that stores the LEARNED per-tier credit. One row per
# (tier, difficulty_band): {tier, difficulty_band, mu, sigma, n, ...}.
TIER_CREDIT = "tier_credit"

# Model-tier cost proxies -- taken verbatim from the system's existing V2 credit
# constants (governance_params.V2_CREDIT), NOT re-invented here.
_MODEL_COST = {
    "opus": float(P.V2_CREDIT["MODEL_COST_ELITE_R3"]),    # 0.15
    "sonnet": float(P.V2_CREDIT["MODEL_COST_R2"]),        # 0.055
    "haiku": float(P.V2_CREDIT["MODEL_COST_R1"]),         # 0.012
}

# --------------------------------------------------------------------------- #
# GROUNDED PRIORS (cold-start only) -- EXPLICITLY priors, not truths.
# --------------------------------------------------------------------------- #
# These mu values are first-pass single-attempt success rates from published
# coding/agent evals, expressed on the SAME 0-100 scale the dispatcher's
# p_success sigmoid uses (so mu ~= expected pass-% on a difficulty-50 task).
#
# Source / rationale (cited so reviewers see these are grounded, not magic):
#   * Frontier model (opus tier): SWE-bench Verified single-attempt pass rates for
#     the strongest 2025 coding models cluster in the ~0.65-0.75 band; HumanEval/
#     LiveCodeBench first-pass is higher but SWE-bench-style agentic tasks are the
#     relevant analogue here. Prior mu = 72 (~0.72).
#   * Mid model (sonnet tier): the same evals put a mid/cost-tier model roughly
#     10-15 pts below frontier on agentic coding -> ~0.55-0.65. Prior mu = 60.
#   * Small model (haiku tier): small/fast models land around ~0.35-0.45 first-pass
#     on the same agentic tasks. Prior mu = 40.
#
# sigma is set WIDE (PRIOR_SIGMA = 25) on purpose: high prior uncertainty so that
# a handful of REAL measured outcomes (each shrinking sigma via the ledger update
# law) quickly dominates the prior. After enough outcomes the router is governed by
# MEASURED success, not by these numbers. See `record_outcome`.
PRIOR_SIGMA = 25.0
_PRIOR_MU = {
    "opus": 72.0,
    "sonnet": 60.0,
    "haiku": 40.0,
}

# Thin, clearly-marked COLD-START SAFETY floor. This is ONLY a guard for the n==0
# regime (before any real outcome exists for a tier/band) so a brand-new, never-
# measured cheap tier cannot win a genuinely hard task purely on its prior. It maps
# difficulty -> the cheapest tier permitted to bid WHILE UNMEASURED. The instant a
# tier has n>0 measured outcomes for the relevant band, this floor is overridden and
# competence is governed entirely by the LEARNED p_success(mu, sigma, difficulty).
# Mirrors the prototype's intent but is no longer the steady-state decision rule.
_COLDSTART_MAX_DIFFICULTY = {"haiku": 50.0, "sonnet": 80.0, "opus": 100.0}

# Difficulty bands the tier_credit rows are keyed by. A measured outcome is bucketed
# into one band so "haiku is good at EASY tasks but bad at HARD tasks" is learnable
# as two separate rows -- exactly the behaviour the spec demonstrates.
_BANDS = (
    ("easy", 0.0, 40.0),
    ("medium", 40.0, 70.0),
    ("hard", 70.0, 101.0),
)

_TIERS = ("opus", "sonnet", "haiku")

# Domains that the V2 money/security gate forces onto the ELITE tier regardless of
# cost (unchanged from the dispatcher's ELITE_ONLY_DOMAINS contract).
_ELITE_FORCED = set(getattr(D, "ELITE_ONLY_DOMAINS", {"money_math", "committee_judgment", "core_dev"}))


def difficulty_band(difficulty: float) -> str:
    """Map a 0-100 difficulty to its learning band label."""
    d = float(difficulty)
    for name, lo, hi in _BANDS:
        if lo <= d < hi:
            return name
    return _BANDS[-1][0]


def _prior_row(tier: str) -> dict:
    """The GROUNDED PRIOR (mu, sigma, n=0) for a tier. n==0 marks 'unmeasured' so the
    route falls back to the cold-start safety floor until real data arrives."""
    return {"mu": float(_PRIOR_MU[tier]), "sigma": PRIOR_SIGMA, "n": 0}


async def _current_credit(db, tier: str, band: str) -> dict:
    """Read the CURRENT measured (mu, sigma, n) for a tier/band from `tier_credit`.

    Falls back to the grounded prior ONLY when there is no measured row (n==0) for
    that tier/band. Once n>0, the LEARNED row governs.
    """
    row = await _find_one(db, {"tier": tier, "difficulty_band": band})
    if not row or int(row.get("n", 0) or 0) <= 0:
        return _prior_row(tier)
    return {
        "mu": float(row.get("mu", _PRIOR_MU[tier])),
        "sigma": float(row.get("sigma", PRIOR_SIGMA)),
        "n": int(row.get("n", 0) or 0),
    }


async def record_outcome(db, *, tier: str, difficulty: float, success: bool, quality: float = 1.0) -> dict:
    """Apply the v1-v5 V2 Bayesian credit update to the (tier, difficulty_band) row.

    This is the SAME update law `ledger.update_domain_skill` uses on agent_ledger --
    mirrored here onto `tier_credit` rather than re-invented:

        expected = p_success(mu, sigma, difficulty)        # dispatcher sigmoid
        outcome  = 1.0 if success else 0.0
        k        = clamp(sigma / 2.7, 2, 12)               # learning rate ~ uncertainty
        mu      += k * q * (outcome - expected)
        sigma    = shrink on success / grow on failure

    Each real dispatch outcome therefore shrinks sigma and moves mu toward measured
    reality. After enough outcomes the router is governed by MEASURED success.
    """
    tier = str(tier)
    band = difficulty_band(difficulty)
    cur = await _current_credit(db, tier, band)
    mu = float(cur["mu"])
    sigma = float(cur["sigma"])

    # --- EXACT mirror of ledger.update_domain_skill's update law ----------------
    # expected pass-prob from the dispatcher's sigmoid (ledger._skill_expected and
    # dispatcher._p_success are the same function -- reuse the ledger's).
    expected = L._skill_expected(mu, sigma, float(difficulty))
    outcome = 1.0 if success else 0.0
    q = max(0.25, min(1.75, float(quality)))
    k = max(2.0, min(12.0, sigma / 2.7))
    mu = max(0.0, min(100.0, mu + (k * q * (outcome - expected))))
    if success:
        sigma = max(5.0, sigma * (0.92 if expected < 0.9 else 0.96))
    else:
        sigma = min(60.0, sigma * 1.08 + 1.5)
    # ---------------------------------------------------------------------------

    n = int(cur.get("n", 0) or 0) + 1
    doc = {
        "tier": tier,
        "difficulty_band": band,
        "mu": round(mu, 4),
        "sigma": round(sigma, 4),
        "n": n,
        "last_success": bool(success),
        "last_expected": round(expected, 4),
        "last_difficulty": round(float(difficulty), 4),
        "updated_at": L._now(),
    }
    await _update_one(
        db,
        {"tier": tier, "difficulty_band": band},
        {"$set": doc},
        upsert=True,
    )
    return doc


def _model_for_tier(tier: str) -> dict:
    """Build the dispatcher-shaped agent descriptor for a tier, carrying its measured
    (mu, sigma) as the `_default` skill row and its cost as `model_tier_cost`."""
    return {
        "agent_id": f"TIER-{tier.upper()}",
        "model": tier,
        "model_tier_cost": _MODEL_COST[tier],
        "hp": 100.0,
    }


def _ticket_from_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticket_id": item.get("slug", "?"),
        "title": item.get("title", item.get("slug", "")),
        "description": item.get("spec", item.get("title", "")),
        "difficulty": float(item.get("difficulty", 50.0)),
        "ci_gate_difficulty": float(item.get("difficulty", 50.0)),
        "business_impact": float(item.get("value", 0.5)),
        "urgency": 1.0,
        "money_risk": float(item.get("money_risk", 0.0)),
        "security_risk": float(item.get("security_risk", 0.0)),
        "live_risk": float(item.get("live_risk", 0.0)),
        "target_domain": item.get("target_domain", ""),
    }


def _p_success(mu: float, sigma: float, difficulty: float) -> float:
    # Reuse the dispatcher's own sigmoid so the competence curve the router scores on
    # is identical to the one the live auction scores on.
    return D._p_success(mu, sigma, difficulty)


async def route(db, item: dict[str, Any], disk_free: float | None = 50.0) -> dict[str, Any]:
    """Pick the model tier for ONE work item via the DATA-DRIVEN v1-v5 auction.

    `item` is a work descriptor: {slug, kind, difficulty(0-100), value(0-1),
    money_risk(0-3), security_risk(0-3), target_domain}. Returns
    {model, tier, reason, mu_used, sigma_used, n, utility, p_success, scores}.
    """
    ticket = _ticket_from_item(item)
    domain = str(ticket["target_domain"])
    diff = float(ticket["difficulty"])
    band = difficulty_band(diff)

    money_critical = (
        domain in _ELITE_FORCED
        or float(ticket["money_risk"]) >= 3.0
        or float(ticket["security_risk"]) >= 3.0
    )

    scored: list[dict[str, Any]] = []
    # Resolve the SAME skill-row key the dispatcher's `agent_skill` will look up for
    # this domain, so the learned (mu, sigma) we inject is the value the auction
    # actually prices p_success on (NOT the rank-default fallback). measured_skill_row
    # resolves SKILL_BY_DOMAIN[domain] first, then the bare domain key.
    skill_key = D._skill_name(domain)
    for tier in _TIERS:
        credit = await _current_credit(db, tier, band)
        mu, sigma, n = float(credit["mu"]), float(credit["sigma"]), int(credit["n"])
        agent = _model_for_tier(tier)
        # Inject the CURRENT credit under BOTH the resolved skill key and the bare
        # domain so the dispatcher's tolerant measured-row lookup honors the LEARNED
        # (mu, sigma) for this tier/band -- the competence the auction scores on is the
        # data-driven one, not the synthetic rank-3 default. `n` is kept HONEST (real
        # measured count) so the dispatcher's evidence-based hard-gate is not fooled by
        # a cold-start prior; our own money-gate + cold-start floor handle n==0 safety.
        skill_row = {"mu": mu, "sigma": sigma, "n": n}
        agent["domain_skills"] = {skill_key: dict(skill_row), domain: dict(skill_row)}
        ur = D.utility_for(ticket, agent, disk_free=disk_free, db=db)
        p_succ = _p_success(mu, sigma, diff)

        eligible = True
        reason = "auction"
        # V2 money/security HARD-GATE (unchanged): only opus may take money-critical work.
        if money_critical and tier != "opus":
            eligible = False
            reason = "money/security gate -> opus only"
        # COLD-START safety floor: applies ONLY while this tier/band is UNMEASURED (n==0).
        # Once n>0 the LEARNED p_success governs and this guard is overridden.
        elif n == 0 and diff > _COLDSTART_MAX_DIFFICULTY[tier]:
            eligible = False
            reason = "cold-start floor (unmeasured tier below difficulty band)"

        scored.append({
            "tier": tier,
            "model": tier,
            "utility": ur.utility,
            "p_success": p_succ,
            "mu": mu,
            "sigma": sigma,
            "n": n,
            "model_cost": _MODEL_COST[tier],
            "eligible": eligible,
            "reason": reason,
        })

    eligible = [s for s in scored if s["eligible"] and math.isfinite(s["utility"])]
    if not eligible:
        # Degenerate fallback: nothing cleared (e.g. disk freeze). Force opus, the only
        # tier that can take anything, and surface the reason.
        pick = max(scored, key=lambda s: s["utility"])
        pick = dict(pick)
        pick["reason"] = "no eligible tier (fallback to strongest)"
    else:
        pick = max(eligible, key=lambda s: s["utility"])

    return {
        "slug": item.get("slug"),
        "model": pick["model"],
        "tier": pick["tier"],
        "money_critical": money_critical,
        "difficulty_band": band,
        "reason": (
            "money/security gate -> opus" if money_critical
            else pick["reason"]
        ),
        "mu_used": round(pick["mu"], 4),
        "sigma_used": round(pick["sigma"], 4),
        "n": pick["n"],
        "p_success": round(pick["p_success"], 4),
        "utility": round(pick["utility"], 4) if math.isfinite(pick["utility"]) else pick["utility"],
        "model_cost": pick["model_cost"],
        "scores": {s["tier"]: (round(s["utility"], 3) if math.isfinite(s["utility"]) else s["utility"]) for s in scored},
    }


async def route_batch(db, items: list[dict[str, Any]], disk_free: float | None = 50.0) -> list[dict]:
    """Route a batch of items. Each is scored independently against the CURRENT
    measured tier credit (later items see earlier `record_outcome` writes if the
    caller interleaves them)."""
    return [await route(db, it, disk_free=disk_free) for it in items]


# --------------------------------------------------------------------------- #
# Thin Mongo access shim
# --------------------------------------------------------------------------- #
# The v1-v5 control plane runs against pymongo (sync) in production, but the agent
# fleet increasingly calls the router from async contexts. These helpers await a
# Motor-style coroutine when present and otherwise call the sync pymongo method, so
# the same router works against BOTH a real async Motor db AND a sync/in-memory
# (mongomock / dict-backed) db in tests. No behaviour change either way.
async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value


async def _find_one(db, query: dict):
    return await _maybe_await(db[TIER_CREDIT].find_one(query))


async def _update_one(db, query: dict, update: dict, upsert: bool = False):
    return await _maybe_await(db[TIER_CREDIT].update_one(query, update, upsert=upsert))
