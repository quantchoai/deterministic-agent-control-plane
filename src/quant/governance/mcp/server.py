"""v1-v5 governance MCP server (hosted-capable; open/closed split).

WHAT THIS IS
------------
A Model Context Protocol (MCP) server that exposes the REAL `quant.governance`
engine as a small set of governance tools. It wraps -- never re-implements --

    * dispatcher.utility_for      -- the V5 risk-adjusted auction objective
    * ledger.calculate_hazard +   -- the V3 hazard model + the absolute-HP event
      ledger.EVENTS / clamp/rank     ledger and its guillotine line
    * risk_core.concentration_check -- the coherent vendor-concentration cap (V5)
                                     + the historical CVaR tail price
    * model_router (tier auction) -- the data-driven opus/sonnet/haiku tier pick,
                                     scored on the dispatcher's own p_success/cost

Exposed MCP tools:
    govern_route(task, agents)        -> {winner, utility, reason, ...}
    record_outcome(agent_id, success, -> {hp, mu, sigma, hazard, guillotined, ...}
                   event)
    survival_check(agent)             -> {verdict, hp, hazard, reason, ...}
    model_route(task)                 -> {tier, model, reason, ...}
    fleet_snapshot(fleet)             -> summary (counts, concentration, at-risk)

HOSTED-CAPABLE  (the moat never ships to users)
-----------------------------------------------
The server is designed to run on the OPERATOR's own infrastructure so the
proprietary "moat math" -- the fitted calibration overlay, the marginal-CVaR /
vendor-correlation fitted constants, and any premium tuning -- is loaded ONLY
server-side. A user connecting an MCP client to the hosted URL gets the full
governed decision; they never receive the closed modules.

OPEN / CLOSED SPLIT
-------------------
The closed premium layer lives in an OPTIONAL `quant.governance._premium`
namespace. At import time we try to load it:

    * present (hosted deployment)  -> PREMIUM_AVAILABLE = True; the premium overlay
                                      refines the open baseline (calibration of the
                                      auction utility / hazard / tier scores, and
                                      the fitted marginal-CVaR vendor-correlation).
    * absent  (local open deploy)  -> PREMIUM_AVAILABLE = False; the engine runs on
                                      its OPEN BASELINE -- the same correct v1-v5
                                      math, just without the fitted premium overlay.

Either way the tool logic is identical CODE; the only difference is whether the
premium overlay hook is wired. This keeps a local open deployment fully functional
(and honest) while a hosted deployment carries the moat.

TRANSPORT GUARD
---------------
The MCP transport (FastMCP) is import-guarded. When the official `mcp` SDK is not
installed the tool LOGIC functions (`*_logic`) still import and run, so they are
unit-testable with no transport and no live MongoDB. `main()` builds and runs the
FastMCP server only when the SDK is present.
"""
from __future__ import annotations

import math
import os
from typing import Any

# --------------------------------------------------------------------------- #
# REAL engine imports -- these are pure-enough for the tool logic: every wrapped
# call below is driven with explicit state (disk_free, agent dict, prior credit)
# so NONE of the logic functions require a live MongoDB connection.
# --------------------------------------------------------------------------- #
from quant.governance import dispatcher as _dispatcher
from quant.governance import ledger as _ledger
from quant.governance import model_router as _model_router
from quant.governance import risk_core as _risk_core

# --------------------------------------------------------------------------- #
# OPEN / CLOSED SPLIT -- optional premium overlay.
# --------------------------------------------------------------------------- #
# The closed moat (fitted calibration + marginal-CVaR vendor-correlation constants)
# lives behind a namespace that is shipped ONLY to the operator's hosted infra.
# When absent we run the open baseline. The overlay, when present, must expose any
# subset of the following optional hooks (all pure, deterministic):
#
#   calibrate_utility(utility: float, ctx: dict) -> float
#   calibrate_hazard(hazard: float, ctx: dict) -> float
#   calibrate_tier_scores(scores: dict, ctx: dict) -> dict
#   vendor_corr_rho(ctx: dict) -> float          # fitted vendor failure correlation
#
# A missing hook simply means "no premium refinement for that leg" -- the open
# baseline value passes through unchanged.
try:  # pragma: no cover - exercised only on a hosted deployment that ships _premium
    from quant.governance import _premium as _premium  # type: ignore

    PREMIUM_AVAILABLE = True
except Exception:
    _premium = None  # type: ignore
    PREMIUM_AVAILABLE = False


def _premium_hook(name: str):
    """Return the named premium overlay callable, or None when running open baseline."""
    if _premium is None:
        return None
    fn = getattr(_premium, name, None)
    return fn if callable(fn) else None


def _apply_premium_scalar(name: str, baseline: float, ctx: dict) -> float:
    """Run a scalar premium overlay hook if present; otherwise return the open baseline.

    Any failure in the closed overlay degrades GRACEFULLY to the open baseline so a
    hosted deployment can never be taken down by a bad overlay -- the worst case is it
    behaves like the open baseline. Non-finite overlay output is rejected.
    """
    fn = _premium_hook(name)
    if fn is None:
        return baseline
    try:
        out = float(fn(baseline, dict(ctx)))
    except Exception:  # pragma: no cover - defensive; overlay must never crash a tool
        return baseline
    return out if math.isfinite(out) else baseline


def _premium_vendor_rho(ctx: dict) -> float | None:
    """Fitted vendor-failure correlation from the premium overlay, else None (open)."""
    fn = _premium_hook("vendor_corr_rho")
    if fn is None:
        return None
    try:
        out = float(fn(dict(ctx)))
    except Exception:  # pragma: no cover
        return None
    if not math.isfinite(out):
        return None
    return max(0.0, min(1.0, out))


def deployment_mode() -> dict:
    """Report whether this process is a HOSTED (full/moat) or LOCAL (open baseline)
    deployment. Surfaced to clients so a local open user knows they are NOT getting
    the premium overlay (numbers-not-adjectives honesty)."""
    return {
        "premium_available": PREMIUM_AVAILABLE,
        "mode": "hosted-full" if PREMIUM_AVAILABLE else "local-open-baseline",
        "marginal_cvar_enabled": bool(_dispatcher.MARGINAL_CVAR_ENABLED),
        "note": (
            "Hosted: fitted calibration + marginal-CVaR overlay applied server-side."
            if PREMIUM_AVAILABLE
            else "Local open baseline: correct v1-v5 math without the premium overlay."
        ),
    }


# --------------------------------------------------------------------------- #
# Shared coercion helpers (deterministic, no I/O).
# --------------------------------------------------------------------------- #
def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _disk_free_arg(disk_free: float | None) -> float:
    """Resolve a disk-free GB value for the auction WITHOUT touching the host.

    The tools are decision endpoints, not the live poller, so by default we price on a
    healthy disk-free figure (the auction's disk penalty is then ~negligible, isolating
    the GOVERNANCE decision). A caller may pin an explicit value to exercise the disk
    governor. We never read the real disk here -- a hosted endpoint must not couple a
    routing decision to the server box's incidental free space."""
    if disk_free is not None:
        return max(0.0, _f(disk_free, 50.0))
    env = os.environ.get("QCHO_MCP_DISK_FREE_GB")
    if env:
        return max(0.0, _f(env, 50.0))
    return 50.0


def _normalize_agent(agent: dict) -> dict:
    """Shape a loosely-typed agent payload into the dispatcher's expected descriptor.

    Accepts the convenience keys an MCP caller is likely to send (hp, mu, sigma, n,
    vendor/model) and lifts a flat (mu, sigma, n) into the dispatcher's
    `domain_skills` measured-row form so the REAL `agent_skill` lookup honors it.
    A caller that already passes a full `domain_skills` dict is left untouched.
    """
    out = dict(agent or {})
    out.setdefault("agent_id", str(out.get("id") or out.get("name") or "agent"))
    if "hp" not in out and "weight" in out:
        out["hp"] = out["weight"]
    out.setdefault("hp", _ledger.DEFAULT_HP)
    if "domain_skills" not in out and any(k in out for k in ("mu", "sigma", "n")):
        row = {
            "mu": _f(out.get("mu"), 60.0),
            "sigma": _f(out.get("sigma"), 24.0),
            "n": int(_f(out.get("n"), 0)),
        }
        # register under a generic key; dispatcher.agent_skill resolves per-domain via
        # the SKILL_BY_DOMAIN map + tolerant normalized fallback, so we also stamp the
        # mapped skill name when the agent carries a target/home domain.
        out["domain_skills"] = {"general_delivery": row}
        dom = out.get("domain") or out.get("home_domain")
        if dom:
            out["domain_skills"][_dispatcher._skill_name(str(dom))] = dict(row)
            out["domain_skills"][str(dom)] = dict(row)
    return out


# =========================================================================== #
# TOOL LOGIC (plain importable functions; no transport, no live DB)
# =========================================================================== #
def govern_route_logic(task: dict, agents: list[dict],
                       disk_free: float | None = None) -> dict:
    """Run the REAL V5 auction (`dispatcher.utility_for`) for `task` over `agents`
    and return the winner.

    Each (task, agent) pair is scored on the exact live objective
        U = p_success*value - model_cost - tail_risk - context - disk - goodhart
    The tail term is the engine's coherent CVaR, the money/security hard-gate and the
    low-confidence-for-risk gate apply unchanged, and -- on a hosted deployment -- the
    premium overlay calibrates the final utility per candidate (open baseline: pass-through).

    Returns: {winner, utility, reason, p_success, mu, sigma, vendor, scores, ...}.
    Deterministic; never touches MongoDB (features are inferred from `task` alone).
    """
    if not isinstance(task, dict):
        return {"error": "task must be an object"}
    if not agents:
        return {"winner": None, "utility": None, "reason": "NO_AGENTS", "scores": []}

    free = _disk_free_arg(disk_free)
    # infer ticket features ONCE (agent-independent) -- the real engine call, no DB.
    features = _dispatcher.infer_ticket_features(task, db=None)
    domain = str(task.get("target_domain") or "")

    scored: list[dict] = []
    for raw in agents:
        agent = _normalize_agent(raw)
        ur = _dispatcher.utility_for(task, agent, disk_free=free, db=None, features=features)
        utility = ur.utility
        # OPEN/CLOSED: premium calibration of the utility (hosted only). Skip non-finite
        # (a hard gate / disk freeze) so a -inf eligibility verdict is never "calibrated".
        if math.isfinite(utility):
            utility = _apply_premium_scalar(
                "calibrate_utility", utility,
                {"domain": domain, "reason": ur.reason, "p_success": ur.p_success,
                 "money_risk": features.money_risk, "security_risk": features.security_risk,
                 "live_risk": features.live_risk},
            )
        scored.append({
            "agent_id": agent.get("agent_id"),
            "utility": utility,
            "p_success": round(ur.p_success, 6),
            "mu": round(ur.mu, 4),
            "sigma": round(ur.sigma, 4),
            "model_cost": round(ur.model_cost, 6),
            "risk_penalty": round(ur.risk_penalty, 6),
            "vendor": ur.vendor,
            "reason": ur.reason,
            "eligible": math.isfinite(utility),
        })

    eligible = [s for s in scored if s["eligible"]]
    if not eligible:
        # Surface the dominant blocking reason so the caller sees WHY nothing won.
        block_reason = scored[0]["reason"] if scored else "NO_ELIGIBLE_AGENT"
        return {
            "winner": None, "utility": None, "reason": block_reason,
            "premium_applied": PREMIUM_AVAILABLE,
            "scores": sorted(scored, key=lambda s: s["agent_id"] or ""),
        }
    winner = max(eligible, key=lambda s: s["utility"])
    return {
        "winner": winner["agent_id"],
        "utility": round(winner["utility"], 6),
        "reason": winner["reason"],
        "p_success": winner["p_success"],
        "mu": winner["mu"],
        "sigma": winner["sigma"],
        "vendor": winner["vendor"],
        "premium_applied": PREMIUM_AVAILABLE,
        "scores": sorted(scored, key=lambda s: (-(s["utility"] if math.isfinite(s["utility"]) else -math.inf), s["agent_id"] or "")),
    }


def record_outcome_logic(agent_id: str, success: bool, event: str | None = None,
                         agent: dict | None = None, task: dict | None = None,
                         quality: float = 1.0) -> dict:
    """Apply the REAL v1-v5 HP + Bayesian-skill update for one observed outcome.

    Mirrors `ledger.apply_delta` (absolute-HP event ledger) and
    `ledger.update_domain_skill` (Elo/Bayesian mu/sigma law) EXACTLY -- the same
    `ledger.EVENTS` deltas, the same `_clamp_hp` / `_rank_for_hp` HP arithmetic, the
    same `k = clamp(sigma/2.7)` / `mu += k*q*(outcome-expected)` / sigma shrink-grow
    update, and the same `GUILLOTINE_HP` line. It is the STATELESS form: prior state
    is supplied via `agent` (hp + optional mu/sigma/n) instead of read from Mongo, so
    the caller (or a hosted store) owns persistence. Recomputes hazard via the real
    `ledger.calculate_hazard` on the post-update telemetry.

    `event` defaults to the canonical success/failure event when omitted:
      success -> 'ci_qa_pass' (+5 HP),  failure -> 'plausibility_fail' (-50 HP).
    A caller may pass any key in `ledger.EVENTS` to apply that specific delta.

    Returns: {hp, mu, sigma, hazard, guillotined, rank, delta, event, ...}.
    """
    a = _normalize_agent(agent or {"agent_id": agent_id})
    old_hp = _ledger._migrate_legacy_weight(a.get("hp", a.get("weight", _ledger.DEFAULT_HP)))

    # --- HP leg: identical to ledger.apply_delta's arithmetic ------------------
    ev = event or ("ci_qa_pass" if success else "plausibility_fail")
    delta = float(_ledger.EVENTS.get(ev, 0.0))
    new_hp = _ledger._clamp_hp(old_hp + delta)
    new_rank = _ledger._rank_for_hp(new_hp)
    guillotined = new_hp < _ledger.GUILLOTINE_HP

    # --- Skill leg: identical to ledger.update_domain_skill's Elo/Bayesian law --
    ticket = task or {"ticket_id": "outcome", "target_domain": a.get("domain", ""),
                      "priority": 3}
    prior_row = None
    skills = a.get("domain_skills") or {}
    if isinstance(skills, dict) and skills:
        # take the row for the ticket's skill if present, else any single row provided.
        key = _ledger._skill_name_for_ticket(ticket)
        prior_row = skills.get(key) or next(iter(skills.values()), None)
    default_row = _ledger._default_skill_row(a)
    row = dict(prior_row or default_row)
    mu = _f(row.get("mu"), default_row["mu"])
    sigma = _f(row.get("sigma"), default_row["sigma"])
    difficulty = _ledger._ticket_difficulty(ticket)
    expected = _ledger._skill_expected(mu, sigma, difficulty)
    outcome = 1.0 if success else 0.0
    q = max(0.25, min(1.75, _f(quality, 1.0)))
    k = max(2.0, min(12.0, sigma / 2.7))
    mu = max(0.0, min(100.0, mu + (k * q * (outcome - expected))))
    if success:
        sigma = max(5.0, sigma * (0.92 if expected < 0.9 else 0.96))
    else:
        sigma = min(60.0, sigma * 1.08 + 1.5)
    n = int(_f(row.get("n"), 0)) + 1

    # --- Hazard leg: the REAL ledger.calculate_hazard on post-update telemetry --
    snap = dict(_ledger.default_telemetry_snapshot())
    snap.update(a.get("telemetry_snapshot") or {})
    if not success:
        snap["streak_failed"] = int(_f(snap.get("streak_failed"), 0)) + 1
        snap["failed_requeue_count"] = int(_f(snap.get("failed_requeue_count"), 0)) + (
            1 if ev in _ledger.STREAK_RESET_EVENTS or delta < 0 else 0
        )
    hazard = _ledger.calculate_hazard(snap)
    hazard_rate = hazard["hazard_rate"]
    # OPEN/CLOSED: premium calibration of the hazard (hosted only).
    hazard_rate = _apply_premium_scalar(
        "calibrate_hazard", hazard_rate,
        {"agent_id": agent_id, "success": success, "event": ev, "new_hp": new_hp},
    )
    hazard_rate = round(max(0.0, min(1.0, hazard_rate)), 4)

    return {
        "agent_id": agent_id,
        "event": ev,
        "success": bool(success),
        "delta": delta,
        "old_hp": round(old_hp, 4),
        "hp": round(new_hp, 4),
        "rank": new_rank,
        "mu": round(mu, 4),
        "sigma": round(sigma, 4),
        "n": n,
        "hazard": hazard_rate,
        "soft_recycle_candidate": bool(hazard["soft_recycle_candidate"]),
        "guillotined": bool(guillotined),
        "premium_applied": PREMIUM_AVAILABLE,
    }


def _has_deterministic_oracle(a: dict) -> bool:
    """Whether a DETERMINISTIC ground-truth oracle (tests / invariant / compiler / an
    explicit deterministic-oracle flag) backs a verdict for this agent.

    Severity-matched authority (V1-V5 safety invariant): only a deterministic oracle may
    authorize an irreversible HARD action (guillotine). High hazard or low HP alone, with
    no deterministic backing, may only RECYCLE/HOLD -- never auto-kill. The signal is read
    from the agent's optional `oracle`/`task` descriptor; absence means no hard action.
    """
    src = a.get("oracle") or a.get("task") or {}
    if not isinstance(src, dict):
        return False
    return bool(src.get("has_deterministic_oracle") or src.get("has_tests")
                or src.get("has_invariant") or src.get("has_compiler"))


def survival_check_logic(agent: dict) -> dict:
    """Read-only survival verdict for one agent: should it be killed, recycled, or kept?

    Combines the two real survival signals WITHOUT mutating anything:
      * the absolute-HP guillotine line (`ledger.GUILLOTINE_HP`) and soft-recycle band
        (`ledger.SOFT_RECYCLE_HP`),
      * the V3 hazard from `ledger.calculate_hazard` against the soft-recycle /
        guillotine hazard thresholds,
      * severity-matched authority: a HARD action (guillotine) is only ALLOWED when a
        deterministic oracle backs the verdict -- so a high hazard or low HP with no
        deterministic oracle can only RECYCLE/HOLD, never auto-kill.

    Verdicts: GUILLOTINE (HP below line / hazard critical, deterministic oracle present),
              SOFT_RECYCLE (in the recycle band or high hazard),
              HOLD (HP below line / hazard critical but no deterministic oracle -> committee),
              SURVIVE (healthy).
    """
    a = _normalize_agent(agent or {})
    hp = _ledger._migrate_legacy_weight(a.get("hp", a.get("weight", _ledger.DEFAULT_HP)))
    snap = dict(_ledger.default_telemetry_snapshot())
    snap.update(a.get("telemetry_snapshot") or {})
    hazard = _ledger.calculate_hazard(snap)
    hazard_rate = _apply_premium_scalar(
        "calibrate_hazard", hazard["hazard_rate"], {"hp": hp, "phase": "survival_check"},
    )
    hazard_rate = round(max(0.0, min(1.0, hazard_rate)), 4)

    # severity-matched authority: is a deterministic (hard-action-permitting) oracle present?
    hard_ok = _has_deterministic_oracle(a)

    below_line = hp < _ledger.GUILLOTINE_HP
    in_recycle_band = _ledger.GUILLOTINE_HP <= hp < _ledger.SOFT_RECYCLE_HP
    hazard_guillotine = hazard_rate >= _ledger.HAZARD_GUILLOTINE_THRESHOLD
    hazard_recycle = hazard_rate >= _ledger.HAZARD_SOFT_RECYCLE_THRESHOLD

    if (below_line or hazard_guillotine) and hard_ok:
        verdict = "GUILLOTINE"
        reason = ("hp<20 (deterministic oracle authorizes hard action)" if below_line
                  else "hazard>=guillotine line (deterministic oracle authorizes)")
    elif below_line or hazard_guillotine:
        # high-severity signal but NO deterministic oracle -> may not auto-kill.
        verdict = "HOLD"
        reason = ("hp<20 / hazard critical but no deterministic oracle -> route to "
                  "committee (severity-matched authority forbids auto-guillotine)")
    elif in_recycle_band or hazard_recycle:
        verdict = "SOFT_RECYCLE"
        reason = ("hp in [20,40) recycle band" if in_recycle_band
                  else "hazard>=soft-recycle threshold")
    else:
        verdict = "SURVIVE"
        reason = "hp healthy and hazard below thresholds"

    return {
        "agent_id": a.get("agent_id"),
        "verdict": verdict,
        "hp": round(hp, 4),
        "hazard": hazard_rate,
        "reason": reason,
        "hard_action_allowed": hard_ok,
        "premium_applied": PREMIUM_AVAILABLE,
    }


def model_route_logic(task: dict, disk_free: float | None = None,
                      tier_credit: dict | None = None) -> dict:
    """Pick the model tier (opus / sonnet / haiku) for `task` via the REAL data-driven
    auction, with NO MongoDB.

    `model_router.route` is async + Mongo-backed; here we drive the SAME decision
    deterministically: for each tier we build the router's tier-agent descriptor
    (`model_router._model_for_tier`), inject the CURRENT measured credit (from the
    optional `tier_credit` override, else the router's grounded cold-start prior), and
    score it on the dispatcher's own `utility_for` -- exactly what the live router does.
    The V2 money/security HARD-GATE (-> opus only) and the cold-start difficulty floor
    are applied identically. On a hosted deployment the premium overlay calibrates the
    per-tier scores (open baseline: pass-through).

    `tier_credit`: optional {tier: {mu, sigma, n}} measured credit by tier (the caller's
    learned state). Absent -> the router's grounded PRIOR per tier for the task's band.

    Returns: {tier, model, reason, money_critical, difficulty_band, scores, ...}.
    """
    if not isinstance(task, dict):
        return {"error": "task must be an object"}
    free = _disk_free_arg(disk_free)
    ticket = _model_router._ticket_from_item(task) if not task.get("target_domain") and "slug" in task else dict(task)
    # ensure the fields the router/dispatcher read are present with sane defaults.
    ticket.setdefault("ticket_id", task.get("slug", task.get("ticket_id", "task")))
    ticket.setdefault("target_domain", task.get("target_domain", ""))
    diff = _f(ticket.get("ci_gate_difficulty", ticket.get("difficulty")), 50.0)
    ticket["ci_gate_difficulty"] = diff
    ticket.setdefault("difficulty", diff)
    domain = str(ticket["target_domain"])
    band = _model_router.difficulty_band(diff)

    money_critical = (
        domain in _model_router._ELITE_FORCED
        or _f(ticket.get("money_risk"), 0.0) >= 3.0
        or _f(ticket.get("security_risk"), 0.0) >= 3.0
    )

    skill_key = _dispatcher._skill_name(domain)
    overrides = tier_credit or {}
    scored: list[dict] = []
    for tier in _model_router._TIERS:
        prior = _model_router._PRIOR_MU[tier]
        credit = overrides.get(tier) or {}
        mu = _f(credit.get("mu"), prior)
        sigma = _f(credit.get("sigma"), _model_router.PRIOR_SIGMA)
        n = int(_f(credit.get("n"), 0))
        agent = _model_router._model_for_tier(tier)
        row = {"mu": mu, "sigma": sigma, "n": n}
        agent["domain_skills"] = {skill_key: dict(row), domain: dict(row)}
        ur = _dispatcher.utility_for(ticket, agent, disk_free=free, db=None)
        p_succ = _model_router._p_success(mu, sigma, diff)
        utility = ur.utility

        eligible = True
        reason = "auction"
        if money_critical and tier != "opus":
            eligible = False
            reason = "money/security gate -> opus only"
        elif n == 0 and diff > _model_router._COLDSTART_MAX_DIFFICULTY[tier]:
            eligible = False
            reason = "cold-start floor (unmeasured tier below difficulty band)"
        scored.append({
            "tier": tier, "model": tier, "utility": utility, "p_success": p_succ,
            "mu": mu, "sigma": sigma, "n": n, "eligible": eligible, "reason": reason,
        })

    # OPEN/CLOSED: premium calibration of the tier scores (hosted only).
    fn = _premium_hook("calibrate_tier_scores")
    if fn is not None:
        try:  # pragma: no cover - hosted overlay only
            refined = fn({s["tier"]: s["utility"] for s in scored},
                         {"domain": domain, "band": band, "money_critical": money_critical})
            if isinstance(refined, dict):
                for s in scored:
                    if s["tier"] in refined and math.isfinite(_f(refined[s["tier"]], math.nan)):
                        s["utility"] = float(refined[s["tier"]])
        except Exception:
            pass

    eligible = [s for s in scored if s["eligible"] and math.isfinite(s["utility"])]
    if not eligible:
        pick = max(scored, key=lambda s: (s["utility"] if math.isfinite(s["utility"]) else -math.inf))
        pick = dict(pick)
        pick["reason"] = "no eligible tier (fallback to strongest)"
    else:
        pick = max(eligible, key=lambda s: s["utility"])

    return {
        "slug": task.get("slug", ticket.get("ticket_id")),
        "tier": pick["tier"],
        "model": pick["model"],
        "money_critical": money_critical,
        "difficulty_band": band,
        "reason": "money/security gate -> opus" if money_critical else pick["reason"],
        "mu_used": round(pick["mu"], 4),
        "sigma_used": round(pick["sigma"], 4),
        "n": pick["n"],
        "p_success": round(pick["p_success"], 4),
        "utility": round(pick["utility"], 4) if math.isfinite(pick["utility"]) else None,
        "premium_applied": PREMIUM_AVAILABLE,
        "scores": {s["tier"]: (round(s["utility"], 3) if math.isfinite(s["utility"]) else None)
                   for s in scored},
    }


def fleet_snapshot_logic(fleet: list[dict]) -> dict:
    """Summarize a fleet of agents: HP/rank distribution, survival verdicts, and the
    REAL vendor-concentration check on in-flight critical work.

    For each agent we run `survival_check_logic` (no mutation) and tally verdicts. The
    vendor concentration uses `risk_core.concentration_check` (the same coherent cap the
    live auction enforces) over the agents' declared vendors/models so an operator can
    see at a glance whether one vendor is over-represented.

    Returns a summary dict (counts, mean HP/hazard, at-risk list, concentration).
    """
    if not fleet:
        return {"count": 0, "summary": "empty fleet", "premium_applied": PREMIUM_AVAILABLE}

    verdict_counts: dict[str, int] = {}
    hps: list[float] = []
    hazards: list[float] = []
    at_risk: list[dict] = []
    vendors: list[str] = []
    for raw in fleet:
        sc = survival_check_logic(raw)
        verdict_counts[sc["verdict"]] = verdict_counts.get(sc["verdict"], 0) + 1
        hps.append(sc["hp"])
        hazards.append(sc["hazard"])
        if sc["verdict"] in ("GUILLOTINE", "SOFT_RECYCLE", "HOLD"):
            at_risk.append({"agent_id": sc["agent_id"], "verdict": sc["verdict"],
                            "hp": sc["hp"], "hazard": sc["hazard"]})
        vendors.append(_dispatcher._agent_vendor(_normalize_agent(raw)))

    concentration = _risk_core.concentration_check(vendors) if vendors else {"error": "no vendors"}
    n = len(fleet)
    return {
        "count": n,
        "verdicts": verdict_counts,
        "mean_hp": round(sum(hps) / n, 4) if n else 0.0,
        "mean_hazard": round(sum(hazards) / n, 4) if n else 0.0,
        "at_risk_count": len(at_risk),
        "at_risk": sorted(at_risk, key=lambda x: x["hazard"], reverse=True)[:25],
        "vendor_concentration": concentration,
        "deployment": deployment_mode(),
        "premium_applied": PREMIUM_AVAILABLE,
    }


# =========================================================================== #
# MCP TRANSPORT (import-guarded; only wired when the official `mcp` SDK exists)
# =========================================================================== #
try:  # pragma: no cover - exercised only when the mcp SDK is installed
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except Exception:
    FastMCP = None  # type: ignore
    MCP_AVAILABLE = False


def build_server():  # pragma: no cover - requires the mcp SDK
    """Construct the FastMCP server with the five governance tools wired to the logic
    functions above. Raises RuntimeError when the `mcp` SDK is not importable."""
    if not MCP_AVAILABLE:
        raise RuntimeError(
            "The 'mcp' package (FastMCP) is not installed. Install with "
            "`pip install mcp` to run the transport; the *_logic functions remain "
            "importable and testable without it."
        )
    mcp = FastMCP("quantchoai-governor")

    @mcp.tool()
    def govern_route(task: dict, agents: list[dict]) -> dict:
        """Auction `task` across `agents`; return the utility-maximizing winner."""
        return govern_route_logic(task, agents)

    @mcp.tool()
    def record_outcome(agent_id: str, success: bool, event: str | None = None,
                       agent: dict | None = None, task: dict | None = None) -> dict:
        """Apply the v1-v5 HP + Bayesian-skill + hazard update for one outcome."""
        return record_outcome_logic(agent_id, success, event=event, agent=agent, task=task)

    @mcp.tool()
    def survival_check(agent: dict) -> dict:
        """Should this agent be guillotined, recycled, held, or kept? (read-only)"""
        return survival_check_logic(agent)

    @mcp.tool()
    def model_route(task: dict) -> dict:
        """Pick the model tier (opus/sonnet/haiku) for `task` via the v1-v5 auction."""
        return model_route_logic(task)

    @mcp.tool()
    def fleet_snapshot(fleet: list[dict]) -> dict:
        """Summarize a fleet: HP/hazard distribution, survival verdicts, vendor concentration."""
        return fleet_snapshot_logic(fleet)

    @mcp.tool()
    def deployment_info() -> dict:
        """Report hosted-full vs local-open-baseline deployment mode."""
        return deployment_mode()

    return mcp


def main() -> None:  # pragma: no cover - process entrypoint
    """Run the MCP server.

    Transport is selected by QCHO_MCP_TRANSPORT (default 'stdio' for a local
    `claude_desktop_config.json` stdio launch). A hosted deployment sets it to
    'streamable-http' (or 'sse') and runs behind the operator's own auth/proxy so
    the premium overlay stays server-side.
    """
    server = build_server()
    transport = os.environ.get("QCHO_MCP_TRANSPORT", "stdio")
    server.run(transport=transport)


if __name__ == "__main__":  # pragma: no cover
    main()
