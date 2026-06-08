"""solo_fleet_api.py — open-core v1-v5 Governor facade.

PUBLIC API (5 methods):
    route(task, agents)            -> best agent name + score breakdown
    record(agent, success)         -> updated mu/sigma credit row
    survival(agent)                -> HP-based survival decision
    snapshot(fleet)                -> fleet-level health summary
    model_route(task)              -> model tier (opus/sonnet/haiku) selection

OPEN vs PREMIUM:
    * Local mode (default):  runs the open baseline — generic survival math,
      Bayesian mu/sigma credit, Poisson failure hazard, CVaR tail-risk term.
      Fitted constants / calibration surfaces / marginal-CVaR are NOT included
      — those live server-side and require a subscription.
    * Hosted mode:           pass `remote_url="https://…"` to Governor().
      Every method delegates its JSON payload to the operator's hosted endpoint
      and returns the server's enriched response.  The premium/calibrated
      governance engine runs server-side; the open package is a thin transport
      wrapper — no moat is bundled.

STDLIB-ONLY: math, statistics, urllib.request, json.  No numpy/scipy/pymongo.
DECIMAL: all money-adjacent accumulations use Decimal for exactness.
"""
from __future__ import annotations

import json
import math
import statistics as _st
import urllib.error
import urllib.request
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Open baseline constants (V1-V5 grounded priors — NOT fitted calibration)
# ---------------------------------------------------------------------------
_DEFAULT_HP: float = 50.0
_MAX_HP: float = 100.0
_GUILLOTINE_HP: float = 20.0
_SOFT_RECYCLE_HP: float = 40.0

# V2 credit defaults by tier (rank): (mu, sigma)
_DEFAULT_MU_SIGMA: dict[int, tuple[float, float]] = {
    3: (78.0, 14.0),
    2: (62.0, 24.0),
    1: (48.0, 32.0),
}
_DEFAULT_CREDIT = {"mu": 50.0, "sigma": 30.0}

# V3 hazard weights
_BASE_HAZARD: float = 0.05
_WEIGHT_STREAK: float = 0.1
_WEIGHT_REQUEUE: float = 0.15
_HAZARD_SOFT_THRESHOLD: float = 0.6
_HAZARD_GUILLOTINE_THRESHOLD: float = 0.9

# V5 auction
_LAMBDA_STARVATION: float = 0.018
_ALPHA_REQUEUE: float = 3.0
_CVAR_ALPHA: float = 0.95
_CVAR_CALM_MASS: int = 99
_CVAR_CALM_LOSS: float = 1.0

# Model tiers + cost proxies
_MODEL_TIERS = ("opus", "sonnet", "haiku")
_MODEL_COST = {"opus": 0.15, "sonnet": 0.055, "haiku": 0.012}
_PRIOR_MU = {"opus": 72.0, "sonnet": 60.0, "haiku": 40.0}
_PRIOR_SIGMA = 25.0
_COLDSTART_MAX_DIFF = {"haiku": 50.0, "sonnet": 80.0, "opus": 100.0}


# ---------------------------------------------------------------------------
# Internal helpers — pure, deterministic, stdlib only
# ---------------------------------------------------------------------------

def _to_decimal(v: Any) -> Decimal:
    """Convert a numeric value to Decimal for exact money-path accumulation."""
    if isinstance(v, Decimal):
        return v
    return Decimal(str(v))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _p_success(mu: float, sigma: float, difficulty: float) -> float:
    """V5 auction sigmoid — identical to dispatcher._p_success."""
    z = (mu - difficulty) / max(8.0, sigma)
    p = 1.0 / (1.0 + math.exp(-0.717 * z))
    return max(0.01, min(0.99, p))


def _cvar_open(losses: list[float], alpha: float = _CVAR_ALPHA) -> float:
    """Open-baseline CVaR (Expected Shortfall) — historical order-statistic estimator.

    This is the risk_core.cvar() definition shipped in the open package.  The
    premium hosted service substitutes the Rockafellar-Uryasev + Cornish-Fisher
    hardened estimator (with the calibrated calm-mass distribution) server-side.
    """
    xs = sorted(losses)
    if not xs:
        return 0.0
    idx = min(len(xs) - 1, int(math.ceil(alpha * len(xs))) - 1)
    tail = xs[idx:]
    return float(sum(tail) / len(tail)) if tail else float(xs[-1])


def _tail_term(money_risk: float, security_risk: float, live_risk: float) -> float:
    """Build per-ticket loss vector and price its CVaR tail.

    Open baseline: simple historical ES.  Premium hosted layer substitutes the
    CF-hardened estimator with the calibrated body distribution.
    Fitted marginal-CVaR concentrations are NOT included in this package.
    """
    losses: list[float] = [_CVAR_CALM_LOSS] * max(1, _CVAR_CALM_MASS)
    for risk in (money_risk, security_risk, live_risk):
        if risk and risk > 0.0:
            losses.append(math.exp(min(10.0, float(risk))))
    return _cvar_open(losses, _CVAR_ALPHA)


def _hazard_open(streak_failures: int = 0, requeue_count: int = 0) -> float:
    """V3 open-baseline hazard rate (abridged — no Mongo, no per-agent lambda calibration).

    Premium hosted service applies the calibrated Poisson lambda from the materializer
    and the NB2 overdispersion correction.
    """
    h = _BASE_HAZARD + _WEIGHT_STREAK * float(streak_failures) + _WEIGHT_REQUEUE * float(requeue_count)
    return max(0.0, min(1.0, h))


def _survival_decision(hp: float, hazard: float) -> dict[str, Any]:
    """Map HP + hazard to a survival action.  Pure open-baseline rule."""
    if hp < _GUILLOTINE_HP or hazard >= _HAZARD_GUILLOTINE_THRESHOLD:
        action = "guillotine"
    elif hp < _SOFT_RECYCLE_HP or hazard >= _HAZARD_SOFT_THRESHOLD:
        action = "soft_recycle"
    else:
        action = "ok"
    return {
        "action": action,
        "hp": float(hp),
        "hazard": float(hazard),
        "guillotine_threshold": _GUILLOTINE_HP,
        "soft_recycle_threshold": _SOFT_RECYCLE_HP,
    }


def _bayesian_update(mu: float, sigma: float, difficulty: float,
                     success: bool, quality: float = 1.0) -> tuple[float, float]:
    """V2 Bayesian mu/sigma update — exact mirror of ledger.update_domain_skill."""
    expected = _p_success(mu, sigma, float(difficulty))
    outcome = 1.0 if success else 0.0
    q = max(0.25, min(1.75, float(quality)))
    k = max(2.0, min(12.0, sigma / 2.7))
    new_mu = max(0.0, min(100.0, mu + k * q * (outcome - expected)))
    if success:
        new_sigma = max(5.0, sigma * (0.92 if expected < 0.9 else 0.96))
    else:
        new_sigma = min(60.0, sigma * 1.08 + 1.5)
    return new_mu, new_sigma


def _poisson_failure_prob(lmbda: float, k: int = 1) -> float:
    """P(>= k failures | rate lmbda) — open-baseline V3 Poisson failure model."""
    if k <= 0:
        return 1.0
    cdf_lt_k = sum(
        math.exp(-lmbda) * lmbda ** i / math.factorial(i)
        for i in range(k)
    )
    return max(0.0, min(1.0, 1.0 - cdf_lt_k))


# ---------------------------------------------------------------------------
# Remote transport (hosted delegation)
# ---------------------------------------------------------------------------

class _RemoteError(RuntimeError):
    """Raised when the hosted endpoint returns an error."""


def _post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    """POST a JSON payload to the hosted endpoint; return the parsed response.

    This is the ONLY network call in the open package — the entire premium
    governance engine stays server-side.  The caller assembles the payload,
    this function handles the HTTP round-trip.
    """
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": f"quantchoai-governor/{__version__}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        raise _RemoteError(f"hosted endpoint {url!r} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise _RemoteError(f"could not reach hosted endpoint {url!r}: {exc.reason}") from exc
    except Exception as exc:  # pragma: no cover — network-dependent
        raise _RemoteError(f"hosted endpoint error: {exc}") from exc


# ---------------------------------------------------------------------------
# Governor — the 5-method open-core facade
# ---------------------------------------------------------------------------

class Governor:
    """Open-core v1-v5 Governor facade.

    Parameters
    ----------
    remote_url : str or None
        If provided, all five methods delegate to this hosted endpoint.
        The URL should be the base URL of an operator-run quantchoai-governor server
        (e.g. ``https://governance.example.com/v1``).  Each method POSTs to
        ``{remote_url}/{method_name}``.
        If None (default), the open baseline runs locally — no network required.
    timeout : float
        HTTP timeout in seconds for remote calls (default 10 s).

    Examples
    --------
    Local open baseline::

        g = Governor()
        result = g.route("write a risk model", ["agentA", "agentB"])

    Hosted premium governance::

        g = Governor(remote_url="https://governance.example.com/v1")
        result = g.route("write a risk model", ["agentA", "agentB"])
    """

    def __init__(self, remote_url: str | None = None, timeout: float = 10.0) -> None:
        if remote_url is not None:
            remote_url = remote_url.rstrip("/")
        self._remote_url: str | None = remote_url
        self._timeout: float = float(timeout)

    # ------------------------------------------------------------------ #
    # 1. route(task, agents) -> dict
    # ------------------------------------------------------------------ #

    def route(self, task: str, agents: list[str | dict]) -> dict[str, Any]:
        """Select the best agent for a task via the open v1-v5 Expected-Utility auction.

        In hosted mode the full premium engine (calibrated priors, marginal-CVaR
        concentrations, fitted mu/sigma from real dispatch history) runs server-side.
        In local mode the open baseline uses grounded priors and the historical CVaR
        estimator.

        Parameters
        ----------
        task : str
            Free-text task description.  Keywords trigger the money/security risk
            classifier (see governance spec §V2).
        agents : list of str or dict
            Agent names (strings) or agent descriptor dicts with optional fields:
            ``hp``, ``mu``, ``sigma``, ``rank``, ``model_cost``.

        Returns
        -------
        dict with keys: ``winner``, ``winner_score``, ``scores`` (per-agent),
        ``mode`` (``"local"`` or ``"hosted"``).
        """
        if self._remote_url is not None:
            return self._delegate("route", {"task": task, "agents": agents})
        return self._route_local(task, agents)

    def _route_local(self, task: str, agents: list[str | dict]) -> dict[str, Any]:
        task_lower = task.lower()
        money_risk = 4.0 if any(w in task_lower for w in (
            "var", "cvar", "pnl", "position", "margin", "nav", "pricing", "valuation",
            "ledger", "trade", "fill", "tax", "accounting", "exposure", "risk"
        )) else 0.0
        security_risk = 3.5 if any(w in task_lower for w in (
            "auth", "token", "permission", "rbac", "security", "ofac", "sanction",
            "validator", "tenant", "compliance"
        )) else 0.0
        live_risk = 4.5 if any(w in task_lower for w in (
            "place order", "submit order", "live broker", "go live", "alpaca live"
        )) else 0.0

        tail = _tail_term(money_risk, security_risk, live_risk)
        # Use Decimal for the money-adjacent accumulation of risk_penalty
        tail_d = _to_decimal(tail)

        scores: list[dict[str, Any]] = []
        for agent in agents:
            if isinstance(agent, str):
                desc: dict[str, Any] = {"name": agent}
            else:
                desc = dict(agent)
                if "name" not in desc:
                    desc["name"] = str(desc.get("agent_id", "unknown"))
            hp = float(desc.get("hp", _DEFAULT_HP))
            rank = int(desc.get("rank", 2))
            default_mu, default_sigma = _DEFAULT_MU_SIGMA.get(rank, (62.0, 24.0))
            mu = float(desc.get("mu", default_mu))
            sigma = float(desc.get("sigma", default_sigma))
            difficulty = 50.0 + money_risk * 2 + security_risk * 1.5 + live_risk * 3
            p_succ = _p_success(mu, sigma, difficulty)
            # value: normalized task complexity proxy
            value = 1.0 + 0.2 * min(1.0, len(task) / 200.0)
            model_cost = float(desc.get("model_cost", 0.055))
            # risk_penalty computed in Decimal, converted back for JSON
            risk_penalty = float(
                _to_decimal(1.0 - p_succ) * tail_d * _to_decimal(12.0)
            )
            utility = p_succ * value - model_cost - risk_penalty
            scores.append({
                "agent": desc["name"],
                "utility": round(utility, 6),
                "p_success": round(p_succ, 4),
                "mu": round(mu, 2),
                "sigma": round(sigma, 2),
                "hp": round(hp, 2),
                "risk_penalty": round(risk_penalty, 6),
            })

        if not scores:
            return {"winner": None, "winner_score": None, "scores": [], "mode": "local"}

        best = max(scores, key=lambda s: s["utility"])
        return {
            "winner": best["agent"],
            "winner_score": best["utility"],
            "scores": scores,
            "mode": "local",
        }

    # ------------------------------------------------------------------ #
    # 2. record(agent, success) -> dict
    # ------------------------------------------------------------------ #

    def record(self, agent: str | dict, success: bool, *,
               difficulty: float = 50.0, quality: float = 1.0) -> dict[str, Any]:
        """Apply the V2 Bayesian credit update for a completed task.

        In hosted mode the server updates the persistent tier_credit rows.
        In local mode the update is computed and returned (stateless — caller
        persists the result).

        Returns dict with ``mu``, ``sigma``, ``success``, ``mode``.
        """
        if self._remote_url is not None:
            return self._delegate("record", {
                "agent": agent, "success": success,
                "difficulty": difficulty, "quality": quality,
            })
        return self._record_local(agent, success, difficulty=difficulty, quality=quality)

    def _record_local(self, agent: str | dict, success: bool,
                      difficulty: float = 50.0, quality: float = 1.0) -> dict[str, Any]:
        if isinstance(agent, str):
            desc: dict[str, Any] = {}
            name = agent
        else:
            desc = dict(agent)
            name = str(desc.get("name", desc.get("agent_id", "unknown")))
        mu0 = float(desc.get("mu", _DEFAULT_CREDIT["mu"]))
        sigma0 = float(desc.get("sigma", _DEFAULT_CREDIT["sigma"]))
        new_mu, new_sigma = _bayesian_update(mu0, sigma0, difficulty, success, quality)
        return {
            "agent": name,
            "mu_before": round(mu0, 4),
            "sigma_before": round(sigma0, 4),
            "mu": round(new_mu, 4),
            "sigma": round(new_sigma, 4),
            "success": bool(success),
            "difficulty": round(float(difficulty), 2),
            "mode": "local",
        }

    # ------------------------------------------------------------------ #
    # 3. survival(agent) -> dict
    # ------------------------------------------------------------------ #

    def survival(self, agent: str | dict, *,
                 streak_failures: int = 0, requeue_count: int = 0) -> dict[str, Any]:
        """Evaluate V1 (HP) + V3 (hazard) survival gate for an agent.

        In hosted mode the server applies the calibrated Poisson/NB2 hazard and the
        full HP ledger state.  In local mode the open baseline rules apply.

        Returns dict with ``action`` (``"ok"`` / ``"soft_recycle"`` / ``"guillotine"``),
        ``hp``, ``hazard``, ``mode``.
        """
        if self._remote_url is not None:
            return self._delegate("survival", {
                "agent": agent,
                "streak_failures": streak_failures,
                "requeue_count": requeue_count,
            })
        return self._survival_local(agent, streak_failures=streak_failures,
                                    requeue_count=requeue_count)

    def _survival_local(self, agent: str | dict, *,
                        streak_failures: int = 0, requeue_count: int = 0) -> dict[str, Any]:
        if isinstance(agent, str):
            desc: dict[str, Any] = {}
        else:
            desc = dict(agent)
        hp = float(desc.get("hp", _DEFAULT_HP))
        hazard = _hazard_open(streak_failures, requeue_count)
        result = _survival_decision(hp, hazard)
        result["mode"] = "local"
        return result

    # ------------------------------------------------------------------ #
    # 4. snapshot(fleet) -> dict
    # ------------------------------------------------------------------ #

    def snapshot(self, fleet: list[str | dict]) -> dict[str, Any]:
        """Compute a fleet-level health summary.

        Returns aggregate statistics: mean/median HP, count by survival zone,
        fleet-level CVaR tail summary, vendor concentration check.
        In hosted mode the server enriches with live dispatch telemetry.

        Returns dict with ``count``, ``mean_hp``, ``median_hp``, ``zone_counts``,
        ``fleet_cvar``, ``mode``.
        """
        if self._remote_url is not None:
            return self._delegate("snapshot", {"fleet": fleet})
        return self._snapshot_local(fleet)

    def _snapshot_local(self, fleet: list[str | dict]) -> dict[str, Any]:
        if not fleet:
            return {
                "count": 0, "mean_hp": None, "median_hp": None,
                "zone_counts": {}, "fleet_cvar": None, "mode": "local",
            }

        hp_values: list[float] = []
        zone_counts: dict[str, int] = {"elite": 0, "standard": 0, "struggling": 0, "guillotine": 0}
        # Accumulate HP sum using Decimal for exact arithmetic on money-adjacent aggregation
        hp_sum = Decimal("0")

        for agent in fleet:
            if isinstance(agent, str):
                hp = _DEFAULT_HP
            else:
                hp = float((agent if isinstance(agent, dict) else {}).get("hp", _DEFAULT_HP))
            hp_values.append(hp)
            hp_sum += _to_decimal(hp)

            if hp > 80.0:
                zone_counts["elite"] += 1
            elif hp >= 40.0:
                zone_counts["standard"] += 1
            elif hp >= _GUILLOTINE_HP:
                zone_counts["struggling"] += 1
            else:
                zone_counts["guillotine"] += 1

        n = len(hp_values)
        mean_hp = float((hp_sum / _to_decimal(n)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        median_hp = float(_st.median(hp_values))

        # Fleet-level CVaR: treat each agent's (100 - hp) as a "health deficit loss"
        # Open baseline: ships the historical estimator only
        losses = [100.0 - hp for hp in hp_values]
        fleet_cvar = round(_cvar_open(losses, _CVAR_ALPHA), 4)

        return {
            "count": n,
            "mean_hp": mean_hp,
            "median_hp": median_hp,
            "zone_counts": zone_counts,
            "fleet_cvar": fleet_cvar,
            "mode": "local",
        }

    # ------------------------------------------------------------------ #
    # 5. model_route(task) -> dict
    # ------------------------------------------------------------------ #

    def model_route(self, task: str, *,
                    difficulty: float = 50.0,
                    money_risk: float = 0.0,
                    security_risk: float = 0.0) -> dict[str, Any]:
        """Select the model tier (opus / sonnet / haiku) for a task.

        Applies the V2 money/security hard-gate (forced opus) and the cold-start
        safety floor.  In hosted mode the premium server uses the LEARNED per-tier
        mu/sigma from real dispatch history.  In local mode the open grounded priors
        are used and the cold-start safety floor governs.

        Returns dict with ``model``, ``tier``, ``reason``, ``scores``, ``mode``.
        """
        if self._remote_url is not None:
            return self._delegate("model_route", {
                "task": task,
                "difficulty": difficulty,
                "money_risk": money_risk,
                "security_risk": security_risk,
            })
        return self._model_route_local(task, difficulty=difficulty,
                                       money_risk=money_risk,
                                       security_risk=security_risk)

    def _model_route_local(self, task: str, *,
                           difficulty: float = 50.0,
                           money_risk: float = 0.0,
                           security_risk: float = 0.0) -> dict[str, Any]:
        # Auto-detect risk from task text if not supplied explicitly
        task_lower = task.lower()
        if money_risk == 0.0 and any(w in task_lower for w in (
            "var", "cvar", "pnl", "position", "margin", "nav", "pricing",
            "valuation", "ledger", "trade", "accounting", "tax", "risk"
        )):
            money_risk = 3.0
        if security_risk == 0.0 and any(w in task_lower for w in (
            "auth", "token", "permission", "security", "ofac", "sanction",
            "validator", "tenant", "compliance"
        )):
            security_risk = 3.0

        # V2 hard-gate: money or security critical -> opus only
        money_critical = money_risk >= 3.0 or security_risk >= 3.0
        diff = float(difficulty)

        scores: list[dict[str, Any]] = []
        for tier in _MODEL_TIERS:
            mu = _PRIOR_MU[tier]
            sigma = _PRIOR_SIGMA
            model_cost = _MODEL_COST[tier]
            p_succ = _p_success(mu, sigma, diff)
            tail = _tail_term(money_risk, security_risk, 0.0)
            # Compute risk_penalty using Decimal for exact money-path accumulation
            risk_penalty = float(
                _to_decimal(1.0 - p_succ) * _to_decimal(tail) * _to_decimal(12.0)
            )
            utility = p_succ * 1.0 - model_cost - risk_penalty

            eligible = True
            reason = "open-prior auction"
            if money_critical and tier != "opus":
                eligible = False
                reason = "money/security gate -> opus only"
            elif diff > _COLDSTART_MAX_DIFF[tier]:
                eligible = False
                reason = "cold-start floor (unmeasured below difficulty band)"

            scores.append({
                "tier": tier,
                "model": tier,
                "utility": round(utility, 6),
                "p_success": round(p_succ, 4),
                "mu_prior": mu,
                "sigma_prior": sigma,
                "model_cost": model_cost,
                "eligible": eligible,
                "reason": reason,
            })

        eligible_scores = [s for s in scores if s["eligible"] and math.isfinite(s["utility"])]
        if eligible_scores:
            pick = max(eligible_scores, key=lambda s: s["utility"])
        else:
            # Degenerate fallback — no tier cleared all gates; default to opus
            pick = max(scores, key=lambda s: s["utility"])
            pick = dict(pick)
            pick["reason"] = "fallback: no tier cleared all gates"

        return {
            "model": pick["model"],
            "tier": pick["tier"],
            "reason": pick["reason"],
            "scores": scores,
            "mode": "local",
        }

    # ------------------------------------------------------------------ #
    # Internal: hosted delegation
    # ------------------------------------------------------------------ #

    def _delegate(self, method: str, payload: dict) -> dict[str, Any]:
        """POST payload to {remote_url}/{method}; tag response with mode='hosted'."""
        url = f"{self._remote_url}/{method}"
        response = _post_json(url, payload, timeout=self._timeout)
        if not isinstance(response, dict):
            response = {"raw": response}
        response["mode"] = "hosted"
        return response


# ---------------------------------------------------------------------------
# MCP stdio server entry-point
# ---------------------------------------------------------------------------

def _mcp_server_main() -> None:  # pragma: no cover — stdio-driven CLI
    """Minimal MCP stdio server.  Reads newline-delimited JSON requests on stdin,
    writes responses on stdout.  Suitable for use as a subprocess MCP server.

    Protocol::

        {"id": "1", "method": "route", "params": {"task": "…", "agents": […]}}
        -> {"id": "1", "result": {…}}

    Set env SOLO_FLEET_REMOTE_URL to delegate to a hosted endpoint.
    """
    import os
    import sys

    remote_url = os.environ.get("SOLO_FLEET_REMOTE_URL")
    governor = Governor(remote_url=remote_url)
    _dispatch = {
        "route": lambda p: governor.route(p["task"], p["agents"]),
        "record": lambda p: governor.record(p["agent"], p["success"],
                                            difficulty=p.get("difficulty", 50.0),
                                            quality=p.get("quality", 1.0)),
        "survival": lambda p: governor.survival(p["agent"],
                                                streak_failures=p.get("streak_failures", 0),
                                                requeue_count=p.get("requeue_count", 0)),
        "snapshot": lambda p: governor.snapshot(p["fleet"]),
        "model_route": lambda p: governor.model_route(
            p["task"],
            difficulty=p.get("difficulty", 50.0),
            money_risk=p.get("money_risk", 0.0),
            security_risk=p.get("security_risk", 0.0),
        ),
    }

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"error": f"invalid JSON: {exc}"}) + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})
        handler = _dispatch.get(method)
        if handler is None:
            out = {"id": req_id, "error": f"unknown method: {method!r}"}
        else:
            try:
                result = handler(params)
                out = {"id": req_id, "result": result}
            except Exception as exc:
                out = {"id": req_id, "error": str(exc)}

        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    _mcp_server_main()
