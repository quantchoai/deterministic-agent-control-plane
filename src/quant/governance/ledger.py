"""ledger.py - absolute HP ledger for QuantChoAI agents.

The persisted Mongo field remains `weight` for compatibility with board.py and
existing queries, but it now means absolute HP, not a normalized fraction.

Reliability (v1-v5 patch):
  - BUFFERED LOG WRITES: system_logs / agent_events inserts are batched through an
    in-process deque, flushed every ~5 s or at size >= 50, via a background daemon
    thread that calls insert_many. flush() and atexit-on-exit are provided.
  - maxPoolSize raised from 2 to 8 (was a dispatch bottleneck under concurrent lanes).
  - IDEMPOTENCY: decay_idle carries a per-agent last_idle_decay_at window guard so a
    cron-overlap or restart cannot double-apply HP decay. enforce_guillotine_agents
    skips agents whose guillotine_at was set within the last 60 s (duplicate-reaper
    protection). A per-process run_id is stamped on every batch-write doc.
"""
from __future__ import annotations

import atexit
import collections
import datetime
import math
import os
import re
import signal
import textwrap
import threading
import uuid
from typing import Any

from pymongo import MongoClient, ReturnDocument


_CLIENT: MongoClient | None = None
_CLIENT_KEY: tuple[str, str] | None = None

# --------------------------------------------------------------------------
# Per-process run identifier — stamped on buffered batch-write docs so that
# parallel reaper runs can be distinguished in post-mortem analysis.
# --------------------------------------------------------------------------
_RUN_ID: str = str(uuid.uuid4())[:8]


def _db():
    global _CLIENT, _CLIENT_KEY
    url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    name = os.environ.get("DB_NAME", "quantchoai")
    key = (url, name)
    if _CLIENT is None or _CLIENT_KEY != key:
        _CLIENT = MongoClient(
            url,
            serverSelectionTimeoutMS=int(os.environ.get("QUANTCHO_MONGO_SELECT_TIMEOUT_MS", "5000")),
            connectTimeoutMS=int(os.environ.get("QUANTCHO_MONGO_CONNECT_TIMEOUT_MS", "5000")),
            socketTimeoutMS=int(os.environ.get("QUANTCHO_MONGO_SOCKET_TIMEOUT_MS", "15000")),
            maxPoolSize=int(os.environ.get("QUANTCHO_MONGO_MAX_POOL_SIZE", "8")),
            minPoolSize=0,
            maxIdleTimeMS=int(os.environ.get("QUANTCHO_MONGO_MAX_IDLE_MS", "30000")),
            retryWrites=False,
        )
        _CLIENT_KEY = key
    return _CLIENT[name]


L = "agent_ledger"
Q = "agent_queue"
LOG = "system_logs"


def _invalidate_dispatcher_roster_cache() -> None:
    """Drop the dispatcher's TTL-cached recently-active roster on a membership change.

    init_agent (new/replacement agent) and the guillotine paths (agent reaped/inactivated)
    are the ONLY events that change which agents the dispatcher auction should consider. The
    dispatcher caches that roster for a short TTL to avoid re-scanning the ledger on every
    poll; we drop it here so a registration/reap is reflected immediately instead of after
    the TTL. Lazy local import to avoid a load-time ledger<->dispatcher coupling, and fully
    best-effort: a cache that cannot be reached just expires on its own TTL (correctness is
    unaffected -- the cache is a pure speed optimization)."""
    try:
        from quant.governance import dispatcher as _dispatcher  # local import: no cycle
        _dispatcher.invalidate_roster_cache()
    except Exception:
        pass
AUTOPSY = "agent_autopsies"
AGENT_EVENTS = "agent_events"
APPEALS_DIR = os.environ.get("QUANTCHO_APPEALS_DIR", r"C:\tmp\appeals")

DEFAULT_HP = 50.0
MAX_HP = 100.0
GUILLOTINE_HP = 20.0
SOFT_RECYCLE_HP = 40.0
SOFT_RECYCLE_COOLDOWN_HOURS = 6
IDLE_DECAY_HP = -2.0
ROTATION_LOCK_MINUTES = 30

# Idempotency window for enforce_guillotine_agents: skip re-autopsy if
# guillotine_at was set within this many seconds (prevents duplicate-autopsy
# on cron overlaps).
GUILLOTINE_DEDUP_SECONDS = int(os.environ.get("QUANTCHO_GUILLOTINE_DEDUP_SECS", "60"))

BASE_HAZARD = float(os.environ.get("QUANTCHO_HAZARD_BASE", "0.05"))
WEIGHT_CONTEXT = float(os.environ.get("QUANTCHO_HAZARD_WEIGHT_CONTEXT", "0.00001"))
WEIGHT_STREAK = float(os.environ.get("QUANTCHO_HAZARD_WEIGHT_STREAK", "0.1"))
WEIGHT_MINUTES = float(os.environ.get("QUANTCHO_HAZARD_WEIGHT_MINUTES", "0.005"))
WEIGHT_REQUEUE = float(os.environ.get("QUANTCHO_HAZARD_WEIGHT_REQUEUE", "0.15"))
# Peer-review fix (2026-06-04): the four weights above are all FIRST-ORDER -- hazard rises
# linearly with how-bad-things-ARE (failed streak, time-on-ticket, requeues). They are blind
# to how-bad-things-are-BECOMING. A skill (mu) trace that is "falling AND accelerating" is the
# inflection that precedes the guillotine; seeing it one or two cycles EARLIER is the
# life-saving signal. WEIGHT_ACCEL converts the (negative) robust second difference of the
# mu-history -- collapse ACCELERATION -- into a hazard bump. Sign convention: d2_mu < 0 ==
# downward curvature == collapse accelerating; only that case adds hazard.
WEIGHT_ACCEL = float(os.environ.get("QUANTCHO_HAZARD_WEIGHT_ACCEL", "0.04"))
# Hard cap on the acceleration contribution so a single violent curvature reading can never
# dominate the linear terms or, by itself, slam hazard to the guillotine line. The non-linear
# term is an EARLY-WARNING nudge, not an independent kill switch.
HAZARD_ACCEL_CAP = float(os.environ.get("QUANTCHO_HAZARD_ACCEL_CAP", "0.30"))
HAZARD_SOFT_RECYCLE_THRESHOLD = float(os.environ.get("QUANTCHO_HAZARD_SOFT_RECYCLE", "0.6"))
HAZARD_GUILLOTINE_THRESHOLD = float(os.environ.get("QUANTCHO_HAZARD_GUILLOTINE", "0.9"))

# ---------------------------------------------------------------------------
# CALIBRATED POISSON-LAMBDA BASE HAZARD (roadmap item 9 wire-in, 2026-06-04)
# ---------------------------------------------------------------------------
# Until now the BASE term of `calculate_hazard` was a single flat BASE_HAZARD=0.05
# for EVERY agent -- a blind prior that cannot tell a 4x-worse agent from a clean
# one. A calibrated estimation layer (the slow-loop materializer's `lambda_fail`,
# part of the private/commercial layer) produces a per-agent/domain empirical-Bayes
# failure RATE lambda_a. The correct base hazard is the probability that the agent
# logs AT LEAST ONE failure in the next window:
#
#     h_base = 1 - exp(-lambda_a)            (Poisson: the right family for discrete
#                                             count-arrival failures, roadmap 1A)
#
# A measured lambda_a=0.2 (a 4x-worse agent than the 0.05 flat) therefore reads
# h_base = 1 - e^-0.2 = 0.1813, versus the flat 1 - e^-0.05 = 0.0488 -- the 3.71x
# ratio the flat term was silently erasing.
#
# OVER-DISPERSION: the materializer's dispersion GoF can REJECT the Poisson family
# (clustered / bursty failures, variance >> mean). When it flags `over_dispersed`,
# a Poisson base would be the WRONG law -- clustering concentrates failures into
# fewer windows, so naively it would OVERSTATE P(>=1) at the same mean. We then use
# the Negative-Binomial (NB2) closed form for P(>=1), the correct over-dispersed
# count law, which reduces continuously to the Poisson value as dispersion -> 1.
#
# GATING: the whole calibrated path is behind QCHO_HAZARD_CALIBRATED, so the legacy
# flat behavior is always recoverable and the change is auditable:
#     "shadow" (default) -> calibrated base used LIVE, but BOTH the calibrated and
#                           the flat base are recorded in `hazard_components` /
#                           `calibration_diagnostics` for shadow-compare auditing.
#     "on"/"1"/"true"    -> calibrated base used live (no shadow record overhead).
#     "off"/"0"/"false"  -> legacy flat BASE_HAZARD only (exact old behavior).
# UNMEASURED -> ALWAYS the flat BASE_HAZARD: an un-profiled / brand-new agent
# (no measured lambda row, n_windows == 0) sees ZERO behavior change. That is the
# empirical-Bayes contract: use data when you have it, the flat prior when you do not.
_HAZARD_CALIBRATED_MODE = os.environ.get("QCHO_HAZARD_CALIBRATED", "shadow").strip().lower()
HAZARD_CALIBRATED_ENABLED = _HAZARD_CALIBRATED_MODE not in ("off", "0", "false", "no", "")
HAZARD_CALIBRATED_SHADOW = _HAZARD_CALIBRATED_MODE in ("shadow", "shadow-compare", "compare")
WARM_EFFICIENCY_ALPHA = float(os.environ.get("QUANTCHO_WARM_EFFICIENCY_ALPHA", "0.08"))
WARM_EFFICIENCY_FLOOR = float(os.environ.get("QUANTCHO_WARM_EFFICIENCY_FLOOR", "0.20"))
PROPULSION_CONTEXT_DRAG_PER_10K = float(os.environ.get("QUANTCHO_PROP_CONTEXT_DRAG_PER_10K", "0.05"))
PROPULSION_TRACEBACK_DRAG_PER_10K = float(os.environ.get("QUANTCHO_PROP_TRACEBACK_DRAG_PER_10K", "0.03"))
PROPULSION_REPO_CHURN_DRAG = float(os.environ.get("QUANTCHO_PROP_REPO_CHURN_DRAG", "0.04"))
PROPULSION_STALE_LOG_DRAG = float(os.environ.get("QUANTCHO_PROP_STALE_LOG_DRAG", "0.02"))
PROPULSION_NO_PROGRESS_DRAG = float(os.environ.get("QUANTCHO_PROP_NO_PROGRESS_DRAG", "0.12"))
PROPULSION_NO_PROGRESS_LIMIT = int(os.environ.get("QUANTCHO_PROP_NO_PROGRESS_LIMIT", "3"))


# P2-4 (2026-06-04): events that mean the agent's WORK was rejected / failed. A warm
# success streak must reset to 0 on any of these, otherwise the V4 propulsion warm_basis
# (which adds 3 min of warm credit per streak point) keeps crediting a now-cold agent.
# Pure HP bookkeeping events (idle_decay, supervisor_guilt, caught_by_qa) do NOT reset the
# streak -- only a genuine own-work failure does.
STREAK_RESET_EVENTS = frozenset({
    "committee_reject",
    "committee_rejected",
    "plausibility_fail",
    "p0_online",
    "qa_missed_bug",
})

EVENTS = {
    "pr_merged_review_branch": +10.0,
    "done_pending_merge": +10.0,
    "committee_reject": -15.0,
    "committee_rejected": -15.0,
    "ci_qa_pass": +5.0,
    "strategy_win": +5.0,
    "qa_caught_bug": +5.0,
    "caught_by_qa": -2.0,
    "plausibility_fail": -50.0,
    "p0_online": -50.0,
    "qa_missed_bug": -40.0,
    "supervisor_guilt": -20.0,
    "idle_decay": IDLE_DECAY_HP,
}

TICKET_REWARDS = {
    "maintenance_low": +1.0,
    "feature_high": +8.0,
    "critical_epic": +20.0,
}

CHAOS_FINDING_REWARDS = {
    "P0": 14.0,
    "P1": 8.0,
    "P2": 4.0,
    "P3": 2.0,
}

DOMAIN_SKILL_MAP = {
    "Risk": "risk_math_skill",
    "Pricing": "risk_math_skill",
    "Analytics": "risk_math_skill",
    "Trading": "execution_skill",
    "Execution": "execution_skill",
    "Compliance": "security_skill",
    "Infra": "security_skill",
    "Ops": "security_skill",
    "FrontOffice": "frontend_skill",
    "Reporting": "frontend_skill",
    "Committee": "committee_judgment",
    "Governance": "committee_judgment",
    "Chaos": "chaos_reproduction",
}

MONEY_TEXT_RE = re.compile(
    r"\b(var|cvar|pnl|p&l|nav|gross mv|pricing|valuation|position|t9|"
    r"margin|exposure|risk|tax|accounting|ledger|trade|fill)\b",
    re.I,
)
SECURITY_TEXT_RE = re.compile(r"\b(auth|tenant|permission|security|token|validator|ofac|sanction)\b", re.I)


# =============================================================================
# BUFFERED LOG WRITER
# =============================================================================
# system_logs and agent_events are fire-and-forget telemetry -- they must all
# eventually be written (no loss), but they are NOT on a consistency path:
# no HP decision reads them mid-flight, so a ~5-second batch delay is safe.
#
# Design:
#   _LOG_BUFFER  — deque of (collection_name, doc) pairs
#   _LOG_LOCK    — guards the deque and the flush logic
#   _FLUSH_INTERVAL_S — wall-clock seconds between timed flushes
#   _FLUSH_BATCH_SIZE — flush early when this many docs are queued
#
# The background daemon thread drains the buffer via insert_many, converting
# per-event round-trips into one bulk write every 5 s or 50 docs.
# A flush() call (and the atexit hook) drain synchronously.
# =============================================================================

_LOG_BUFFER: collections.deque = collections.deque()
_LOG_LOCK = threading.Lock()
_FLUSH_INTERVAL_S: float = float(os.environ.get("QUANTCHO_LOG_FLUSH_INTERVAL_S", "5"))
_FLUSH_BATCH_SIZE: int = int(os.environ.get("QUANTCHO_LOG_FLUSH_BATCH_SIZE", "50"))
_flush_thread: threading.Thread | None = None


def _drain_buffer() -> dict[str, int]:
    """Pop everything from the buffer, group by collection, insert_many.

    Returns {collection_name: count_inserted}. Failures are silently swallowed
    so a dead Mongo connection never kills the main process.
    """
    if not _LOG_BUFFER:
        return {}
    with _LOG_LOCK:
        batch = list(_LOG_BUFFER)
        _LOG_BUFFER.clear()
    if not batch:
        return {}
    # Group by collection name.
    groups: dict[str, list[dict]] = {}
    for coll_name, doc in batch:
        groups.setdefault(coll_name, []).append(doc)
    counts: dict[str, int] = {}
    try:
        db = _db()
        for coll_name, docs in groups.items():
            try:
                db[coll_name].insert_many(docs, ordered=False)
                counts[coll_name] = len(docs)
            except Exception:
                # Best-effort: if Mongo is down we lose this batch, but we do
                # not crash the process. Telemetry loss is acceptable; HP
                # correctness is not.
                pass
    except Exception:
        pass
    return counts


def flush() -> dict[str, int]:
    """Synchronously drain all pending buffered log docs to Mongo.

    Call this before process exit, at test teardown, or whenever you need the
    log state to be durable.
    """
    return _drain_buffer()


def _buffer_log(collection: str, doc: dict) -> None:
    """Enqueue a log doc for deferred batch-write. Thread-safe."""
    # Stamp the process run_id on every buffered doc for post-mortem
    # distinguishability across parallel cron runs.
    stamped = dict(doc)
    stamped.setdefault("run_id", _RUN_ID)
    with _LOG_LOCK:
        _LOG_BUFFER.append((collection, stamped))
    # Trigger an early flush if we've hit the batch-size threshold.
    if len(_LOG_BUFFER) >= _FLUSH_BATCH_SIZE:
        # Don't hold the lock while calling _drain_buffer; the drain itself
        # takes the lock to snapshot+clear.
        _drain_buffer()


def _flush_loop() -> None:
    """Background daemon: flush every _FLUSH_INTERVAL_S seconds."""
    import time
    while True:
        time.sleep(_FLUSH_INTERVAL_S)
        try:
            _drain_buffer()
        except Exception:
            pass


def _start_flush_thread() -> None:
    global _flush_thread
    if _flush_thread is not None and _flush_thread.is_alive():
        return
    t = threading.Thread(target=_flush_loop, name="ledger-log-flusher", daemon=True)
    t.start()
    _flush_thread = t


# Start the background flusher when this module is imported (daemon thread
# so it never blocks process exit). Also register atexit to drain on shutdown.
_start_flush_thread()
atexit.register(flush)


# =============================================================================
# end BUFFERED LOG WRITER
# =============================================================================


def default_telemetry_snapshot() -> dict:
    return {
        "hazard_rate": 0.0,
        "context_tokens": 0,
        "streak_failed": 0,
        "same_ticket_minutes": 0,
        "failed_requeue_count": 0,
        "last_logic_failure_at": None,
        "last_infra_noise_at": None,
        "soft_recycle_candidate": False,
        "shadow_would_guillotine": False,
        "warm_runtime_minutes": 0.0,
        "successful_ticket_streak": 0,
        "useful_progress_events": 0,
        "accepted_progress_events": 0,
        "elapsed_minutes": 0.0,
        "traceback_noise_chars": 0,
        "repo_churn_files": 0,
        "stale_log_entries": 0,
        "repeated_no_progress_cycles": 0,
        "ticket_escape_velocity": 0.0,
        "propulsion": {
            "warm_efficiency": WARM_EFFICIENCY_FLOOR,
            "specific_impulse": 0.0,
            "cognitive_velocity": 0.0,
            "drag_penalty": 0.0,
            "effective_thrust": 0.0,
            "escape_velocity": 0.0,
            "can_escape": True,
            "soft_recycle_preferred": False,
            "shadow_guillotine_guardrail": False,
        },
    }


def default_credit_rating() -> dict:
    return {"mu": 50.0, "sigma": 30.0}


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _parse_ts(value: Any) -> datetime.datetime | None:
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _clamp_hp(value: float) -> float:
    return max(0.0, min(MAX_HP, float(value)))


def _rank_for_hp(hp: float) -> int:
    if hp > 80.0:
        return 3
    if hp >= 40.0:
        return 2
    return 1


def _migrate_legacy_weight(value: Any) -> float:
    try:
        hp = float(value)
    except Exception:
        return DEFAULT_HP
    if 0.0 <= hp <= 1.0:
        return _clamp_hp(hp * 100.0)
    return _clamp_hp(hp)


def initialize_governance_v3_schema() -> dict:
    """Create Governance V3 materialized telemetry fields and TTL event index.

    This is intentionally deterministic and local. Dispatchers should read the
    materialized `telemetry_snapshot` fields instead of calculating hazard on the
    hot path.
    """
    db = _db()
    telemetry = default_telemetry_snapshot()
    credit = default_credit_rating()
    telemetry_res = db[L].update_many(
        {"telemetry_snapshot": {"$exists": False}},
        {"$set": {"telemetry_snapshot": telemetry}},
    )
    credit_res = db[L].update_many(
        {"credit_rating": {"$exists": False}},
        {"$set": {"credit_rating": credit}},
    )
    # Backfill newly added nested fields without overwriting live values.
    nested_updates = 0
    for key, value in telemetry.items():
        res = db[L].update_many(
            {f"telemetry_snapshot.{key}": {"$exists": False}},
            {"$set": {f"telemetry_snapshot.{key}": value}},
        )
        nested_updates += int(getattr(res, "modified_count", 0) or 0)
    for key, value in credit.items():
        res = db[L].update_many(
            {f"credit_rating.{key}": {"$exists": False}},
            {"$set": {f"credit_rating.{key}": value}},
        )
        nested_updates += int(getattr(res, "modified_count", 0) or 0)
    index_name = db[AGENT_EVENTS].create_index("created_at", expireAfterSeconds=259200)
    return {
        "telemetry_initialized": int(getattr(telemetry_res, "modified_count", 0) or 0),
        "credit_initialized": int(getattr(credit_res, "modified_count", 0) or 0),
        "nested_backfilled": nested_updates,
        "agent_events_ttl_index": index_name,
    }


def record_agent_event(
    agent_id: str,
    event_type: str,
    ticket_id: str | None = None,
    domain: str | None = None,
    context_tokens: int | None = None,
    error_class: str | None = None,
    reason: str | None = None,
) -> dict:
    """Append a compact telemetry event for async Governance V3 sync.

    Uses the buffered writer — fire-and-forget, NOT a consistency path.
    """
    doc = {
        "agent_id": agent_id,
        "event_type": event_type,
        "ticket_id": ticket_id,
        "domain": domain,
        "context_tokens": int(context_tokens or 0),
        "error_class": error_class,
        "reason": reason,
        "created_at": datetime.datetime.utcnow(),
        "ts": _now(),
    }
    _buffer_log(AGENT_EVENTS, doc)
    return doc


def _accel_hazard_term(snapshot: dict) -> tuple[float, dict]:
    """Non-linear (second-order) hazard contribution from mu-collapse ACCELERATION.

    The linear terms answer "how bad is the agent NOW?". This answers "how fast is the
    agent getting worse?". We read the agent's mu-history (newest-LAST) and compute the
    plain second difference of mu over the last three points:

        d2_mu = mu[-1] - 2*mu[-2] + mu[-3]

        d2_mu < 0  ->  downward curvature  ->  the decline is ACCELERATING (pre-guillotine
                       inflection: "falling AND speeding up")
        d2_mu ~ 0  ->  flat / linear drift (a plateau or a steady glide -- NOT an emergency)
        d2_mu > 0  ->  recovery accelerating (good; never raises hazard)

    This is the hot-path-cheap deterministic read (the same second-difference shape as
    risk_core.gamma_acceleration_second_difference). Only a downward curvature beyond a
    small noise threshold contributes; a barely-significant wiggle adds almost nothing
    while a genuine collapse adds real hazard. The term is hard-capped at HAZARD_ACCEL_CAP
    so it is an early-warning NUDGE, never an independent kill switch. (A robust,
    significance-gated local-quadratic refinement is part of the private/commercial layer.)
    """
    mu_history = snapshot.get("mu_history")
    components = {"accel": 0.0, "d2_mu": 0.0, "significant": False, "direction": "flat"}
    if not mu_history or len(mu_history) < 3:
        return 0.0, components
    try:
        h = [float(v) for v in mu_history if v is not None]
    except (TypeError, ValueError):
        return 0.0, components
    if len(h) < 3:
        return 0.0, components
    d2_mu = h[-1] - 2.0 * h[-2] + h[-3]
    # A small deterministic noise floor (in mu points) below which curvature is ignored,
    # so a noisy-but-flat trace does not false-alarm.
    noise_floor = float(os.environ.get("QUANTCHO_HAZARD_ACCEL_NOISE_FLOOR", "1.0"))
    significant = abs(d2_mu) > noise_floor
    direction = "down" if d2_mu < 0.0 else ("up" if d2_mu > 0.0 else "flat")
    components["d2_mu"] = round(d2_mu, 5)
    components["significant"] = significant
    components["direction"] = direction
    # Only a significant DOWNWARD curvature (accelerating collapse) raises hazard.
    if not significant or d2_mu >= 0.0:
        return 0.0, components
    # Charge only for curvature beyond the noise floor -> SNR-aware magnitude.
    excess = max(0.0, abs(d2_mu) - noise_floor)
    accel_term = min(HAZARD_ACCEL_CAP, WEIGHT_ACCEL * excess)
    components["accel"] = round(accel_term, 4)
    return accel_term, components


def _negbinom_prob_at_least_one(lmbda: float, dispersion: float) -> float:
    """P(>=1 failure in the next window) under a Negative-Binomial(NB2) with mean lambda.

    When the dispersion goodness-of-fit REJECTS the Poisson assumption (clustered /
    bursty failures: variance d*lambda with d > 1) the Poisson P(>=1) = 1 - e^-lambda is
    the wrong law -- it ignores that the failures pile into fewer windows, leaving more
    windows clean. The Negative-Binomial NB2 is the canonical over-dispersed count model:

        mean       = lambda
        variance   = d * lambda          (d = dispersion index = var/mean > 1)
        size r     = lambda / (d - 1)    (number of "successes" param)
        P(X = 0)   = (r / (r + lambda))^r = (1/d)^r
        P(X >= 1)  = 1 - (1/d)^(lambda / (d - 1))

    As d -> 1+ this converges CONTINUOUSLY to the Poisson 1 - e^-lambda (the limit
    (1/d)^(lambda/(d-1)) -> e^-lambda), so the over-dispersed path is a strict, smooth
    generalization, never a discontinuity. For an over-dispersed stream it returns a
    SMALLER P(>=1) than the Poisson at the same mean -- the statistically correct
    "more clean windows because failures cluster" behavior.
    """
    if lmbda <= 0.0:
        return 0.0
    # Degenerate / non-over-dispersed dispersion -> fall back to the Poisson closed form
    # (also the exact d -> 1 limit), so this never produces a worse number than Poisson.
    if dispersion <= 1.0 or not math.isfinite(dispersion):
        return max(0.0, min(1.0, 1.0 - math.exp(-lmbda)))
    r = lmbda / (dispersion - 1.0)
    # P(X=0) = (1/d)^r ; guard the pathological r->inf / d->1+ edge via the Poisson limit.
    try:
        p_zero = math.exp(-r * math.log(dispersion))
    except (ValueError, OverflowError):
        p_zero = math.exp(-lmbda)
    return max(0.0, min(1.0, 1.0 - p_zero))


def _calibrated_base_hazard(snapshot: dict) -> tuple[float, dict]:
    """Empirical-Bayes base hazard from the calibrated per-agent/domain Poisson lambda.

    Replaces the flat BASE_HAZARD with h_base = 1 - exp(-lambda_a) when a MEASURED
    lambda is available for this agent/domain (a `lambda_fail` materialized by the
    slow-loop calibration layer). Falls back to the flat BASE_HAZARD when unmeasured
    (no row / n_windows == 0) so an un-profiled agent sees no behavior change.

    Returns (base_hazard, diagnostics). `diagnostics` is always populated for auditing:
    it records BOTH the calibrated and the flat base, the measured lambda, the window
    count, whether the NegBinom (over-dispersed) path was taken, and the live `source`.

    Snapshot keys consumed (all materialized by the slow-loop calibration layer into the
    agent telemetry snapshot; any may be absent on an un-profiled agent):
        lambda_fail            -- EWMA-tracked calibrated failure rate (failures/window)
        poisson_n_windows      -- number of observed failure windows (0 => unmeasured)
        poisson_over_dispersed -- dispersion GoF rejected Poisson (clustered failures)
        poisson_dispersion     -- the var/mean dispersion index (used by the NegBinom path)
    """
    diag = {
        "source": "flat",
        "flat_base": BASE_HAZARD,
        "calibrated_base": None,
        "lambda_fail": None,
        "n_windows": 0,
        "over_dispersed": False,
        "dispersion": None,
        "calibrated_enabled": HAZARD_CALIBRATED_ENABLED,
    }

    # Gate OFF -> exact legacy flat behavior, no calibrated computation at all.
    if not HAZARD_CALIBRATED_ENABLED:
        return BASE_HAZARD, diag

    n_windows = int(snapshot.get("poisson_n_windows") or 0)
    lam_raw = snapshot.get("lambda_fail")
    # UNMEASURED: no measured lambda row (n_windows == 0) or no lambda value present ->
    # fall back to the flat prior. This is the empirical-Bayes "prior when no data" leg
    # and guarantees ZERO behavior change for un-profiled / brand-new agents.
    if n_windows <= 0 or lam_raw is None:
        return BASE_HAZARD, diag

    try:
        lam = float(lam_raw)
    except (TypeError, ValueError):
        return BASE_HAZARD, diag
    if not math.isfinite(lam) or lam < 0.0:
        return BASE_HAZARD, diag

    over_dispersed = bool(snapshot.get("poisson_over_dispersed"))
    dispersion_raw = snapshot.get("poisson_dispersion")
    try:
        dispersion = float(dispersion_raw) if dispersion_raw is not None else 0.0
    except (TypeError, ValueError):
        dispersion = 0.0

    if over_dispersed:
        calibrated = _negbinom_prob_at_least_one(lam, dispersion)
        source = "negbinom"
    else:
        # Poisson empirical-Bayes base: h_base = 1 - e^-lambda.
        calibrated = max(0.0, min(1.0, 1.0 - math.exp(-lam)))
        source = "poisson"

    diag.update({
        "source": source,
        "calibrated_base": round(calibrated, 6),
        "lambda_fail": lam,
        "n_windows": n_windows,
        "over_dispersed": over_dispersed,
        "dispersion": dispersion if dispersion_raw is not None else None,
    })
    return calibrated, diag


def calculate_hazard(snapshot: dict | None) -> dict:
    snapshot = snapshot or {}
    context_tokens = int(snapshot.get("context_tokens") or 0)
    streak_failed = int(snapshot.get("streak_failed") or 0)
    same_ticket_minutes = float(snapshot.get("same_ticket_minutes") or 0.0)
    failed_requeue_count = int(snapshot.get("failed_requeue_count") or 0)
    accel_term, accel_components = _accel_hazard_term(snapshot)
    # Empirical-Bayes calibrated base: h_base = 1 - e^-lambda_a when a measured per-agent/
    # domain Poisson rate exists (NegBinom path on over-dispersion), else the flat BASE_HAZARD.
    # Flag-gated + auditable; identical to the old flat base for unmeasured agents.
    base_hazard, calibration_diag = _calibrated_base_hazard(snapshot)
    raw_hazard = (
        base_hazard
        + (WEIGHT_CONTEXT * context_tokens)
        + (WEIGHT_STREAK * streak_failed)
        + (WEIGHT_MINUTES * same_ticket_minutes)
        + (WEIGHT_REQUEUE * failed_requeue_count)
        + accel_term  # non-linear: collapse-acceleration early-warning (>= 0, capped)
    )
    hazard = round(max(0.0, min(1.0, raw_hazard)), 4)
    result = {
        "hazard_rate": hazard,
        "soft_recycle_candidate": hazard >= HAZARD_SOFT_RECYCLE_THRESHOLD,
        "shadow_would_guillotine": hazard >= HAZARD_GUILLOTINE_THRESHOLD,
        # `hazard_components` is, by contract, a FLAT map of non-negative NUMERIC
        # contributions. The accel term
        # is a real contribution and belongs here; its non-numeric diagnostics (d2_mu sign,
        # significance, direction) live separately under `accel_diagnostics`. The base is now
        # the calibrated empirical-Bayes h_base (or the flat prior when unmeasured) -- still a
        # single non-negative numeric contribution, so the flat-map contract is preserved.
        "hazard_components": {
            "base": round(base_hazard, 6),
            "context": round(WEIGHT_CONTEXT * context_tokens, 4),
            "streak": round(WEIGHT_STREAK * streak_failed, 4),
            "same_ticket": round(WEIGHT_MINUTES * same_ticket_minutes, 4),
            "requeue": round(WEIGHT_REQUEUE * failed_requeue_count, 4),
            "accel": accel_components["accel"],
        },
        "accel_diagnostics": {
            "d2_mu": accel_components["d2_mu"],
            "significant": accel_components["significant"],
            "direction": accel_components["direction"],
        },
        # Calibrated-base provenance: which base was used (poisson / negbinom / flat), the
        # measured lambda, window count, and -- in shadow-compare mode -- the flat base the
        # legacy path WOULD have used, so the wire-in is fully auditable / reversible.
        "calibration_diagnostics": calibration_diag,
    }
    # Shadow-compare auditing: record what the OLD flat-base hazard would have been so the
    # calibrated change can be diffed agent-by-agent before any hard promotion.
    if HAZARD_CALIBRATED_SHADOW and calibration_diag.get("source") != "flat":
        flat_raw = (
            BASE_HAZARD
            + (WEIGHT_CONTEXT * context_tokens)
            + (WEIGHT_STREAK * streak_failed)
            + (WEIGHT_MINUTES * same_ticket_minutes)
            + (WEIGHT_REQUEUE * failed_requeue_count)
            + accel_term
        )
        result["calibration_diagnostics"]["shadow_flat_hazard_rate"] = round(
            max(0.0, min(1.0, flat_raw)), 4
        )
    return result


def calculate_propulsion(snapshot: dict | None, credit_rating: dict | None = None) -> dict:
    """Calculate V4.1 warm-state propulsion metrics in shadow mode.

    This is pure O(1) local math. It measures whether a warm agent is producing
    useful progress or merely burning compute against context drag. The result is
    advisory telemetry only; it must not directly kill or route agents until
    calibrated against production samples.
    """
    snapshot = snapshot or {}
    credit_rating = credit_rating or {}
    context_tokens = max(0, int(snapshot.get("context_tokens") or 0))
    progress = max(
        0,
        int(snapshot.get("useful_progress_events") or snapshot.get("accepted_progress_events") or 0),
    )
    elapsed_minutes = max(1.0, float(snapshot.get("elapsed_minutes") or snapshot.get("same_ticket_minutes") or 1.0))
    warm_basis = max(
        0.0,
        float(snapshot.get("warm_runtime_minutes") or 0.0)
        + (3.0 * float(snapshot.get("successful_ticket_streak") or 0.0)),
    )
    warm_efficiency = max(
        WARM_EFFICIENCY_FLOOR,
        min(1.0, 1.0 - math.exp(-WARM_EFFICIENCY_ALPHA * warm_basis)),
    )
    specific_impulse = progress / max(context_tokens / 1000.0, 1.0)
    cognitive_velocity = progress / elapsed_minutes
    drag_penalty = (
        (context_tokens / 10000.0 * PROPULSION_CONTEXT_DRAG_PER_10K)
        + (float(snapshot.get("traceback_noise_chars") or 0.0) / 10000.0 * PROPULSION_TRACEBACK_DRAG_PER_10K)
        + (float(snapshot.get("repo_churn_files") or 0.0) * PROPULSION_REPO_CHURN_DRAG)
        + (float(snapshot.get("stale_log_entries") or 0.0) * PROPULSION_STALE_LOG_DRAG)
        # P1-2 fix (2026-06-04): requeue is already charged in V3 hazard (WEIGHT_REQUEUE) and
        # V5 context penalty (ALPHA). Do NOT triple-count it here in V4 drag.
        + (float(snapshot.get("repeated_no_progress_cycles") or 0.0) * PROPULSION_NO_PROGRESS_DRAG)
    )
    mu = float(credit_rating.get("mu") or snapshot.get("mu") or 50.0)
    effective_thrust = (mu * specific_impulse * warm_efficiency) - drag_penalty
    escape_velocity = max(
        0.0,
        float(
            snapshot.get("ticket_escape_velocity")
            or snapshot.get("ci_gate_difficulty")
            or snapshot.get("ticket_difficulty")
            or 0.0
        ),
    )
    can_escape = True if escape_velocity <= 0.0 else effective_thrust >= escape_velocity
    hazard = float(snapshot.get("hazard_rate") or 0.0)
    no_progress_cycles = int(snapshot.get("repeated_no_progress_cycles") or 0)
    shadow_guillotine = (
        not can_escape
        and hazard >= HAZARD_GUILLOTINE_THRESHOLD
        and no_progress_cycles >= PROPULSION_NO_PROGRESS_LIMIT
    )
    soft_recycle_preferred = (
        not can_escape
        and not shadow_guillotine
        and hazard >= HAZARD_SOFT_RECYCLE_THRESHOLD
    )
    return {
        "warm_efficiency": round(warm_efficiency, 4),
        "specific_impulse": round(specific_impulse, 4),
        "cognitive_velocity": round(cognitive_velocity, 4),
        "drag_penalty": round(drag_penalty, 4),
        "effective_thrust": round(effective_thrust, 4),
        "escape_velocity": round(escape_velocity, 4),
        "can_escape": can_escape,
        "soft_recycle_preferred": soft_recycle_preferred,
        "shadow_guillotine_guardrail": shadow_guillotine,
    }


def sync_hazard_rates_shadow(limit: int = 500) -> dict:
    """Materialize Governance V3 hazard telemetry in shadow mode.

    The function records would-act decisions but does not recycle or kill agents.
    """
    db = _db()
    initialize_governance_v3_schema()
    in_progress = {
        row.get("picked_by"): row
        for row in db[Q].find(
            {"status": "IN_PROGRESS", "picked_by": {"$exists": True}},
            {"_id": 0, "picked_by": 1, "picked_at": 1, "ticket_id": 1, "target_domain": 1},
        )
        if row.get("picked_by")
    }
    recent_events: dict[str, dict[str, int]] = {}
    since = datetime.datetime.utcnow() - datetime.timedelta(hours=72)
    try:
        for row in db[AGENT_EVENTS].aggregate(
            [
                {"$match": {"created_at": {"$gte": since}}},
                {"$group": {"_id": {"agent_id": "$agent_id", "event_type": "$event_type"}, "n": {"$sum": 1}}},
            ],
            allowDiskUse=False,
        ):
            agent_id = row["_id"].get("agent_id")
            event_type = row["_id"].get("event_type")
            if agent_id and event_type:
                recent_events.setdefault(agent_id, {})[event_type] = int(row.get("n", 0))
    except Exception:
        recent_events = {}

    scanned = synced = soft = guillotine = 0
    high_risk: list[dict] = []
    for agent in db[L].find({"active": {"$ne": False}}).limit(limit):
        scanned += 1
        agent_id = agent.get("agent_id")
        snapshot = dict(default_telemetry_snapshot())
        snapshot.update(agent.get("telemetry_snapshot") or {})
        # Wire the calibrated per-agent Poisson rate (materialized by the slow-loop
        # calibration layer) into the hazard snapshot so calculate_hazard's base term is the
        # empirical-Bayes h_base = 1 - e^-lambda instead of the flat BASE_HAZARD. The
        # calibration layer writes the headline `lambda_fail` at the agent root and the
        # GoF/over-dispersion flags under `shadow`; absence of any of these leaves the
        # snapshot unmeasured -> flat fallback.
        materialized_lambda = agent.get("lambda_fail")
        materialized_shadow = agent.get("shadow") or {}
        if materialized_lambda is not None and "lambda_fail" not in (agent.get("telemetry_snapshot") or {}):
            snapshot["lambda_fail"] = materialized_lambda
        if "poisson_n_windows" not in snapshot and materialized_shadow.get("n_failure_windows") is not None:
            snapshot["poisson_n_windows"] = materialized_shadow.get("n_failure_windows")
        if "poisson_over_dispersed" not in snapshot and "over_dispersed" in materialized_shadow:
            snapshot["poisson_over_dispersed"] = materialized_shadow.get("over_dispersed")
        if "poisson_dispersion" not in snapshot and materialized_shadow.get("dispersion") is not None:
            snapshot["poisson_dispersion"] = materialized_shadow.get("dispersion")
        events = recent_events.get(agent_id, {})
        if events:
            snapshot["streak_failed"] = max(int(snapshot.get("streak_failed") or 0), events.get("logic_failure", 0))
            snapshot["failed_requeue_count"] = max(
                int(snapshot.get("failed_requeue_count") or 0),
                events.get("ticket_requeued", 0) + events.get("logic_failure", 0),
            )
        ticket = in_progress.get(agent_id)
        if ticket:
            picked_at = _parse_ts(ticket.get("picked_at"))
            if picked_at:
                snapshot["same_ticket_minutes"] = max(
                    0.0,
                    (datetime.datetime.utcnow() - picked_at).total_seconds() / 60.0,
                )
            snapshot["current_ticket_id"] = ticket.get("ticket_id")
            snapshot["current_domain"] = ticket.get("target_domain")
        hazard = calculate_hazard(snapshot)
        snapshot["hazard_rate"] = hazard["hazard_rate"]
        propulsion = calculate_propulsion(snapshot, agent.get("credit_rating") or agent.get("domain_skills") or {})
        set_doc = {
            "telemetry_snapshot.hazard_rate": hazard["hazard_rate"],
            "telemetry_snapshot.soft_recycle_candidate": hazard["soft_recycle_candidate"],
            "telemetry_snapshot.shadow_would_guillotine": hazard["shadow_would_guillotine"],
            "telemetry_snapshot.hazard_components": hazard["hazard_components"],
            "telemetry_snapshot.calibration_diagnostics": hazard.get("calibration_diagnostics", {}),
            "telemetry_snapshot.propulsion": propulsion,
            "telemetry_snapshot.context_tokens": int(snapshot.get("context_tokens") or 0),
            "telemetry_snapshot.streak_failed": int(snapshot.get("streak_failed") or 0),
            "telemetry_snapshot.same_ticket_minutes": round(float(snapshot.get("same_ticket_minutes") or 0.0), 2),
            "telemetry_snapshot.failed_requeue_count": int(snapshot.get("failed_requeue_count") or 0),
            "telemetry_snapshot.last_synced_at": _now(),
        }
        if "current_ticket_id" in snapshot:
            set_doc["telemetry_snapshot.current_ticket_id"] = snapshot["current_ticket_id"]
            set_doc["telemetry_snapshot.current_domain"] = snapshot.get("current_domain")
        db[L].update_one({"_id": agent["_id"]}, {"$set": set_doc})
        synced += 1
        if hazard["soft_recycle_candidate"]:
            soft += 1
        if hazard["shadow_would_guillotine"]:
            guillotine += 1
        if hazard["soft_recycle_candidate"] or hazard["shadow_would_guillotine"]:
            high_risk.append(
                {
                    "agent_id": agent_id,
                    "hp": agent.get("hp", agent.get("weight")),
                    "hazard_rate": hazard["hazard_rate"],
                    "soft_recycle_candidate": hazard["soft_recycle_candidate"],
                    "shadow_would_guillotine": hazard["shadow_would_guillotine"],
                    "components": hazard["hazard_components"],
                }
            )
    return {
        "mode": "shadow",
        "scanned": scanned,
        "synced": synced,
        "soft_recycle_candidates": soft,
        "shadow_would_guillotine": guillotine,
        "high_risk_sample": sorted(high_risk, key=lambda x: x["hazard_rate"], reverse=True)[:20],
    }


def init_agent(agent_id: str, rank: int, squad: str, weight: float = DEFAULT_HP, process_id: int | None = None):
    db = _db()
    existing = db[L].find_one(
        {"agent_id": agent_id},
        {
            "weight": 1,
            "hp": 1,
            "guillotine_at": 1,
            "guillotine_reason": 1,
            "last_autopsy_file": 1,
            "last_autopsy_summary": 1,
        },
    )
    hp = _migrate_legacy_weight(existing.get("weight")) if existing else _clamp_hp(weight)
    replacement = bool(existing and (hp < GUILLOTINE_HP or existing.get("guillotine_at")))
    if replacement:
        hp = DEFAULT_HP
    update_doc = {
        "agent_id": agent_id,
        "rank": _rank_for_hp(hp),
        "squad": squad,
        "weight": hp,
        "hp": hp,
        "active": True,
        "last_seen_at": _now(),
    }
    if replacement:
        update_doc.update(
            {
                "replaced_at": _now(),
                "replacement_reason": existing.get("guillotine_reason") or "hp_below_20",
                "context_epoch": _now(),
                "inherited_autopsy_file": existing.get("last_autopsy_file"),
                "inherited_autopsy_summary": existing.get("last_autopsy_summary"),
            }
        )
    if process_id is not None:
        update_doc["process_id"] = int(process_id)
    op: dict[str, Any] = {
        "$set": update_doc,
        "$setOnInsert": {"promotions": 0, "demotions": 0, "spawned_at": _now()},
    }
    if replacement:
        op["$unset"] = {"guillotine_at": "", "guillotine_reason": ""}
        op["$inc"] = {"replacements": 1, "generation": 1}
    db[L].update_one(
        {"agent_id": agent_id},
        op,
        upsert=True,
    )
    # Roster membership changed -> drop the dispatcher's TTL roster cache so the new /
    # replaced agent is auction-visible immediately (efficiency 2026-06-04).
    _invalidate_dispatcher_roster_cache()


def get(agent_id: str) -> dict:
    doc = _db()[L].find_one({"agent_id": agent_id})
    if not doc:
        return {"weight": DEFAULT_HP, "hp": DEFAULT_HP, "rank": 2, "active": True}
    hp = _migrate_legacy_weight(doc.get("hp", doc.get("weight", DEFAULT_HP)))
    if doc.get("weight") != hp or doc.get("hp") != hp or doc.get("rank") != _rank_for_hp(hp):
        _db()[L].update_one(
            {"agent_id": agent_id},
            {"$set": {"weight": hp, "hp": hp, "rank": _rank_for_hp(hp), "last_seen_at": _now()}},
        )
        doc["weight"] = hp
        doc["hp"] = hp
        doc["rank"] = _rank_for_hp(hp)
    return doc


def model_profile(agent_id: str) -> dict:
    agent = get(agent_id)
    hp = _migrate_legacy_weight(agent.get("hp", agent.get("weight", DEFAULT_HP)))
    inherited = ""
    if agent.get("inherited_autopsy_summary") or agent.get("inherited_autopsy_file"):
        inherited = (
            "Autopsy inheritance: your predecessor failed. "
            f"Summary: {agent.get('inherited_autopsy_summary') or 'see file'} "
            f"Autopsy file: {agent.get('inherited_autopsy_file') or 'n/a'}. "
            "Do not repeat the same failure mode."
        )
    if hp > 80.0:
        profile = {"hp": hp, "rank": 3, "temperature": 0.1, "mode": "exploitation"}
        if inherited:
            profile["system_injection"] = inherited
        return profile
    if hp >= 40.0:
        profile = {"hp": hp, "rank": 2, "temperature": 0.4, "mode": "standard"}
        if inherited:
            profile["system_injection"] = inherited
        return profile
    return {
        "hp": hp,
        "rank": 1,
        "temperature": 0.8,
        "mode": "exploration",
        "system_injection": (
            "Your previous logic is failing. Discard your current mental model "
            "and explore a completely different approach."
        ) + (f" {inherited}" if inherited else ""),
    }


def allowed_tier(agent_id: str) -> dict:
    agent = get(agent_id)
    hp = _migrate_legacy_weight(agent.get("hp", agent.get("weight", DEFAULT_HP)))
    tier0 = os.environ.get("QUANTCHO_TIER0_MODEL", "")
    tier1 = os.environ.get("QUANTCHO_TIER1_MODEL", "")
    tier2 = os.environ.get("QUANTCHO_TIER2_MODEL", "")
    profile = model_profile(agent_id)
    if hp > 80.0:
        return {
            "tier": "tier2_elite_verification",
            "model": tier2,
            "max_tokens": 32000,
            **profile,
        }
    if hp >= 40.0:
        model = tier2 if any(key in agent_id for key in ("Risk-R2", "Pricing-R2", "Compliance-R2", "Tax-R2", "Accounting-R2")) else tier1
        return {"tier": "tier1_standard", "model": model, "max_tokens": 8000, **profile}
    return {"tier": "tier0_exploration", "model": tier0, "max_tokens": 1500, **profile}


def locked_domains(agent_id: str) -> list[str]:
    agent = get(agent_id)
    locks = agent.get("domain_locks") or {}
    now = datetime.datetime.utcnow()
    active: list[str] = []
    expired: list[str] = []
    for domain, until in locks.items():
        try:
            until_dt = datetime.datetime.fromisoformat(str(until).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            expired.append(domain)
            continue
        if until_dt > now:
            active.append(domain)
        else:
            expired.append(domain)
    if expired:
        _db()[L].update_one({"agent_id": agent_id}, {"$unset": {f"domain_locks.{d}": "" for d in expired}})
    return active


def record_completion(agent_id: str, domain: str) -> dict:
    db = _db()
    agent = get(agent_id)
    recent = list(agent.get("recent_domains") or [])
    recent.append({"domain": domain, "ts": _now()})
    recent = recent[-3:]
    update_doc: dict[str, Any] = {"recent_domains": recent, "last_seen_at": _now()}
    locked = False
    if len(recent) == 3 and all(row.get("domain") == domain for row in recent):
        until = (datetime.datetime.utcnow() + datetime.timedelta(minutes=ROTATION_LOCK_MINUTES)).isoformat() + "Z"
        update_doc[f"domain_locks.{domain}"] = until
        update_doc["recent_domains"] = []
        locked = True
    db[L].update_one({"agent_id": agent_id}, {"$set": update_doc})
    return {"locked": locked, "domain": domain, "recent": recent}


def bump_warm_streak(agent_id: str) -> int:
    """P2-4: increment the warm successful_ticket_streak on a genuine own-work success.

    Returns the new streak value. Atomic $inc so concurrent workers do not clobber it.
    """
    db = _db()
    doc = db[L].find_one_and_update(
        {"agent_id": agent_id},
        {"$inc": {"telemetry_snapshot.successful_ticket_streak": 1},
         "$set": {"telemetry_snapshot.last_streak_event_at": _now()}},
        projection={"telemetry_snapshot.successful_ticket_streak": 1},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    snap = (doc or {}).get("telemetry_snapshot") or {}
    return int(snap.get("successful_ticket_streak") or 0)


def reset_warm_streak(agent_id: str, reason: str = "failure") -> None:
    """P2-4: reset the warm successful_ticket_streak to 0 on a genuine own-work failure.

    Without this, calculate_propulsion.warm_basis keeps adding 3 warm-minutes per stale
    streak point, so a failing agent reads as a healthy warm engine -- exactly the
    drag/credit mis-attribution the V4 layer is meant to avoid.
    """
    _db()[L].update_one(
        {"agent_id": agent_id},
        {"$set": {
            "telemetry_snapshot.successful_ticket_streak": 0,
            "telemetry_snapshot.last_streak_reset_at": _now(),
            "telemetry_snapshot.last_streak_reset_reason": str(reason)[:120],
        }},
        upsert=True,
    )


def ticket_reward(ticket: dict) -> tuple[str, float]:
    priority = int(ticket.get("priority", 5) or 5)
    task_type = str(ticket.get("task_type") or "").lower()
    if priority <= 1 or task_type in {"epic", "architectural", "architecture"}:
        return "critical_epic", TICKET_REWARDS["critical_epic"]
    if priority <= 2 or task_type in {"build", "code_fix", "feature", "rfc"}:
        return "feature_high", TICKET_REWARDS["feature_high"]
    return "maintenance_low", TICKET_REWARDS["maintenance_low"]


def _skill_name_for_ticket(ticket: dict) -> str:
    domain = str(ticket.get("target_domain") or "")
    if str(ticket.get("task_type") or "").lower() == "chaos_test":
        return "chaos_reproduction"
    return DOMAIN_SKILL_MAP.get(domain, "general_delivery")


def _ticket_text(ticket: dict) -> str:
    payload = ticket.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {"desc": str(payload)}
    return "\n".join([
        str(ticket.get("ticket_id", "")),
        str(ticket.get("target_domain", "")),
        str(ticket.get("task_type", "")),
        str(payload.get("desc", "")),
    ])


def _ticket_difficulty(ticket: dict) -> float:
    try:
        priority = int(ticket.get("priority", 5) or 5)
    except Exception:
        priority = 5
    difficulty = {0: 82.0, 1: 74.0, 2: 62.0, 3: 50.0, 4: 36.0, 5: 28.0}.get(priority, 45.0)
    text = _ticket_text(ticket)
    if MONEY_TEXT_RE.search(text):
        difficulty += 10.0
    if SECURITY_TEXT_RE.search(text):
        difficulty += 7.0
    if str(ticket.get("task_type") or "").lower() in {"epic", "architectural", "architecture"}:
        difficulty += 8.0
    return max(10.0, min(100.0, difficulty))


def _default_skill_row(agent: dict) -> dict:
    hp = _migrate_legacy_weight(agent.get("hp", agent.get("weight", DEFAULT_HP)))
    rank = int(agent.get("rank", _rank_for_hp(hp)))
    if rank >= 3:
        return {"mu": 78.0, "sigma": 14.0, "n": 0}
    if rank == 2:
        return {"mu": 62.0, "sigma": 24.0, "n": 0}
    return {"mu": 48.0, "sigma": 32.0, "n": 0}


def _skill_expected(mu: float, sigma: float, difficulty: float) -> float:
    z = (mu - difficulty) / max(8.0, sigma)
    return max(0.01, min(0.99, 1.0 / (1.0 + math.exp(-0.717 * z))))


def update_domain_skill(
    agent_id: str,
    ticket: dict,
    success: bool,
    quality: float = 1.0,
    skill_name: str | None = None,
) -> dict:
    """Bayesian/Elo-style domain skill update.

    HP remains separate. This controls future auction routing confidence.
    """
    db = _db()
    agent = get(agent_id)
    skill = skill_name or _skill_name_for_ticket(ticket)
    skills = agent.get("domain_skills") or {}
    row = dict(skills.get(skill) or _default_skill_row(agent))
    mu = float(row.get("mu", _default_skill_row(agent)["mu"]))
    sigma = float(row.get("sigma", _default_skill_row(agent)["sigma"]))
    difficulty = _ticket_difficulty(ticket)
    expected = _skill_expected(mu, sigma, difficulty)
    outcome = 1.0 if success else 0.0
    q = max(0.25, min(1.75, float(quality)))
    k = max(2.0, min(12.0, sigma / 2.7))
    mu = max(0.0, min(100.0, mu + (k * q * (outcome - expected))))
    if success:
        sigma = max(5.0, sigma * (0.92 if expected < 0.9 else 0.96))
    else:
        sigma = min(60.0, sigma * 1.08 + 1.5)
    n = int(row.get("n", 0) or 0) + 1
    update_doc = {
        f"domain_skills.{skill}": {
            "mu": round(mu, 4),
            "sigma": round(sigma, 4),
            "n": n,
            "last_ticket": ticket.get("ticket_id"),
            "last_success": bool(success),
            "last_expected": round(expected, 4),
            "last_difficulty": round(difficulty, 4),
            "updated_at": _now(),
        },
        "last_skill_update_at": _now(),
    }
    db[L].update_one({"agent_id": agent_id}, {"$set": update_doc}, upsert=True)
    return update_doc[f"domain_skills.{skill}"]


def apply_delta(agent_id: str, event: str, delta: float, extra: dict | None = None) -> dict:
    db = _db()
    cur = get(agent_id)
    old_hp = _migrate_legacy_weight(cur.get("hp", cur.get("weight", DEFAULT_HP)))
    old_rank = int(cur.get("rank", _rank_for_hp(old_hp)))
    new_hp = _clamp_hp(old_hp + delta)
    new_rank = _rank_for_hp(new_hp)
    inc_doc: dict[str, int] = {}
    if new_rank > old_rank:
        inc_doc["promotions"] = 1
    if new_rank < old_rank:
        inc_doc["demotions"] = 1
    update_doc = {
        "weight": new_hp,
        "hp": new_hp,
        "rank": new_rank,
        "last_event": event,
        "last_delta": float(delta),
        "last_event_at": _now(),
        "last_seen_at": _now(),
    }
    if extra:
        update_doc.update(extra)
    # P2-4 (2026-06-04): a genuine own-work failure resets the warm success streak in the
    # SAME write, so the V4 propulsion warm_basis stops crediting a now-cold agent.
    if event in STREAK_RESET_EVENTS:
        update_doc["telemetry_snapshot.successful_ticket_streak"] = 0
        update_doc["telemetry_snapshot.last_streak_reset_at"] = _now()
        update_doc["telemetry_snapshot.last_streak_reset_reason"] = str(event)[:120]
    op: dict[str, Any] = {"$set": update_doc}
    if inc_doc:
        op["$inc"] = inc_doc
    db[L].update_one({"agent_id": agent_id}, op, upsert=True)
    guillotine = new_hp < GUILLOTINE_HP
    if guillotine:
        autopsy_doc = write_agent_autopsy(agent_id, event, old_hp, new_hp)
        db[L].update_one(
            {"agent_id": agent_id},
            {"$set": {
                "active": False,
                "guillotine_at": _now(),
                "guillotine_reason": event,
                "last_autopsy_file": autopsy_doc.get("file"),
                "last_autopsy_summary": autopsy_doc.get("summary"),
            }},
        )
    return {
        "agent_id": agent_id,
        "old_hp": old_hp,
        "hp": new_hp,
        "rank": new_rank,
        "event": event,
        "delta": delta,
        "guillotine": guillotine,
    }


def update(
    agent_id: str,
    event: str,
    supervisor_id: str | None = None,
    age_minutes: float | None = None,
    completed_enqueued_at: str | None = None,
) -> dict:
    result = apply_delta(agent_id, event, EVENTS.get(event, 0.0))
    if supervisor_id and event in ("p0_online", "plausibility_fail"):
        update(supervisor_id, "supervisor_guilt")
    return result


def update_for_ticket(agent_id: str, ticket: dict) -> dict:
    reward_class, delta = ticket_reward(ticket)
    result = apply_delta(agent_id, reward_class, delta)
    result["skill"] = update_domain_skill(agent_id, ticket, success=True)
    rotation = record_completion(agent_id, str(ticket.get("target_domain") or "Unknown"))
    # P2-4: a genuine completion advances the warm success streak (V4 warm_basis).
    result["successful_ticket_streak"] = bump_warm_streak(agent_id)
    result["reward_class"] = reward_class
    result["rotation"] = rotation
    return result


def update_for_chaos_result(agent_id: str, ticket: dict, result_payload: dict | None) -> dict:
    """Score Chaos by reproducer quality, not by raw ticket completion count."""
    result_payload = result_payload or {}
    findings = list(result_payload.get("findings") or [])
    probes = list(result_payload.get("probes") or [])
    uat_db = str(result_payload.get("uat_db_name") or "")
    artifact = result_payload.get("chaos_probe")

    delta = 0.0
    reasons: list[str] = []
    if uat_db.endswith("_chaos_uat"):
        delta += 2.0
        reasons.append("sandbox_ok")
    else:
        delta -= 12.0
        reasons.append("sandbox_missing")
    if artifact and probes:
        delta += 3.0
        reasons.append("reproducible_probe")
    elif probes:
        delta += 1.0
        reasons.append("probe_no_artifact")
    else:
        delta -= 4.0
        reasons.append("no_probe")
    for finding in findings:
        severity = str(finding.get("severity") or "P2").upper()
        delta += CHAOS_FINDING_REWARDS.get(severity, 2.0)
        reasons.append(f"finding_{severity}")
    if not findings and probes and uat_db.endswith("_chaos_uat"):
        delta += 1.0
        reasons.append("clean_negative_probe")
    delta = max(-20.0, min(25.0, delta))
    scored = apply_delta(
        agent_id,
        "chaos_quality",
        delta,
        extra={
            "last_chaos_score": delta,
            "last_chaos_reasons": reasons,
            "last_chaos_findings": len(findings),
        },
    )
    scored["skill"] = update_domain_skill(
        agent_id,
        ticket,
        success=True,
        quality=1.0 + min(1.0, len(findings) * 0.2),
        skill_name="chaos_reproduction",
    )
    rotation = record_completion(agent_id, str(ticket.get("target_domain") or "Chaos"))
    # P2-4: a chaos success with a clean sandbox advances the streak; a sandbox-missing
    # run (the only hard failure signal here) resets it.
    if uat_db.endswith("_chaos_uat"):
        scored["successful_ticket_streak"] = bump_warm_streak(agent_id)
    else:
        reset_warm_streak(agent_id, reason="chaos_sandbox_missing")
        scored["successful_ticket_streak"] = 0
    scored["reward_class"] = "chaos_quality"
    scored["rotation"] = rotation
    scored["reasons"] = reasons
    return scored


def _idle_decay_window_start(minutes: int) -> str:
    """ISO timestamp of the window boundary used by decay_idle.

    Any agent whose last_idle_decay_at is AFTER this timestamp has already been
    decayed in the current window and must be skipped to enforce idempotency.
    """
    return (datetime.datetime.utcnow() - datetime.timedelta(minutes=minutes)).isoformat() + "Z"


def decay_idle(minutes: int = 30) -> dict:
    """Apply IDLE_DECAY_HP to agents that have not been seen within `minutes`.

    IDEMPOTENCY: Each agent carries a `last_idle_decay_at` timestamp. An agent is
    only decayed if last_idle_decay_at is NOT within the current window (i.e., it is
    either absent or older than the cutoff). The decay write sets last_idle_decay_at
    in the SAME atomic $set as the HP delta (via apply_delta extra), so a cron-overlap
    or restart cannot double-fire the decay within the same window.
    """
    now = datetime.datetime.utcnow()
    cutoff_iso = (now - datetime.timedelta(minutes=minutes)).isoformat() + "Z"
    # Window start = same as the cutoff for activity; any agent decayed more
    # recently than this is already in the window.
    window_start_iso = cutoff_iso
    db = _db()
    touched = killed = 0
    for agent in db[L].find({
        "active": {"$ne": False},
        "last_seen_at": {"$lt": cutoff_iso},
        # Idempotency guard: skip if already decayed within this window.
        # An absent last_idle_decay_at is always eligible (never been decayed).
        "$or": [
            {"last_idle_decay_at": {"$exists": False}},
            {"last_idle_decay_at": {"$lt": window_start_iso}},
        ],
    }):
        agent_id = agent["agent_id"]
        # Stamp last_idle_decay_at atomically in the same write as the HP delta.
        result = apply_delta(
            agent_id,
            "idle_decay",
            IDLE_DECAY_HP,
            extra={"last_idle_decay_at": _now()},
        )
        touched += 1
        if result["guillotine"]:
            killed += 1
    return {"decayed": touched, "guillotine": killed, "minutes": minutes}


def migrate_all() -> dict:
    db = _db()
    touched = 0
    for agent in db[L].find({}, {"agent_id": 1, "weight": 1, "hp": 1}):
        hp = _migrate_legacy_weight(agent.get("hp", agent.get("weight", DEFAULT_HP)))
        db[L].update_one(
            {"_id": agent["_id"]},
            {"$set": {"weight": hp, "hp": hp, "rank": _rank_for_hp(hp), "hp_migrated_at": _now()}},
        )
        touched += 1
    return {"migrated": touched}


def guillotine_self_if_needed(agent_id: str) -> bool:
    agent = get(agent_id)
    hp = _migrate_legacy_weight(agent.get("hp", agent.get("weight", DEFAULT_HP)))
    if hp >= GUILLOTINE_HP:
        return False
    db = _db()
    autopsy_doc = write_agent_autopsy(agent_id, "hp_below_20", hp, hp)
    db[L].update_one(
        {"agent_id": agent_id},
        {"$set": {
            "active": False,
            "guillotine_at": _now(),
            "guillotine_reason": "hp_below_20",
            "last_autopsy_file": autopsy_doc.get("file"),
            "last_autopsy_summary": autopsy_doc.get("summary"),
        }},
    )
    # Reaped self -> drop the dispatcher roster cache so this now-inactive agent stops
    # being auction-visible immediately (efficiency 2026-06-04).
    _invalidate_dispatcher_roster_cache()
    # P2-1 fix (2026-06-04): do NOT SIGTERM self. The caller (base_agent.loop) checks this
    # return value and returns cleanly, so the worker exits gracefully instead of taking an
    # abrupt signal that can truncate an in-flight ack/fail (worse on Windows).
    # enforce_guillotine_agents() remains the external backstop for agents that cannot self-exit.
    return True


def enforce_guillotine_agents(kill_process: bool = True) -> dict:
    """External guillotine reaper for agents that cannot self-terminate.

    Self-kill is not enough when an agent is idle, blocked in a child process,
    or already dead but still visible in the ledger. This deterministic pass
    marks HP<20 agents inactive and best-effort kills their recorded process.

    IDEMPOTENCY: If guillotine_at was set within the last GUILLOTINE_DEDUP_SECONDS
    (default 60 s), the agent has already been processed by a concurrent reaper run.
    We skip the autopsy/log write to avoid duplicate autopsy records and redundant
    process-kill attempts. The `active=False` flag may still be set defensively if
    somehow the agent came back active, but the expensive write path is skipped.
    """
    db = _db()
    killed = marked = skipped_duplicate = 0
    now = datetime.datetime.utcnow()
    dedup_cutoff = (now - datetime.timedelta(seconds=GUILLOTINE_DEDUP_SECONDS)).isoformat() + "Z"
    for agent in db[L].find({
        "active": {"$ne": False},
        "$or": [{"hp": {"$lt": GUILLOTINE_HP}}, {"weight": {"$lt": GUILLOTINE_HP}}],
    }):
        # Idempotency guard: skip if guillotine_at was set very recently
        # (within the dedup window) — indicates a concurrent reaper already ran.
        guillotine_at = agent.get("guillotine_at")
        if guillotine_at:
            g_ts = _parse_ts(guillotine_at)
            if g_ts and g_ts > now - datetime.timedelta(seconds=GUILLOTINE_DEDUP_SECONDS):
                # Already guillotined within dedup window. Just ensure active=False
                # without re-writing autopsy or log.
                flt = {"_id": agent["_id"]} if "_id" in agent else {"agent_id": agent["agent_id"]}
                db[L].update_one(
                    flt,
                    {"$set": {"active": False}},
                )
                skipped_duplicate += 1
                continue

        pid = agent.get("process_id")
        if kill_process and pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed += 1
            except Exception:
                pass
        hp = _migrate_legacy_weight(agent.get("hp", agent.get("weight", 0.0)))
        autopsy_doc = write_agent_autopsy(agent["agent_id"], "hp_below_20_external_reaper", hp, hp)
        flt = {"_id": agent["_id"]} if "_id" in agent else {"agent_id": agent["agent_id"]}
        db[L].update_one(
            flt,
            {"$set": {
                "active": False,
                "guillotine_at": _now(),
                "guillotine_reason": "hp_below_20_external_reaper",
                "last_autopsy_file": autopsy_doc.get("file"),
                "last_autopsy_summary": autopsy_doc.get("summary"),
            }},
        )
        marked += 1
    if marked:
        # Reaped one or more agents -> drop the dispatcher roster cache so they leave the
        # auction roster immediately rather than at the next TTL (efficiency 2026-06-04).
        _invalidate_dispatcher_roster_cache()
    return {
        "marked_inactive": marked,
        "process_kill_attempts": killed,
        "skipped_duplicate_guillotine": skipped_duplicate,
    }


def _soft_recycle_summary(agent_id: str, hp: float) -> str:
    return (
        f"Agent {agent_id} is alive but exhausted at {hp:.1f} HP. "
        "Context was wiped without changing score; replacement strategy must be smaller, "
        "test-first, and must avoid prior failed paths."
    )


def _active_ticket_lease(db, agent_id: str) -> dict | None:
    return db[Q].find_one(
        {"picked_by": agent_id, "status": {"$in": ["IN_PROGRESS", "WORKING"]}},
        {"_id": 0, "ticket_id": 1, "status": 1, "picked_at": 1},
    )


def soft_recycle_agent(agent_id: str, reason: str = "hp_20_to_40_context_exhaustion") -> dict:
    """Wipe low-HP agent context without killing the process or altering HP.

    This is a zero-token local lifecycle control. It preserves rank, HP, squad,
    and domain skill statistics; it clears only context/history fields that can
    trap the worker in a bad local minimum.
    """
    db = _db()
    agent = db[L].find_one({"agent_id": agent_id})
    if not agent:
        return {"agent_id": agent_id, "soft_recycled": False, "reason": "missing_agent"}
    hp = _migrate_legacy_weight(agent.get("hp", agent.get("weight", DEFAULT_HP)))
    if hp < GUILLOTINE_HP:
        return {"agent_id": agent_id, "soft_recycled": False, "reason": "below_guillotine"}
    if hp >= SOFT_RECYCLE_HP:
        return {"agent_id": agent_id, "soft_recycled": False, "reason": "hp_not_low_enough", "hp": hp}
    if str(agent.get("status") or agent.get("state") or "").upper() == "WORKING":
        return {"agent_id": agent_id, "soft_recycled": False, "reason": "agent_working", "hp": hp}
    # P1-6 fix (2026-06-04): the WORKING flag is never stamped on the ledger, so the guard above
    # was effectively dead. Harden against a lease read-lag by also refusing to recycle an agent
    # that was active in the last 5 minutes (likely mid-ticket).
    last_seen = _parse_ts(agent.get("last_seen_at"))
    if last_seen and last_seen > datetime.datetime.utcnow() - datetime.timedelta(minutes=5):
        return {"agent_id": agent_id, "soft_recycled": False, "reason": "recently_active", "hp": hp}
    active_lease = _active_ticket_lease(db, agent_id)
    if active_lease:
        return {
            "agent_id": agent_id,
            "soft_recycled": False,
            "reason": "active_ticket_lease",
            "hp": hp,
            "ticket_id": active_lease.get("ticket_id"),
        }

    now = datetime.datetime.utcnow()
    last_recycled = _parse_ts(agent.get("soft_recycle_at"))
    if last_recycled and last_recycled > now - datetime.timedelta(hours=SOFT_RECYCLE_COOLDOWN_HOURS):
        return {
            "agent_id": agent_id,
            "soft_recycled": False,
            "reason": "cooldown",
            "hp": hp,
            "last_soft_recycle_at": agent.get("soft_recycle_at"),
        }

    genesis_prompt = agent.get("system_genesis_prompt") or agent.get("genesis_prompt")
    summary = _soft_recycle_summary(agent_id, hp)
    context_epoch = _now()
    set_doc: dict[str, Any] = {
        "soft_recycle_at": context_epoch,
        "soft_recycle_reason": reason,
        "structural_break": {
            "at": context_epoch,
            "reason": reason,
            "reset_confidence_window": True,
            "hp_preserved": hp,
        },
        "context_epoch": context_epoch,
        "last_autopsy_summary": summary,
        "last_seen_at": context_epoch,
    }
    if genesis_prompt:
        set_doc["context_memory"] = [{"role": "system", "content": str(genesis_prompt)[:4000]}]
    unset_doc = {
        "message_history": "",
        "conversation_history": "",
        "prompt_history": "",
        "scratchpad": "",
        "working_memory": "",
        "failed_prompt_context": "",
        "last_large_context": "",
    }
    db[L].update_one({"agent_id": agent_id}, {"$set": set_doc, "$unset": unset_doc})

    doc = {
        "agent_id": agent_id,
        "ts": context_epoch,
        "reason": reason,
        "old_hp": hp,
        "new_hp": hp,
        "summary": summary,
        "soft_recycle": True,
    }
    db[AUTOPSY].update_one(
        {"agent_id": agent_id, "reason": reason, "soft_recycle": True},
        {"$set": doc},
        upsert=True,
    )
    _buffer_log(LOG, {
        "ts": context_epoch,
        "agent_id": agent_id,
        "event": "SOFT_RECYCLE_WIPE_EXECUTED",
        "status": "DONE",
        "level": "WARN",
        "msg": summary,
    })
    return {"agent_id": agent_id, "soft_recycled": True, "hp": hp, "summary": summary}


def enforce_soft_recycle_agents() -> dict:
    db = _db()
    scanned = recycled = 0
    results: list[dict] = []
    for agent in db[L].find({
        "active": {"$ne": False},
        "hp": {"$gte": GUILLOTINE_HP, "$lt": SOFT_RECYCLE_HP},
    }, {"agent_id": 1, "hp": 1}):
        scanned += 1
        result = soft_recycle_agent(agent["agent_id"])
        if result.get("soft_recycled"):
            recycled += 1
        results.append(result)
    return {"scanned": scanned, "soft_recycled": recycled, "results": results[:20]}


def write_agent_autopsy(agent_id: str, reason: str, old_hp: float, new_hp: float) -> dict:
    """Persist an agent-level autopsy so replacements inherit failure context."""
    db = _db()
    agent = db[L].find_one({"agent_id": agent_id}) or {"agent_id": agent_id}
    recent_logs = list(
        db[LOG]
        .find({"agent_id": agent_id}, {"_id": 0, "ts": 1, "event": 1, "ticket_id": 1, "status": 1, "level": 1, "msg": 1})
        .sort("ts", -1)
        .limit(8)
    )
    recent_tickets = list(
        db[Q]
        .find(
            {"$or": [{"picked_by": agent_id}, {"progress.by": agent_id}]},
            {"_id": 0, "ticket_id": 1, "status": 1, "target_domain": 1, "last_error": 1, "repair_mode": 1},
        )
        .sort("picked_at", -1)
        .limit(8)
    )
    log_text = " | ".join(
        f"{x.get('event')}:{x.get('status')}:{str(x.get('msg',''))[:120]}" for x in recent_logs[:4]
    )
    if "idle_decay" in reason or agent.get("last_event") == "idle_decay":
        summary = "Agent died by idle decay / low useful throughput; replacement must take active tickets only and avoid sitting in idle fleet."
    elif any("local CI gate failed" in str(t.get("last_error", "")) for t in recent_tickets):
        summary = "Agent died after CI-gated failures; replacement must inspect prior PR artifacts and add/fix focused tests before retry."
    elif any("git" in str(x.get("msg", "")).lower() for x in recent_logs):
        summary = "Agent died amid git/worktree infra churn; replacement must avoid lock contention and use compact prompt/worktree recovery."
    else:
        summary = "Agent fell below HP threshold; replacement must avoid the recent failed strategy and use bounded, test-first changes."

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_agent = re.sub(r"[^A-Za-z0-9_.-]+", "-", agent_id).strip("-")[:80]
    path = os.path.join(APPEALS_DIR, f"{timestamp}_GUILLOTINE_{safe_agent}.md")
    os.makedirs(APPEALS_DIR, exist_ok=True)
    body = textwrap.dedent(
        f"""\
        # Agent Guillotine Autopsy

        - agent_id: {agent_id}
        - squad: {agent.get('squad')}
        - reason: {reason}
        - old_hp: {old_hp}
        - new_hp: {new_hp}
        - last_event: {agent.get('last_event')}
        - last_delta: {agent.get('last_delta')}
        - process_id: {agent.get('process_id')}

        ## Root Cause
        {summary}

        ## Do Not Repeat
        Do not continue the same idle/failed execution path. Do not retry old large-context prompts blindly. Do not compete for git worktree locks without bounded concurrency.

        ## Required Next Approach
        Replacement starts at fresh context, reads this autopsy summary, picks only eligible tickets through the dispatcher, and must produce focused diffs with local tests before claiming success.

        ## Recent Logs
        {log_text or 'none'}

        ## Recent Tickets
        {recent_tickets}
        """
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    doc = {
        "agent_id": agent_id,
        "ts": _now(),
        "reason": reason,
        "old_hp": old_hp,
        "new_hp": new_hp,
        "summary": summary,
        "file": path,
        "recent_logs": recent_logs,
        "recent_tickets": recent_tickets,
    }
    db[AUTOPSY].update_one({"agent_id": agent_id, "reason": reason}, {"$set": doc}, upsert=True)
    try:
        _buffer_log(LOG, {
            "ts": _now(),
            "agent_id": "guillotine-reaper",
            "event": "agent_autopsy_written",
            "status": "DONE",
            "level": "WARN",
            "msg": f"{agent_id}: {summary} file={path}",
        })
    except Exception:
        pass
    return doc


def rebalance_fleet() -> dict:
    return {"rebalanced": False, "reason": "absolute_hp_no_normalization"}


def leaderboard(n: int = 5):
    db = _db()
    projection = {"_id": 0, "agent_id": 1, "weight": 1, "hp": 1, "rank": 1, "active": 1}
    active_q = {"active": {"$ne": False}}
    dead_q = {"active": False}
    top = list(db[L].find(active_q, projection).sort("weight", -1).limit(n))
    bottom = list(db[L].find(active_q, projection).sort("weight", 1).limit(n))
    dead = list(db[L].find(dead_q, projection).sort("weight", 1).limit(n))
    return {"top": top, "bottom": bottom, "dead": dead}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "decay":
        print(decay_idle(int(sys.argv[2]) if len(sys.argv) > 2 else 30))
    elif len(sys.argv) > 1 and sys.argv[1] == "migrate":
        print(migrate_all())
    elif len(sys.argv) > 1 and sys.argv[1] == "init-governance-v3":
        print(initialize_governance_v3_schema())
    elif len(sys.argv) > 1 and sys.argv[1] == "sync-hazard-shadow":
        print(sync_hazard_rates_shadow())
    elif len(sys.argv) > 1 and sys.argv[1] == "soft-recycle":
        print(enforce_soft_recycle_agents())
    else:
        print(leaderboard())
