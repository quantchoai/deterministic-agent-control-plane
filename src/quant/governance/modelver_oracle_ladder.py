"""modelver_oracle_ladder.py -- V6 deterministic verification whitelist ("Oracle Ladder").

PRIVATE / OFFLINE control-plane module. Standard-library + governance_params only.
NO LLM in the plane; NO Mongo; NO mutation of ledger.py. This module is a *pure,
deterministic gate* the integrate phase will wire in front of ledger.update_domain_skill.

------------------------------------------------------------------------------------
WHY THIS EXISTS  (V6 mandate, sections 1.4 / 2 / 4 / 5)
------------------------------------------------------------------------------------
The V2 learning rule (ledger.update_domain_skill) moves an agent's credit (mu/sigma)
on EVERY ticket outcome. That is dangerous if the "outcome" came from a probe that can
be gamed or that is merely a noisy/LLM-judge signal. V6 doctrine #4:

    "Severity matches the oracle's determinism. Deterministic ground truth may drive
     hard actions (HP/guillotine). Probabilistic / LLM-judge signals may only adjust
     sigma, raise a shadow flag, or route to committee. A noisy sensor never triggers
     an irreversible kill."

and section 4 (Oracle whitelist, strongest first):

    1. Deterministic ground truth : compiler / unit tests / type+schema / invariant
                                     (NAV/VaR sum-checks) / replay+idempotency.
    2. Structural probes          : forced JSON schema, required fields, resolvable
                                     citations, numeric plausibility bounds.
    3. Cross-source verification  : output re-checked vs the authoritative source/DB.
    4. Redundancy / consensus     : N independent agents; divergence RAISES sigma,
                                     never kills.
    5. LLM-as-Judge               : QUARANTINED -- advisory/shadow only, never a sole
                                     or hard trigger.

This module turns that prose into a deterministic registry + a single decision
function. The integrate phase calls `eligible_skill_update(probe, ...)`; it returns a
*bounded, signed* mu/sigma delta ONLY for whitelisted probes, and otherwise returns a
zero-effect verdict (shadow-only). Severity is clamped to the probe's determinism tier
so a fuzzy probe can never punch above its weight.

------------------------------------------------------------------------------------
THE MATH  (deterministic, closed-form -- mirrors ledger.update_domain_skill)
------------------------------------------------------------------------------------
ledger's live rule, for a probe with binary outcome `o in {0,1}` and quality q:

    expected = sigmoid(0.717 * (mu - difficulty) / max(8, sigma))      # V2 sigmoid
    k        = clamp(sigma / 2.7, 2, 12)                               # Elo-style step
    mu'      = clamp(mu + k * q * (o - expected), 0, 100)
    sigma'   = sigma * 0.92..0.96      if success
               min(60, sigma*1.08 + 1.5) if failure

The Oracle Ladder wraps this with a per-tier *trust weight* w_T in [0,1] and a
*severity ceiling* so the realized delta is

    mu_delta_raw    = k * q * (o - expected)
    mu_delta        = w_T * mu_delta_raw                       (trust-attenuated)
    |mu_delta|      <= MU_DELTA_CEILING[T]                     (severity ceiling)

and a tier-dependent sigma policy:

  * Deterministic / structural / cross-source  -> may TIGHTEN sigma on a verified
    success (information gain) and may LOOSEN on a verified failure (lost confidence).
  * Consensus                                  -> may ONLY RAISE sigma (divergence =
    "we are less sure"); never tightens, never moves mu downward as a kill.
  * LLM-judge / none                           -> NO mu change, NO sigma tighten; at
    most a small sigma *widening* shadow flag. Quarantined.

The trust weight w_T is itself derived, not hand-waved: it is the lower 95%-confidence
bound on the oracle's own historical agreement rate with deterministic ground truth
(Wilson score interval). A consensus oracle that has only ever agreed with the
deterministic oracle 60% of the time gets w_T ~= 0.6, not 1.0. Deterministic oracles
are pinned to w_T = 1.0 by definition (they ARE ground truth).

Anti-Goodhart (section 5): a whitelisted probe whose task `difficulty` is trivially low
AND whose information content is near-zero is down-weighted, and a *declined* eligible
task yields a negative mu delta -- "safe garbage" cannot farm credit.

Everything here is deterministic given its inputs (no clock, no RNG, no network), so the
same probe always yields the same verdict -- a hard requirement for an auditable money/
security control plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from quant.governance import governance_params as gp

# ----------------------------------------------------------------------------------
# 0. Constants sourced from the governance registry (single source of truth).
#    We deliberately reuse the SAME numbers ledger.py / V2_CREDIT use so the gate is a
#    faithful wrapper of the live learning rule, not a divergent re-implementation.
# ----------------------------------------------------------------------------------
_MU_LO, _MU_HI = gp.V2_CREDIT["MU_CLAMP"]                 # (0.0, 100.0)
_SIGMA_LO, _SIGMA_HI = gp.V2_CREDIT["SIGMA_CLAMP"]        # (3.0, 60.0)

# Elo step bounds + sigmoid temperature -- identical to ledger.update_domain_skill /
# ledger._skill_expected so the wrapped delta matches the live engine exactly.
_K_MIN, _K_MAX = 2.0, 12.0
_K_DIVISOR = 2.7
_SIGMOID_TEMP = 0.717
_SIGMA_SUCCESS_EASY = 0.92    # expected < 0.9 -> bigger tighten
_SIGMA_SUCCESS_HARD = 0.96    # expected >= 0.9 -> smaller tighten (already confident)
_SIGMA_FAIL_MULT = 1.08
_SIGMA_FAIL_ADD = 1.5
_SIGMA_FLOOR_UPDATE = 5.0     # ledger floors sigma at 5.0 on a success tighten
_Q_LO, _Q_HI = 0.25, 1.75     # ledger clamps quality into this band


# ----------------------------------------------------------------------------------
# 1. Determinism tiers (the ladder). Order = strongest first; index = severity rank.
# ----------------------------------------------------------------------------------
TIER_DETERMINISTIC = "deterministic"   # compiler/unit/invariant/replay  -> ground truth
TIER_STRUCTURAL = "structural"         # schema/required-fields/bounds
TIER_CROSS_SOURCE = "cross_source"     # re-checked vs authoritative source/DB
TIER_CONSENSUS = "consensus"           # N independent agents; divergence raises sigma
TIER_LLM_JUDGE = "llm_judge"           # QUARANTINED, advisory/shadow only
TIER_NONE = "none"                     # no oracle -> no credit movement at all

TIER_ORDER = (
    TIER_DETERMINISTIC,
    TIER_STRUCTURAL,
    TIER_CROSS_SOURCE,
    TIER_CONSENSUS,
    TIER_LLM_JUDGE,
    TIER_NONE,
)
_TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}

# Tiers whose probes are allowed to move mu at all (the *whitelist*).
# LLM-judge and none are quarantined: shadow-only, zero credit movement.
_MU_ELIGIBLE_TIERS = frozenset({
    TIER_DETERMINISTIC, TIER_STRUCTURAL, TIER_CROSS_SOURCE, TIER_CONSENSUS,
})
# Only deterministic ground truth may drive HARD actions (HP/guillotine).
_HARD_ACTION_TIERS = frozenset({TIER_DETERMINISTIC})
# Tiers permitted to TIGHTEN sigma (claim information gain). Consensus may only widen.
_SIGMA_TIGHTEN_TIERS = frozenset({
    TIER_DETERMINISTIC, TIER_STRUCTURAL, TIER_CROSS_SOURCE,
})

# Severity ceiling on |mu_delta| per tier. Deterministic gets the full Elo step;
# weaker tiers are capped so a fuzzy oracle cannot swing credit hard even on a
# nominal "success". These match the V2 step band (k in [2,12]) scaled down by trust.
MU_DELTA_CEILING = {
    TIER_DETERMINISTIC: _K_MAX,          # 12.0 -- full authority
    TIER_STRUCTURAL: 6.0,                # half
    TIER_CROSS_SOURCE: 6.0,
    TIER_CONSENSUS: 3.0,                 # quarter; consensus is weak evidence for mu
    TIER_LLM_JUDGE: 0.0,                 # no mu authority at all
    TIER_NONE: 0.0,
}

# Maximum sigma WIDENING a quarantined (consensus divergence / llm-judge) probe may
# request as a shadow uncertainty flag. Never tightens, only flags "less sure".
_SHADOW_SIGMA_WIDEN_MAX = 2.0


# ----------------------------------------------------------------------------------
# 2. The probe registry. A *named* oracle probe with a determinism tier and a tracked
#    historical agreement rate vs deterministic ground truth (drives its trust weight).
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class OracleProbe:
    """A whitelisted verification probe.

    name        : stable identifier (e.g. "unit_pytest", "nav_invariant_recon").
    tier        : determinism tier (one of TIER_*).
    description : human note.
    agree_hits  : # times this probe's verdict matched deterministic ground truth.
    agree_total : # times it was cross-checked against deterministic ground truth.
                  (Deterministic probes are ground truth, so they are pinned to w=1.)
    """
    name: str
    tier: str
    description: str = ""
    agree_hits: int = 0
    agree_total: int = 0

    def __post_init__(self) -> None:
        if self.tier not in _TIER_RANK:
            raise ValueError(f"unknown determinism tier: {self.tier!r}")
        if self.agree_hits < 0 or self.agree_total < 0 or self.agree_hits > self.agree_total:
            raise ValueError("invalid agreement counts")


def _wilson_lower_bound(hits: int, total: int, z: float = 1.95996398454) -> float:
    """Lower bound of the 95% Wilson score interval for a binomial proportion.

    Deterministic, closed-form. Used as the oracle's TRUST WEIGHT: we credit a probe
    only as much as its *worst plausible* historical agreement with ground truth, so a
    probe with few observations is conservatively distrusted (wide interval -> low LB).

        p_hat = hits/total
        center = (p_hat + z^2/2n) / (1 + z^2/n)
        margin = z*sqrt( p_hat(1-p_hat)/n + z^2/4n^2 ) / (1 + z^2/n)
        LB = center - margin
    """
    if total <= 0:
        return 0.0
    n = float(total)
    p_hat = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)) / denom
    return max(0.0, min(1.0, center - margin))


def trust_weight(probe: OracleProbe) -> float:
    """Trust weight w_T in [0,1] for a probe.

    Deterministic probes ARE ground truth -> pinned to 1.0.
    All weaker tiers earn trust = Wilson lower bound of their agreement with the
    deterministic oracle (conservative; unproven probes are distrusted).
    """
    if probe.tier == TIER_DETERMINISTIC:
        return 1.0
    if probe.agree_total <= 0:
        # No track record -> minimal trust floor so a brand-new structural/cross-source
        # probe can still nudge (not zero), but a consensus/judge one effectively can't.
        floor = {
            TIER_STRUCTURAL: 0.50,
            TIER_CROSS_SOURCE: 0.50,
            TIER_CONSENSUS: 0.20,
        }.get(probe.tier, 0.0)
        return floor
    return _wilson_lower_bound(probe.agree_hits, probe.agree_total)


# Default registry. The integrate phase may extend this; the *tiers* are the binding
# whitelist, not the individual names. Agreement counts here are illustrative defaults
# (deterministic pinned to 1.0 regardless).
DEFAULT_REGISTRY: dict[str, OracleProbe] = {}


def register_probe(probe: OracleProbe, registry: dict[str, OracleProbe] | None = None) -> None:
    reg = DEFAULT_REGISTRY if registry is None else registry
    reg[probe.name] = probe


def _seed_default_registry() -> None:
    seeds = [
        # ---- Tier 1: deterministic ground truth (w = 1.0, may drive HARD actions) ----
        OracleProbe("unit_pytest", TIER_DETERMINISTIC, "pytest unit suite (pass/fail)"),
        OracleProbe("compiler_typecheck", TIER_DETERMINISTIC, "compile / mypy / tsc"),
        OracleProbe("nav_invariant_recon", TIER_DETERMINISTIC,
                    "NAV/VaR sum-check reconciliation invariant"),
        OracleProbe("replay_idempotency", TIER_DETERMINISTIC,
                    "deterministic replay / idempotency check"),
        OracleProbe("golden_fixture", TIER_DETERMINISTIC,
                    "golden-master byte/structure equality"),
        # ---- Tier 2: structural probes ----
        OracleProbe("json_schema", TIER_STRUCTURAL, "forced JSON schema validation",
                    agree_hits=92, agree_total=100),
        OracleProbe("numeric_bounds", TIER_STRUCTURAL, "numeric plausibility bounds",
                    agree_hits=88, agree_total=100),
        # ---- Tier 3: cross-source verification ----
        OracleProbe("source_recheck", TIER_CROSS_SOURCE,
                    "output re-checked vs authoritative source/DB",
                    agree_hits=80, agree_total=100),
        # ---- Tier 4: redundancy / consensus (sigma-only) ----
        OracleProbe("agent_consensus", TIER_CONSENSUS,
                    "N independent agents; divergence raises sigma",
                    agree_hits=63, agree_total=100),
        # ---- Tier 5: LLM-as-judge (QUARANTINED, shadow-only) ----
        OracleProbe("llm_judge", TIER_LLM_JUDGE, "LLM-as-judge advisory (quarantined)",
                    agree_hits=55, agree_total=100),
    ]
    for p in seeds:
        DEFAULT_REGISTRY[p.name] = p


_seed_default_registry()


# ----------------------------------------------------------------------------------
# 3. The V2 delta math (faithful re-derivation of ledger.update_domain_skill, pure).
# ----------------------------------------------------------------------------------
def _sigmoid_expected(mu: float, sigma: float, difficulty: float) -> float:
    """Mirrors ledger._skill_expected: P(success) sigmoid, clamped [0.01, 0.99]."""
    z = (mu - difficulty) / max(8.0, sigma)
    return max(0.01, min(0.99, 1.0 / (1.0 + math.exp(-_SIGMOID_TEMP * z))))


def _elo_step(sigma: float) -> float:
    """Mirrors ledger: k = clamp(sigma/2.7, 2, 12)."""
    return max(_K_MIN, min(_K_MAX, sigma / _K_DIVISOR))


def _raw_mu_delta(mu: float, sigma: float, difficulty: float, outcome: float, quality: float) -> float:
    """The unattenuated V2 mu step: k * q * (outcome - expected)."""
    expected = _sigmoid_expected(mu, sigma, difficulty)
    k = _elo_step(sigma)
    q = max(_Q_LO, min(_Q_HI, float(quality)))
    return k * q * (outcome - expected)


def _sigma_after(sigma: float, success: bool, expected: float) -> float:
    """The V2 sigma move (deterministic-grade success/failure), ledger-faithful."""
    if success:
        mult = _SIGMA_SUCCESS_EASY if expected < 0.9 else _SIGMA_SUCCESS_HARD
        return max(_SIGMA_FLOOR_UPDATE, sigma * mult)
    return min(_SIGMA_HI, sigma * _SIGMA_FAIL_MULT + _SIGMA_FAIL_ADD)


# ----------------------------------------------------------------------------------
# 4. Anti-Goodhart guard.
# ----------------------------------------------------------------------------------
def anti_goodhart_factor(difficulty: float, info_content: float) -> float:
    """Down-weight trivial / near-empty 'safe garbage' so it cannot farm credit.

    - info_content < 0.2 (near-empty output): factor -> 0.1 (almost no credit).
    - otherwise scale by difficulty (harder verified work earns its full step):
          factor = 0.7 + 0.3 * clamp(difficulty/100, 0, 1)
      so a trivial task tops out at 0.7 of the step, a hard one at 1.0.
    Deterministic, bounded in [0.1, 1.0].
    """
    if info_content < 0.2:
        return 0.1
    return 0.7 + 0.3 * max(0.0, min(1.0, difficulty / 100.0))


# ----------------------------------------------------------------------------------
# 5. The verdict object + the core decision function.
# ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class SkillUpdateVerdict:
    """Result of running a probe through the oracle ladder.

    eligible          : whether this probe may move mu at all (whitelisted tier).
    tier              : the probe's determinism tier.
    trust_weight      : derived w_T in [0,1].
    mu_delta          : signed, bounded change to apply to mu (0.0 if not eligible).
    sigma_delta       : signed change to apply to sigma (negative = tighten).
    hard_action_allowed : may this probe drive HP/guillotine? (deterministic only)
    shadow_only       : True => log, do not mutate credit (quarantined or non-whitelisted).
    reason            : machine string for the audit log.
    components        : breakdown for traceability.
    """
    eligible: bool
    tier: str
    trust_weight: float
    mu_delta: float
    sigma_delta: float
    hard_action_allowed: bool
    shadow_only: bool
    reason: str
    components: dict[str, Any] = field(default_factory=dict)

    def apply(self, mu: float, sigma: float) -> tuple[float, float]:
        """Apply this verdict to a (mu, sigma) pair, clamped to governance bounds.

        Pure: returns the new (mu, sigma); does NOT touch the ledger. The integrate
        phase calls this and persists via ledger if it chooses.
        """
        new_mu = max(_MU_LO, min(_MU_HI, mu + self.mu_delta))
        new_sigma = max(_SIGMA_LO, min(_SIGMA_HI, sigma + self.sigma_delta))
        return new_mu, new_sigma


def _resolve_probe(probe: Any, registry: dict[str, OracleProbe] | None) -> OracleProbe | None:
    """Accept an OracleProbe, a registry name, or a dict {name|tier,...}."""
    reg = DEFAULT_REGISTRY if registry is None else registry
    if isinstance(probe, OracleProbe):
        return probe
    if isinstance(probe, str):
        return reg.get(probe)
    if isinstance(probe, dict):
        name = probe.get("name")
        if name and name in reg:
            return reg[name]
        tier = probe.get("tier")
        if tier is not None:
            # Ad-hoc probe described inline; must still be a known tier and, if not
            # deterministic, carry agreement counts (else it gets only the trust floor).
            try:
                return OracleProbe(
                    name=str(name or f"adhoc_{tier}"),
                    tier=str(tier),
                    description=str(probe.get("description", "ad-hoc")),
                    agree_hits=int(probe.get("agree_hits", 0) or 0),
                    agree_total=int(probe.get("agree_total", 0) or 0),
                )
            except ValueError:
                return None
    return None


def eligible_skill_update(
    probe: Any,
    *,
    mu: float = 50.0,
    sigma: float = 30.0,
    difficulty: float = 50.0,
    outcome: float = 1.0,
    quality: float = 1.0,
    info_content: float = 1.0,
    declined: bool = False,
    registry: dict[str, OracleProbe] | None = None,
) -> SkillUpdateVerdict:
    """Decide whether `probe` may update mu/sigma, and by how much.

    This is the SINGLE entry point the integrate phase wires in front of
    ledger.update_domain_skill. It NEVER mutates anything; it returns a verdict.

    Whitelist semantics:
      * Non-whitelisted probe (unknown name, llm_judge, none, or unrecognized tier)
        -> shadow_only, mu_delta == 0, no sigma tighten. (Test: "no mu/sigma change".)
      * Whitelisted deterministic probe -> bounded, signed mu_delta within
        MU_DELTA_CEILING; full hard-action authority; sigma may tighten on success.
      * Structural / cross-source -> attenuated by trust weight + lower severity ceiling.
      * Consensus -> mu nudge only on agreement-success; on divergence (outcome 0)
        it may ONLY RAISE sigma, never move mu down as a kill.

    Args mirror the inputs ledger.update_domain_skill derives from a ticket:
      mu, sigma, difficulty : current agent credit + ticket difficulty.
      outcome               : 1.0 success / 0.0 failure (deterministic verdict of probe).
      quality               : ledger quality multiplier (clamped 0.25..1.75).
      info_content          : anti-Goodhart info signal in [0,1].
      declined              : agent refused an eligible task (anti-Goodhart penalty).
    """
    resolved = _resolve_probe(probe, registry)

    # --- Declined eligible task: anti-Goodhart penalty, only meaningful if the probe
    #     itself is whitelisted (you can't be penalized via a non-oracle). ---
    if resolved is None:
        return SkillUpdateVerdict(
            eligible=False, tier=TIER_NONE, trust_weight=0.0,
            mu_delta=0.0, sigma_delta=0.0, hard_action_allowed=False,
            shadow_only=True, reason="probe_not_whitelisted",
            components={"probe": repr(probe)},
        )

    tier = resolved.tier
    w = trust_weight(resolved)

    # --- Quarantine: llm_judge / none -> shadow only. At most a small sigma widening
    #     flag (uncertainty), NEVER a mu move and NEVER a sigma tighten. ---
    if tier not in _MU_ELIGIBLE_TIERS:
        widen = min(_SHADOW_SIGMA_WIDEN_MAX, 0.5) if outcome < 1.0 else 0.0
        return SkillUpdateVerdict(
            eligible=False, tier=tier, trust_weight=w,
            mu_delta=0.0, sigma_delta=widen, hard_action_allowed=False,
            shadow_only=True, reason="quarantined_non_deterministic",
            components={"probe": resolved.name, "tier": tier},
        )

    success = outcome >= 1.0
    expected = _sigmoid_expected(mu, sigma, difficulty)

    # --- Anti-Goodhart: declined eligible task costs credit (negative, ceiling-bounded). ---
    if declined:
        penalty = -min(MU_DELTA_CEILING[tier], abs(_raw_mu_delta(mu, sigma, difficulty, 1.0, quality)) * 0.5)
        return SkillUpdateVerdict(
            eligible=True, tier=tier, trust_weight=w,
            mu_delta=penalty, sigma_delta=0.0,
            hard_action_allowed=(tier in _HARD_ACTION_TIERS),
            shadow_only=False, reason="anti_goodhart_declined_eligible",
            components={"probe": resolved.name, "tier": tier, "penalty": penalty},
        )

    # --- Consensus: divergence (failure) may ONLY raise sigma, never move mu down. ---
    if tier == TIER_CONSENSUS and not success:
        widen = min(_SHADOW_SIGMA_WIDEN_MAX, (1.0 - w) * _SHADOW_SIGMA_WIDEN_MAX + 0.5)
        return SkillUpdateVerdict(
            eligible=True, tier=tier, trust_weight=w,
            mu_delta=0.0, sigma_delta=widen, hard_action_allowed=False,
            shadow_only=False, reason="consensus_divergence_sigma_only",
            components={"probe": resolved.name, "tier": tier, "expected": round(expected, 4)},
        )

    # --- The whitelisted, trust-attenuated, severity-capped mu delta. ---
    raw = _raw_mu_delta(mu, sigma, difficulty, 1.0 if success else 0.0, quality)
    ag = anti_goodhart_factor(difficulty, info_content)
    attenuated = w * ag * raw
    ceiling = MU_DELTA_CEILING[tier]
    mu_delta = max(-ceiling, min(ceiling, attenuated))

    # --- sigma policy by tier. ---
    if tier in _SIGMA_TIGHTEN_TIERS:
        # Information gain/loss permitted: re-derive ledger's sigma move, then scale
        # the *magnitude* of the move by trust (a half-trusted probe learns half as
        # fast). Deterministic (w=1) reproduces ledger exactly.
        sigma_target = _sigma_after(sigma, success, expected)
        sigma_delta = w * (sigma_target - sigma)
    else:
        # Consensus success: weak positive evidence -> tiny tighten, trust-scaled, but
        # never below the structural rate; bounded so consensus can't over-tighten.
        sigma_delta = w * (max(_SIGMA_FLOOR_UPDATE, sigma * _SIGMA_SUCCESS_HARD) - sigma) * 0.5

    return SkillUpdateVerdict(
        eligible=True, tier=tier, trust_weight=round(w, 6),
        mu_delta=mu_delta, sigma_delta=sigma_delta,
        hard_action_allowed=(tier in _HARD_ACTION_TIERS),
        shadow_only=False, reason="whitelisted_deterministic_update",
        components={
            "probe": resolved.name, "tier": tier,
            "expected": round(expected, 6), "raw_mu_delta": round(raw, 6),
            "anti_goodhart_factor": round(ag, 6), "ceiling": ceiling,
            "success": success,
        },
    )


def hard_action_allowed(probe: Any, registry: dict[str, OracleProbe] | None = None) -> bool:
    """Convenience: may this probe drive an irreversible HP/guillotine action?

    True ONLY for whitelisted deterministic ground-truth probes (V6 doctrine #4).
    """
    resolved = _resolve_probe(probe, registry)
    return bool(resolved and resolved.tier in _HARD_ACTION_TIERS)


__all__ = [
    "OracleProbe", "SkillUpdateVerdict",
    "TIER_DETERMINISTIC", "TIER_STRUCTURAL", "TIER_CROSS_SOURCE",
    "TIER_CONSENSUS", "TIER_LLM_JUDGE", "TIER_NONE", "TIER_ORDER",
    "MU_DELTA_CEILING", "DEFAULT_REGISTRY",
    "register_probe", "trust_weight", "anti_goodhart_factor",
    "eligible_skill_update", "hard_action_allowed",
]


if __name__ == "__main__":
    # Read-only demo; no side effects.
    det = eligible_skill_update("unit_pytest", mu=60, sigma=25, difficulty=55, outcome=1.0)
    judge = eligible_skill_update("llm_judge", mu=60, sigma=25, difficulty=55, outcome=1.0)
    bogus = eligible_skill_update("not_a_real_probe", mu=60, sigma=25, difficulty=55, outcome=1.0)
    print("deterministic success :", round(det.mu_delta, 4), round(det.sigma_delta, 4),
          "hard=", det.hard_action_allowed)
    print("llm-judge (quarantined):", judge.mu_delta, judge.sigma_delta, "shadow=", judge.shadow_only)
    print("non-whitelisted        :", bogus.mu_delta, bogus.sigma_delta, "shadow=", bogus.shadow_only)
