# A Deterministic Control Plane for Governing Fleets of Autonomous Agents
### Theory, Architecture, and Empirical Results

**Status:** Public / open-core doctrine. Portable across domains.
**Version:** 2026-06-05.

> This is the *open doctrine* — the architecture and the general mathematics. Fitted
> parameters, calibration data, and certain advanced risk extensions are retained in a
> private implementation and are referenced here only by shape.

---

## Abstract

Systems increasingly dispatch work to *fleets of autonomous agents* — LLM workers, trading
strategies, microservices, or any pool of interchangeable actors with measurable outcomes.
The common pattern is to let **another LLM orchestrate** the fleet, which is expensive,
non-deterministic, and unauditable. We describe a **deterministic, no-LLM-in-the-loop control
plane** that decides which agents receive work and capital, rates them on measured
performance, removes the failing, rewards the rising, and prices tail risk — all in closed
form, reproducibly, with a full audit trail, and *fail-closed* on critical actions. On
reproducible synthetic workloads the policy lifts first-attempt success ≈ **+26%** over FIFO
with a ≈ **−38%** retry reduction; on one operator's real 689-task history it lifts ≈ **+17%**.
Routing costs an estimated **10⁴–10⁶×** less than an LLM manager. *All results are validated in
shadow/replay; live activation is a pending operator decision.* The architecture is
**domain-agnostic**: any fleet of competing workers maps onto it unchanged.

---

## 1. Problem, Prior Work, and Design Principles

### 1.1 The governance questions

A fleet of agents raises **six** recurring questions, which this architecture answers as six
layers (V1–V6):

1. **Survival** — which agents are healthy enough to keep? *(V1)*
2. **Credit** — how good is each agent, and how *sure* are we? *(V2)*
3. **Hazard** — which agents are about to fail, before they do? *(V3)*
4. **Propulsion** — which agents are improving and deserve more? *(V4)*
5. **Allocation** — given cost, value, and *tail risk*, who gets the next job and the capital? *(V5)*
6. **Promotion** — how do we roll out a new policy without betting the system on it? *(V6)*

### 1.2 Related work

- **LLM-orchestrated fleets** (AutoGen, CrewAI, LangGraph-style supervisors) place an LLM in
  the routing loop: flexible, but per-decision-costly, non-deterministic, and hard to audit.
- **Multi-armed and contextual bandits** [Auer et al. 2002; Garivier & Cappé 2011] solve
  explore/exploit allocation, but standard formulations do not carry a survival reservoir,
  coherent tail pricing, or a fail-closed authority model.
- **Financial risk budgeting** [Rockafellar & Uryasev 2000] supplies coherent tail measures
  (CVaR/Expected Shortfall) but is not normally applied to *agent dispatch*.

**Contribution.** The novelty is the *combination*: a **deterministic** plane that fuses a
risk-bearing survival ledger, measured-credit gating, a coherent-tail-priced allocation
auction, and **severity-matched kill authority** (only a deterministic rule may take an
irreversible action). To our knowledge no published agent-orchestration system unifies these.

### 1.3 Design principles (the invariants)

- **P1 — Determinism.** Identical `(state, events)` ⇒ identical decisions, byte-for-byte. No
  RNG, clock, or network in the decision path. This is what makes the plane *auditable* and
  *replayable*.
- **P2 — No LLM in the decision loop.** The plane is pure arithmetic; LLMs are the *workers*,
  never the *judge*.
- **P3 — Fail-closed on critical actions.** An unproven agent is *never* routed to a
  money/security-critical action; the default is denial.
- **P4 — Shadow → Canary → Full.** No new policy acts on live agents until it has *earned*
  promotion on measured calibration evidence.
- **P5 — Slow/Hot split.** Heavy estimation runs once per tick (slow loop) and bakes a
  snapshot; the hot path reads it in O(1) per decision.

### 1.4 Notation

Agent $a$: health $\mathrm{HP}_a\in[0,100]$; per-domain skill posterior $(\mu_a,\sigma_a)$;
failure rate $\lambda_a$; latency distribution. Task $t$: difficulty $d_t$, value $v_t$, risk
legs. Context-penalty weights are $w_r,w_t$; the CVaR confidence level is $\alpha$; Gamma prior
hyperparameters are $(a_0,b_0)$. Tuned thresholds appear as symbols ($\theta_{kill}$, $\mu^\*$,
$\sigma^\*$, $c_3>c_2>c_1$, …); their fitted values are implementation-specific.

---

## 2. The V1–V6 Stack

### V1 — Survival (health reservoir / guillotine)

Each agent holds a health reservoir; outcomes apply signed deltas; sustained failure drains it
to a kill line — a circuit breaker.
$$\mathrm{HP}_a \leftarrow \mathrm{clip}\big(\mathrm{HP}_a + \Delta(e),\,0,\,100\big),\qquad
\mathrm{HP}_a \le \theta_{kill}\ \Rightarrow\ \text{remove}.$$
The event table $\Delta(e)$ is **graduated by blast radius**: routine successes give small
positive deltas; catastrophic failures (e.g. a production incident) give large negative ones,
calibrated so that *two* catastrophic events end an agent while routine noise does not. *(The
exact ladder is implementation-specific.)*

**Improves:** bounds the blast radius of a degrading agent; graduation avoids killing on noise.

### V2 — Credit (Bayesian skill posterior + the critical-action gate)

Each agent carries a posterior over skill: mean $\mu$ (how good) and spread $\sigma$ (how sure).
Skill moves toward each outcome with a gain that shrinks with confidence:
$$\mu \leftarrow \mu + K(o-\hat o),\quad K=\frac{\sigma^2}{\sigma^2+\tau^2},\quad
\sigma^2\leftarrow(1-K)\sigma^2.$$
Success probability on difficulty $d$ is a logistic
$p_{succ}(\mu,\sigma,d)=\big(1+e^{-s(\mu-d)/\max(\sigma,\sigma_0)}\big)^{-1}$, clamped to a
working range. *(The reference implementation uses an Elo-flavored update; the principled form
is a Gaussian-density filter — Glicko-2 / TrueSkill [Glickman 2012; Herbrich et al. 2007] —
which would make $\sigma$ a calibrated credible interval; see §6/§7.)*

**Critical-action gate (P3):** eligible for money/security work iff
$$\mathrm{HP}>\mathrm{HP}^\*\ \wedge\ \mu>\mu^\*\ \wedge\ \sigma<\sigma^\*\ \wedge\ (\text{a measured track record}).$$
Requiring **both** high skill **and** low uncertainty, on a *measured* row, blocks the
lucky-but-unproven agent from capital.

**Improves:** turns "trust" into a measured confidence statement.

### V3 — Hazard (failure-rate prior + acceleration)

Two predictive signals. **Base rate:** failures arrive as a Poisson process with per-agent rate
$\lambda_a$ (Gamma-Poisson conjugate; §3.1); base hazard $h=1-e^{-\lambda_a}$ replaces a flat
constant, so an agent failing $k\times$ more often than assumed shows a correspondingly higher
per-tick hazard. **Acceleration:** the recent skill trajectory's curvature $\ddot\mu$ signals
whether a decline is *speeding up*. The hot-path read is a simple second difference; a
significance-gated local-quadratic refinement (an OLS curvature with a t-test on its standard
error, so noise does not raise an alarm) runs in the slow loop. Sign convention:
$$\ddot\mu<0 \Rightarrow \text{decline accelerating (raise hazard, pre-emptive)},\qquad
\ddot\mu>0 \Rightarrow \text{recovery accelerating (never penalize)}.$$
Illustratively, a trajectory $90\to88\to82$ has $\ddot\mu=-4$; $90\to80\to60$ has $-10$;
$60\to70\to85$ has $+5$; a flat series has $0$ (no alarm).

**Improves:** detection *latency* — curvature can flag an accelerating collapse before the
linear hazard reaches the kill line.

### V4 — Warm-Propulsion

Symmetric scoring ignores momentum. V4 uses the same $\ddot\mu$: upward acceleration earns
extra allocation (the positive mirror of V3), while a cold, swollen working context is charged
a cost (the context term in V5). Capital concentrates on *rising* agents earlier than a
level-only policy would.

### V5 — Risk-Budgeted Auction (the allocation engine)

Every dispatch is an auction; each candidate's bid is its expected utility, gated by
eligibility:
$$\boxed{\,U(a,t)=\mathbb{1}[\text{eligible}(a,t)]\cdot\Big(p_{succ}\,v_t \;-\; c_{model}(a)\;-\;\rho_{\text{CVaR}}(t)\;-\;w_t\,\text{ctx}(a,t)\Big)\,}$$
where $\mathbb{1}[\text{eligible}]=0$ hard-blocks under the critical-action gate, a disk/resource
sentinel, or a live-switch guard. $v_t$ rises with starvation/unblocking; $c_{model}$ is the
actor's cost tier; $\rho_{\text{CVaR}}$ is the coherent tail penalty (§3.2);
$\text{ctx}=w_r\,\text{requeue}+w_t\,\text{traceback\_len}$ prices the **re-read / token-waste
cost**, so cold re-dispatch is dispreferred and warm context preferred.

**Improves:** allocation that is simultaneously cost-, risk-, and starvation-aware; the single
knob through which fail-closed safety and token economy are enforced.

### V6 — Calibration & Promotion (shadow → canary → full)

No new policy acts on live agents until it earns it. Promotion is a gate on measured
calibration, with an **immutable safety invariant: a probabilistic model can never earn
deterministic-kill authority** (severity-matched authority). Entry to canary requires a minimum
sample, sufficient agreement, a calibration error below a threshold (predicted vs realized
failure frequency), and stability; a canary stage carries a **pre-registered exit criterion**
and an explicit rollback trigger; critical-action models remain operator-gated regardless of
metrics.

**Improves:** turns "we changed the policy and hoped" into a pre-registered, evidence-gated,
reversible rollout — the model-risk discipline regulated deployments require.

---

## 3. The Risk & Estimation Core

### 3.1 Poisson–Gamma conjugacy

Failure counts are Poisson with rate $\lambda$; the conjugate prior is **Gamma**, so the same
Gamma family that models latency (§3.3) is also the prior that makes the failure-rate update
closed-form: $\lambda\mid\text{data}\sim\mathrm{Gamma}(a_0+\sum x_i,\,b_0+n)$. Deterministic, no
sampling. When a dispersion test rejects Poisson (clustered failures), the principled fallback
is Negative-Binomial.

### 3.2 Coherent tail risk (CVaR)

The tail penalty is Expected Shortfall via the Rockafellar–Uryasev estimator [2000], not a naive
average-above-VaR:
$$\mathrm{CVaR}_\alpha(L)=\mathrm{VaR}_\alpha+\frac{1}{1-\alpha}\,\mathbb{E}\big[(L-\mathrm{VaR}_\alpha)^+\big],$$
sub-additive (coherent), with a Cornish–Fisher parametric fallback [Cornish & Fisher 1938]
floored at VaR for small samples. Empirically, the failure-loss tail can run an **order of
magnitude** beyond the VaR point estimate — which is precisely why the critical-action gate is
fail-closed. *(The specific fitted tail of any deployment is implementation data.)* The live
auction prices the tail from the materialized historical CVaR with a capped Cornish–Fisher
uplift; the full hardened estimator lives in the budget layer.

### 3.3 Gamma latency

Task latency is non-negative and right-skewed → $\mathrm{Gamma}(k,\theta)$. The mean $k\theta$
prices worker-slot occupancy (cost); upper quantiles $q_{95},q_{99}$ price SLA/starvation tail.
Both feed V5.

### 3.4 Stable control

The resource shadow-price (scarcity multiplier) is driven by a controller that exposes a
Jury/Schur-verified stable-gain region for the modeled first-order plant [Jury 1964]; operating
inside that region guarantees no oscillation, with back-calculation anti-windup.

---

## 4. Empirical Results

| Metric | Result | Basis |
|---|---|---|
| Success lift, **synthetic** | **≈ +26%** (95% CI excludes 0, 40 seeds) | reproducible from seeds |
| Retry reduction, synthetic | **≈ −38%** | reproducible from seeds |
| Expected-utility gain, synthetic | **≈ +65%** | reproducible from seeds |
| Success lift, **real replay** | **≈ +17%** | one operator's 689-task, 32-agent history |
| Routing cost vs LLM manager | **10⁴–10⁶× cheaper** (est.) | first-principles; deterministic side is a CPU-cost estimate |

**Honest boundaries.** (i) **The plane is validated in *shadow/replay*; production routing is
not yet the auction** — live activation is a pending promotion decision. (ii) The synthetic
results are reproducible from seeds and carry no proprietary data; lead with them. (iii) The
real-replay lift uses *current* agent skill against *historical* tasks (a look-ahead confound on
skill); an at-assignment snapshot closes it (§6). (iv) Tail ratios are sample-dependent: a
dedicated trouble-set shows a large ratio while a thin same-window tail can be near 1× — neither
alone is decisive.

---

## 5. Portability — Deploying to Other Domains

The plane governs **any** fleet whose actors have (i) measurable outcomes, (ii) resource cost,
(iii) a track record, (iv) risky actions, (v) a need for safe rollout. To port:

1. Define the **event table** (V1 $\Delta$) for your domain's successes/failures.
2. Define the **outcome signal** (V2 $o$) and the difficulty/value features per task.
3. Define the **risk legs** (V5 $\rho_{\text{CVaR}}$) — what "critical" means in your domain.
4. Pick the **actors** — model tiers, microservices, trading strategies, even human teams.
5. The survival/credit/hazard/propulsion/auction/promotion math is **unchanged**.

*Illustrative port:* treating cost-differentiated model tiers as the actors and tier cost as the
auction's cost term, the same V5 auction + V2 gate routes most low-difficulty work to cheaper
tiers while forcing critical/hard work to the top tier — a substantial cost reduction whose
magnitude depends on the task mix.

---

## 6. Roadmap

*(Efficiency figures below are complexity-derived estimates, not yet measured.)* Memoize the
per-task tail term (hot path est. 3–50×); streaming sufficient statistics for the rate/latency
fits (slow loop est. −40–65%); buffered telemetry writes (est. −60–80% DB round-trips). Wire
the calibrated failure-rate prior into the live hazard; add measurement (oracle regret vs an
optimal-in-hindsight assignment; pre-registered canary exit criteria; calibration-drift alerts;
per-dispatch decision provenance; false-removal / false-block rates). Principled upgrades
(shadow-first): **Thompson Sampling** over UCB1 [Chapelle & Li 2011] (lower finite-horizon
regret); a true Bayesian credit filter [Glickman 2012; Herbrich et al. 2007]; a **Whittle index**
[Whittle 1988] unifying the survival/recycle/propulsion thresholds into one near-optimal index.
*(Certain second-order / correlated-tail extensions are retained in the private implementation.)*

---

## 7. What Is Already Strong (honest credit)

CVaR uses the coherent Rockafellar–Uryasev estimator (not naive ES). The slow-loop accelerator
uses a significance-gated local-quadratic fit (the hot-path read is a plain second difference).
The shadow-price controller exposes a Jury/Schur-verified stable-gain region. The promotion gate
forbids a probabilistic model from earning deterministic-kill authority — strong Goodhart/safety
hygiene. The exploration sandbox carries the Auer et al. regret bound and KL-UCB.

---

## 8. Limitations

- **Shadow, not active.** As described, the plane scores, compares, and logs but does not yet
  *act* on live dispatch; the lifts are proven on replay.
- **Heuristic credit spread.** The $\sigma$ in the critical-action gate is a heuristic spread,
  not yet a calibrated posterior interval (Glicko-2 upgrade, §6).
- **Uncalibrated constants.** Several weights are explicitly shadow/uncalibrated and must clear
  the V6 gate before going canary→full.
- **Benchmark confound.** See §4.

---

## 9. Closing

This is a deterministic, auditable answer to a problem the field commonly solves with an
expensive, non-deterministic LLM-in-the-loop. Its value is threefold: **cost** (orders of
magnitude cheaper routing), **performance** (measured success and retry improvements), and —
most importantly for regulated deployment — **trust**: every decision is reproducible and
explainable, and an unproven actor can *provably never* touch a critical action. The same math
governs any fleet; the doctrine is portable.

---

## References

- Auer, Cesa-Bianchi & Fischer (2002). *Finite-time Analysis of the Multiarmed Bandit Problem.* Machine Learning 47:235–256.
- Chapelle & Li (2011). *An Empirical Evaluation of Thompson Sampling.* NeurIPS.
- Cornish & Fisher (1938). *Moments and Cumulants in the Specification of Distributions.* Rev. Int. Stat. Inst.
- Garivier & Cappé (2011). *The KL-UCB Algorithm for Bounded Stochastic Bandits.* COLT.
- Glickman (2012). *Example of the Glicko-2 System.*
- Herbrich, Minka & Graepel (2007). *TrueSkill™: A Bayesian Skill Rating System.* NeurIPS.
- Jury (1964). *Theory and Application of the z-Transform Method.*
- Rockafellar & Uryasev (2000). *Optimization of Conditional Value-at-Risk.* J. Risk 2(3):21–41.
- Whittle (1988). *Restless Bandits: Activity Allocation in a Changing World.* J. Applied Probability 25A:287–298.

*Numbers herein are first-principles/complexity-derived, drawn from reproducible synthetic runs, or labeled estimates. Shadow-only and uncalibrated items are flagged as such.*

---

## Appendix A — Mathematics of the Roadmap Improvements

*(All gains below are complexity-derived or cited-empirical; none are yet measured in production.)*

**A.1 Hot-path memoization (efficiency).** The per-task tail penalty $\rho_{\text{CVaR}}$ is a
pure function of a small discrete risk vector $r\in\mathcal{R}^m$. Naïvely it costs a sort plus a
constant number of linear passes, $O(n\log n + cn)$ per evaluation; memoization makes it $O(1)$
after warm-up, since the domain has at most $|\mathcal{R}|^m$ distinct keys (hit rate $\to 1$).
For an auction scanning $A$ agents over $T$ tasks, computing the *task* features once instead of
per agent–task pair reduces $A\cdot T$ tail evaluations to $T$.

**A.2 Streaming sufficient statistics (efficiency).** The Poisson MLE, dispersion index, and
goodness-of-fit are all closed forms of $(N,S_1{=}\sum x,S_2{=}\sum x^2)$:
$$\hat\lambda=\tfrac{S_1}{N},\quad
\widehat{\text{disp}}=\frac{S_2/N-(S_1/N)^2}{S_1/N}.$$
Carrying these running sums turns a per-tick re-fit from $O(n)$ into $O(1)$ (an
$\tfrac{n-1}{n}$ reduction). The Gamma latency fit is analogous in $(N,\sum x,\sum\ln x)$.

**A.3 Calibrated hazard (decision quality).** Replacing a flat base hazard $h_0=1-e^{-\lambda_0}$
with the empirical-Bayes per-agent rate gives $h=1-e^{-\lambda}$. For an agent failing $c\times$
more often than assumed, the detected per-tick hazard rises by
$$\frac{1-e^{-c\lambda_0}}{1-e^{-\lambda_0}}\ \xrightarrow{\ \lambda_0\to0\ }\ c,$$
i.e. it scales with the true-to-assumed rate ratio. When a dispersion test rejects Poisson, the
Gamma–Poisson posterior is replaced by a Negative-Binomial (the conjugate over-dispersed count
model).

**A.4 Thompson Sampling over UCB1 (exploration).** UCB1 has regret
$R_n=O\!\big(\sum_i \tfrac{\ln n}{\Delta_i}\big)$ [Auer et al. 2002]; Thompson Sampling attains
the Lai–Robbins lower bound $\sum_i \tfrac{\Delta_i}{\mathrm{KL}(\theta_i,\theta^\*)}\ln n$
asymptotically and empirically lowers finite-horizon cumulative regret by $\approx 20\!-\!40\%$
[Chapelle & Li 2011] — fewer dispatches wasted on unproven agents.

**A.5 Whittle index for keep/recycle/kill (principled retirement).** The retire-vs-keep decision
under a decaying, resettable health state is a *restless bandit*; the near-optimal policy is the
**Whittle index** $W(s)$ — the subsidy that makes passivity and activity indifferent in state
$s$ — retiring an agent iff $W(s)$ falls below the subsidy for passivity. This replaces several
hand-tuned thresholds with one decision-theoretic quantity [Whittle 1988].

**A.6 Bayesian credit (calibrated confidence).** Replacing the Elo-flavored update with a
Gaussian-density filter (Glicko-2 / TrueSkill) yields a true posterior $(\mu,\phi)$ with
calibrated variance, so the gate condition $\sigma<\sigma^\*$ becomes a genuine credible-interval
statement rather than a heuristic spread, and uncertainty inflates correctly under staleness
[Glickman 2012; Herbrich et al. 2007].

*Certain correlated-tail / portfolio-risk extensions to the live auction are retained in the
private implementation.*
