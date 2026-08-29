# Theory samples

Three results from the doctrine this control plane is built on. They are
general — none of them is specific to the V1–V5 stack, and none is part of
the private fitted layer. They are here because they are the load-bearing
ideas: everything else in the design follows from taking these seriously.

---

## 1 · A claim is a tuple, not a sentence

Every consequential statement is represented as

$$
\mathcal{C}=(S,\;D,\;A,\;E,\;F,\;U)
$$

| | |
|---|---|
| $S$ | the exact statement |
| $D$ | the domain over which it is asserted |
| $A$ | the **complete** assumption set |
| $E$ | the evidence and its provenance |
| $F$ | a falsifier — the condition under which $S$ would be wrong |
| $U$ | the maximum authority the claim may support |

**A claim is incomplete if any component is absent.** That rule does most of
the work in practice:

- an equation without units and assumptions is not a product theorem;
- a benchmark without a dataset identity and a comparator is not a measured
  result;
- source code alone is not evidence that anything runs.

The component people omit most often is $F$. A statement with no falsifier
cannot be wrong, which sounds like strength and is the opposite: it means no
observation could ever update it. If you cannot say what would falsify a
claim, you do not yet have one.

The component that gets *inflated* most often is $U$. A result that holds on
a benchmark supports authority over that benchmark. Extending it to
production is a separate claim, with its own $A$, $E$ and $F$.

---

## 2 · Authority is conjunctive

For consequential operation, authority is not a score, and it is not inferred
from the strongest available evidence label.

$$
\mathrm{Authority}=
\mathrm{MechanismProperty}\land
\mathrm{MeasuredFitness}\land
\mathrm{AuthenticatedMandate}\land
\mathrm{SeparationOfDuty}\land
\mathrm{OperatorRelease}\land
\mathrm{RuntimePrerequisites}
$$

**If any term is false or unknown, authority is false.** Unknown counts as
false — that is the fail-closed reading, and it is deliberate.

The practical force of this is in what *cannot* substitute for what:

> A theorem can support the first term. A controlled evaluation can support
> the second. **Neither can mint the remaining terms.**

This is why a very good model does not thereby acquire permission. Proving a
mechanism has a property, and measuring that it performs, together establish
two of six conjuncts. Mandate, separation of duty, operator release and
runtime prerequisites are not earned by being correct — they are granted, and
they are granted by someone other than the thing being granted them.

Most "the AI decided X" incidents are a conjunction quietly evaluated as a
maximum.

---

## 3 · Single-writer effect under fencing

*The one that is a real theorem with a real proof.*

**Setup.** Every successful claim on a work item receives a strictly
increasing fencing token $f\in\mathbb{N}$. The authoritative result store
accepts a transition only when the token presented equals the currently
active token for that item.

**Theorem.** A worker holding an expired token cannot commit a result after a
newer claim has been granted.

**Proof.** A later claim carries $f' > f$ and becomes the active token. The
equality precondition rejects any subsequent transition presenting $f$. $\;\blacksquare$

**What the proof does not need.** The two workers' clocks never have to
agree, at commit time or at any other time. The argument requires only an
atomic token comparison at one authoritative store. That is why it survives
clock skew, GC pauses, suspended VMs and closed laptop lids — the failure
modes that defeat every lease scheme built on timeouts alone.

**Failure mode.** A lease *ID* without a monotone token is insufficient. A
worker that pauses can wake after expiry and overwrite newer work unless
**every** side effect re-checks the token at the moment of its write. A check
performed at claim time is not a check; the window it leaves open is exactly
the pause.

---

## Why these three

They compose into the discipline the rest of the system inherits.

The claim tuple says what a statement must carry before it counts. The
conjunctive authority predicate says what a statement — however well
supported — still cannot buy. The fencing theorem is what a claim looks like
when it *is* fully carried: a stated result, an explicit precondition, a
proof, and a named failure mode that tells you precisely when it stops
holding.

A system that applies the first two to itself ends up with the third
everywhere, or with honest gaps where the third does not yet exist. Both are
acceptable. What is not acceptable is the shape in between — a confident
claim with no falsifier, holding authority nobody granted it.

---

*These are drawn from the open theory core, which is shared unchanged across
every edition of the underlying doctrine. The fitted calibration layer — the
part that learns constants from real outcome history — is a separate
private/commercial layer and is not represented here.*
