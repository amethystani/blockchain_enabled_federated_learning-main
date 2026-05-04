# FORMALISM_AGENT.md
# Agent Role: Distributed Systems Formalism Auditor

> **Purpose:** Ensure all core properties (Self-Stabilization, Byzantine Robustness) are formally defined, scoped, and argued using distributed systems methodology.
> **Criticality:** HIGH. Prevents rejection for "lack of rigor" at formal venues like SSS.
> **Working directory:** `paper/`
> **Known state:** self_stabilization.tex EXISTS with Theorem + Lemma + proof sketches.
> Focus this agent on VERIFYING INTERNAL CONSISTENCY, not recreating structure.

---

## 0. Pre-Audit: Read These Files First

```bash
# Read the existing formal content — do this before any audit
cat paper/self_stabilization.tex
cat paper/system_model.tex
cat paper/theory.tex
cat paper/convergence.tex
```

Key questions for the existing content:
1. Does Lemma (Closure via Blockchain Immutability) in self_stabilization.tex correctly
   depend on the assumptions stated in system_model.tex (Assumptions 1–5)?
2. Does the proof sketch in Theorem (Self-Stabilization) correctly cite all lemmas it uses?
3. Does the stabilization time T* = O(σf/ε² + f²/ε) appear consistently in both the
   introduction.tex contributions and self_stabilization.tex theorem?
4. Is the σ²f² < 0.25 phase transition condition referenced consistently across all files?

---

## 1. Property Formalization Audit

For each core claim, verify the existence of a formal definition:

### A. Self-Stabilization
- **Definition Check:** Is there a definition of "Legitimate Configuration" ($L$)?
- **Convergence Check:** Is there a formal statement that starting from *any* configuration $C$, the system reaches $L$ within $T^*$ rounds?
- **Closure Check:** Is there a formal statement that once the system is in $L$, it remains in $L$ (unless a new transient fault occurs)?
- **Assumption Scope:** Are the assumptions on the initial state (arbitrary) and subsequent faults (Byzantine bound $f$) explicitly stated?

### B. Byzantine Robustness
- **Fault Model:** Is the adversary formally defined (Adaptive? Static? Colluding?)?
- **Safety Invariant:** Is there a formal bound on the maximum deviation of the aggregate from the honest mean (e.g., $||\hat{g} - \bar{g}|| \leq \varepsilon$)?
- **Liveness:** Is there a guarantee that the protocol continues to make progress (updates are committed) despite $f$ faulty nodes?

---

## 2. Terminology De-ML-ification

Review the use of these terms and ensure they are grounded in DS theory:

| Term | ML-style usage (Avoid) | DS-style usage (Required) |
|---|---|---|
| **Robust** | "Our defense is robust." | "The protocol satisfies the $(f, \varepsilon)$-Byzantine resilience property." |
| **Self-Stabilizing** | "The model self-stabilizes." | "The distributed protocol recovers to a legitimate configuration." |
| **Spectral Filter** | "We use PCA to find bad nodes." | "The spectral filter partitions the process set based on eigenvalue structure." |
| **Checkpoint** | "We save the model." | "A distributed commit of the configuration state $w^t$ to an immutable ledger." |

---

## 3. Theorem & Proof Structure

1. **Assumptions:** Are they numbered (A1, A2, ...) and referenced in every theorem?
2. **Phase Transition:** Is the $\sigma^2 f^2 < 0.25$ threshold presented as an information-theoretic limit?
3. **Impossibility:** Do you reference why the problem is impossible if your assumptions are violated?

---

## Output: FORMALISM_REPORT.md

Produce a report listing:
- [ ] List of "Marketing" terms that need to be replaced with "Formal" terms.
- [ ] Missing formal definitions for core properties.
- [ ] Gaps in the stabilization/recovery argument.
- [ ] Suggestions for making the system model section more "SSS-ready."

*End of FORMALISM_AGENT.md*
