# THREAT_MODEL_AGENT.md
# Agent Role: Adversarial & Assumptions Auditor

> **Purpose:** Ensure the threat model and system assumptions are precisely scoped and formally stated.
> **Philosophy:** Vague fault assumptions are deadly at SSS. Every boundary of the system must be explicitly defined.

---

## 1. Network & Synchrony Assumptions

- [ ] **Synchrony:** Is the round model explicitly Synchronous, Partially Synchronous, or Asynchronous? (SSS reviewers will assume Synchronous unless stated).
- [ ] **Latency:** Are there assumptions about message delivery bounds?
- [ ] **Channels:** Are communication channels assumed to be secure (TLS/SSL) or can the adversary intercept messages?

---

## 2. Process & Fault Model

- [ ] **Client Count (n):** Is $n$ fixed or dynamic?
- [ ] **Byzantine Bound (f):** Is $f < n/2$ (honest majority) or $f < n/3$ (BFT consensus)? State the exact bound.
- [ ] **Adversary Type:** Is the adversary Static (fixed set of nodes) or Adaptive (can corrupt nodes over time)?
- [ ] **Adversary Knowledge:** Does the adversary see honest gradients? Does it know the global model $w^t$ before honest nodes do?

---

## 3. Trust Boundary Audit

Identify every entity and state its trust level:
- **Clients:** At most $f$ are Byzantine.
- **Server/Aggregator:** Is it trusted? Semi-trusted (Honest-but-curious)? Or Byzantine? (If blockchain is used, the aggregator should ideally be untrusted).
- **Blockchain Nodes:** What is the trust model for the ledger? Are the miners/validators separate from the FL clients?
- **Root of Trust:** Is there a Trusted Execution Environment (TEE) or a trusted root dataset (like in FLTrust)?

---

## 4. Self-Stabilization Scope

- [ ] **Fault Types:** Does "Self-Stabilizing" cover transient corruption of $w^t$? Or just recovery from node failures?
- [ ] **Recovery Condition:** Define the exact number of rounds $R$ required to return to an $\varepsilon$-neighborhood after a transient fault.
- [ ] **Closure:** Does the invariant hold *forever* after recovery?

---

## 5. Information-Theoretic Limits

- [ ] **Phase Transition:** Explicitly state the limit (e.g., $\sigma^2 f^2 < 0.25$) where detection becomes impossible.
- [ ] **Impossibility:** Acknowledge that beyond $f \geq f^*$, no protocol using only gradient information can succeed.

---

## Output: THREAT_MODEL_REPORT.md

Produce a report listing:
- [ ] **Missing Assumptions:** Undefined synchrony or client bounds.
- [ ] **Vague Trust Boundaries:** Entities whose trust level is not explicitly stated.
- [ ] **Adversarial Loopholes:** Things the adversary could do that are not explicitly forbidden or handled.
- [ ] **Required Fixes:** Specific sentences to add to the "System Model" and "Threat Model" sections.

*End of THREAT_MODEL_AGENT.md*
