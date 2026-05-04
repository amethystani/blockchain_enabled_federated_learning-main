# CLAIMS_AUDIT_AGENT.md
# Agent Role: Technical Claims Auditor

> **Purpose:** Audit every technical claim in the paper for support level and risk.
> Output: `CLAIMS_AUDIT_REPORT.md` with a complete claim classification table.
> **Run after PAPER_REVIEWER_AGENT, before SSS_REVISION_AGENT.**

---

## Agent Instructions

```bash
# ALL COMMANDS USE paper/ prefix — files are NOT in repo root

# Step 1: Find all tex files in paper directory
find paper/ -name "*.tex" | sort

# Step 2: Extract high-risk claims from all paper tex files
grep -n "guarantee\|ensures\|proves\|shows\|eliminates\|prevents\|fully\|always\|never\|optimal\|state-of-the-art\|self-stabiliz\|converge\|robust\|secure\|Byzantine\|detects\|provably\|100%" paper/*.tex | grep -v "^paper/bibliography"

# Step 3: Find forbidden phrases specifically
grep -in "groundbreaking\|revolutionary\|unprecedented\|perfectly\|completely robust\|fully self-stab\|guarantees security\|eliminates Byzantine" paper/*.tex

# Step 4: Find the specific 100% detection claim
grep -n "100%" paper/*.tex

# Step 5: Find the identical baseline issue
grep -n "63.4\|63\.4" paper/experiments.tex
```

Read the ENTIRE paper. For every sentence making a technical claim, classify it.

---

## Claim Classification System

Classify each claim into ONE of six categories:

| Code | Category | Meaning |
|---|---|---|
| **T** | Theory/Proof | Supported by a complete formal proof in paper or appendix |
| **E** | Experiment | Supported by experimental results in the paper |
| **C** | Citation | Supported by cited prior work (the claim is known from literature) |
| **P** | Plausible | Reasonable and likely true but not supported in this paper |
| **O** | Overclaim | States more than what theory/experiments actually show |
| **U** | Undefined/Vague | The claim cannot be evaluated because key terms are undefined |

---

## Forbidden Phrases — Automatic Overclaim Flag

The following phrases are ALWAYS flagged as overclaims unless backed by a complete proof:

| Forbidden phrase | Why it's dangerous | Safer alternative |
|---|---|---|
| "guarantees security" | Security requires a formal definition + proof | "is designed to improve robustness under..." |
| "ensures convergence" | Convergence requires distributional assumptions + proof | "empirically converges under..." OR "converges under Assumption A" |
| "eliminates Byzantine attacks" | Impossibility results exist; no system eliminates all attacks | "bounds the influence of Byzantine updates to ε" |
| "definitely detects malicious clients" | No spectral filter has perfect detection without assumptions | "detects Byzantine updates when spectral gap exceeds τ" |
| "fully self-stabilizing" | Requires formal proof from any starting configuration | "practically stabilizing" or "provides bounded recovery under..." |
| "completely robust" | No FL system is completely robust | "robust under assumptions A1–A3" |
| "state-of-the-art" | Requires documented comparison to current SOTA | cite specific result or remove |
| "provably" | Requires a proof | remove "provably" or add the proof |
| "optimal" | Requires a lower bound | remove or qualify: "empirically efficient" |
| "our system is secure" | Security without definition is meaningless | "satisfies [Definition X] of security under [threat model]" |

---

## Claims Audit Table Template

For each claim found, add a row:

```markdown
| # | Claim (exact quote) | Section | Support Code | Risk Level | Required Fix |
|---|---|---|---|---|---|
| 1 | "our approach guarantees convergence" | Abstract | O | HIGH | Replace: "empirically converges" or add convergence proof |
| 2 | "We achieve state-of-the-art accuracy" | §5 | O | HIGH | Add comparison to specific current SOTA paper with citation |
| 3 | "The count-sketch reduces communication by O(d/k)" | §3 | C | LOW | Cite Charikar et al. 2002 |
```

---

## Special Focus Areas

### A. Self-Stabilization Claims
Find every use of "self-stabiliz*" in the paper:
```bash
grep -n "self-stabiliz" paper/*.tex
```
Note: self_stabilization.tex has a formal Definition + Theorem already.
Check that EVERY USE of "self-stabilizing" in introduction.tex/abstract
is consistent with the formal definition in self_stabilization.tex.
Informal uses that are stronger than the theorem → OVERCLAIM.

### B. Byzantine Robustness Claims
Find every use of "Byzantine*" + "robust*" or "toleran*":
```bash
grep -n "Byzantine\|robust" paper/*.tex | grep -iv "related\|prior\|existing"
```
For each:
- Is the Byzantine fraction bound stated? (f < n/2)
- Is the detection guarantee conditional on Assumption 3 (σ²f² < 0.25)?
- Does it claim to handle ALL Byzantine attacks or specific enumerated types?

### C. Convergence Claims
Find every use of "converg*":
```bash
grep -n "converg" paper/*.tex
```

### D. Spectral Detection Claims
Find claims about what the spectral filter guarantees:
```bash
grep -n "spectral\|eigenv\|detect\|filter" paper/*.tex | grep -v "%"
```
The paper claims "100% Byzantine detection rate" — check whether this is:
- Scoped by attack type and f/n value (acceptable)
- Or a bare claim of universal detection (overclaim)

### E. Communication/Efficiency Claims
Find claims about communication reduction, memory, speed:
```bash
grep -n "reduc\|efficien\|memory\|communicat\|overhead\|faster\|O(" paper/*.tex
```

### F. Comparison Claims
Find every claim comparing to baselines:
```bash
grep -n "outperform\|better\|superior\|compared\|versus\|vs\." paper/*.tex
```
For each:
- Is there a table/figure supporting this comparison?
- Are the baselines the same configuration (same dataset, same f, same architecture)?
- Is the comparison fair (no cherry-picked hyperparameters)?

---

## Output: CLAIMS_AUDIT_REPORT.md

```markdown
# CLAIMS_AUDIT_REPORT.md
# Paper: our approach
# Auditor: Claims Audit Agent
# Date: [date]

---

## Summary Statistics

Total claims audited: [N]
- Proved (T): [n]
- Experiment-supported (E): [n]
- Citation-supported (C): [n]
- Plausible but unsupported (P): [n]
- Overclaims (O): [n]
- Undefined/Vague (U): [n]

**HIGH RISK claims requiring immediate revision: [n]**

---

## Full Claims Table

| # | Claim (exact quote, with line number) | Section | Code | Risk | Required Fix |
|---|---|---|---|---|---|
[fill in every claim found]

---

## Critical Overclaims — Fix Before Submission

[List every O-coded claim with specific rewrite suggestion]

1. **Claim:** "[exact quote]" (line [N], [section])
   **Problem:** [why this is overclaimed]
   **Fix:** Replace with: "[safer alternative wording]"

---

## Undefined Terms — Define Before Submission

[List every U-coded claim]

1. **Term:** "[undefined term]" used in "[context]"
   **Required:** Add Definition [N]: "[formal definition]"

---

## Missing Proofs — Priority Order

[List every claim marked T that lacks a complete proof]

1. Lemma [N]: [statement] — needs proof
   Priority: [HIGH/MEDIUM/LOW based on centrality to paper's contribution]
   Suggested approach: [hint at proof strategy if possible]

---

## Forbidden Phrases Found

[List every occurrence of forbidden phrases]

| Phrase | Location | Replacement |
|---|---|---|

---

## Safe Alternative Phrasings Glossary

Standard rewrites to use throughout the revised paper:

| Replace | With |
|---|---|
| "guarantees security" | "is designed to improve robustness under [threat model]" |
| "ensures convergence" | "empirically converges under [conditions]" OR "converges under Assumption [N]" |
| "eliminates Byzantine attacks" | "bounds the influence of Byzantine updates to ε per round under Assumption [N]" |
| "detects all malicious clients" | "detects Byzantine updates when the spectral gap exceeds threshold τ (Lemma [N])" |
| "fully self-stabilizing" | "provides bounded recovery under [stated assumptions] (Theorem [N])" |
| "completely robust" | "robust to [specific attacks] under [fraction bound] Byzantine clients" |
| "state-of-the-art" | "competitive with [specific cited method] under [specific conditions]" |
| "provably robust" | "empirically robust" OR add the proof |
| "our system is secure" | "satisfies [security property] under [adversary model]" |
```

---

*End of CLAIMS_AUDIT_AGENT.md*
