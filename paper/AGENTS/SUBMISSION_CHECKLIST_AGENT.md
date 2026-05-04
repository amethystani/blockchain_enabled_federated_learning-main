# SUBMISSION_CHECKLIST_AGENT.md
# Agent Role: Final Submission & Compliance Auditor

> **Purpose:** A "pre-flight" checklist to ensure the paper meets all SSS 2026 and Springer LNCS requirements.
> **Constraint:** Do not submit until every box is checked.

---

## 1. Venue-Specific Checklist (SSS 2026)

- [ ] **Track Alignment:** Is the paper submitted to Track D (Distributed AI/ML)?
- [ ] **Anonymization:** Are all author names, affiliations, and identifying URLs removed for double-blind review?
- [ ] **Page Limit:** Is the body (incl. figures/tables) $\leq$ 15 pages? (References can be extra).
- [ ] **Round 2 Deadline:** Is the submission ready before May 15, 2026 AoE?
- [ ] **Keywords:** Are keywords like "Self-Stabilization" and "Byzantine Fault Tolerance" present?

---

## 2. Formatting Checklist (Springer LNCS)

- [ ] **Document Class:** `\documentclass{llncs}` is used.
- [ ] **Graphics Path:** `\graphicspath{{figures/}}` is correctly set.
- [ ] **Captions:** Figure captions are *below*; Table captions are *above*.
- [ ] **References:** Citation style is `\cite{...}` with alphabetical or appearance order as per LNCS.
- [ ] **Package Conflicts:** Ensure `amsmath`, `amssymb`, and `mathtools` are loaded without error.

---

## 3. Technical Integrity Checklist

- [ ] **Spectral Diagram:** Is the architecture diagram updated to `flowdiag.jpeg`?
- [ ] **Branding:** Is "Spectral Sentinel" replaced with "our approach" or similar everywhere?
- [ ] **Math symbols:** Are all variables ($n, f, \eta, \sigma$) defined?
- [ ] **Self-Stab Proof:** Is the recovery round complexity $T^*$ stated clearly in the abstract?

---

## 4. Submission Manifest

```bash
# Run from repo root or paper/ directory

# 1. Compile main.tex
cd paper && latexmk -pdf main.tex

# 2. Check for missing citations/references
grep "Warning" paper/main.log | grep -v "Overfull\|Underfull" | head -20

# 3. Check page count (body only — references don't count toward 15-page limit)
pdfinfo paper/main.pdf | grep Pages

# 4. Check for TODO markers left by revision agents (must all be resolved before submission)
grep -rn "TODO" paper/*.tex | grep -v "%.* TODO"

# 5. Verify double-blind: no author names
grep -n "author\|affil\|email\|university\|lab\|grant" paper/main.tex | grep -v "%\|Anonymous"

# 6. Check for undefined references
grep "LaTeX Warning: Reference" paper/main.log
```

---

## 4b. SSS Content Checklist (Go/No-Go Gate)

**The paper MUST pass all of these before submission:**

- [ ] **Related work has SSS citations:** `grep -c "SSS\|Dijkstra\|Dolev\|Lamport" paper/related_work.tex` → must be > 0
- [ ] **Abstract does NOT lead with accuracy:** `head -5 abstract` should not contain "accuracy" or "achieve X%"
- [ ] **63.4% baseline coincidence resolved:** FLTrust/FLAME/CRFL/ByzShield results either explained or corrected
- [ ] **Recovery time R stated explicitly:** `grep -n "recovery time\|rounds to recover\|R =" paper/experiments.tex` → must find something
- [ ] **"100% detection" claim is scoped:** not bare claim — must have "under [conditions]"
- [ ] **conclusion.tex > 10 lines:** `wc -l paper/conclusion.tex` → must be > 10
- [ ] **related_work.tex has subsections:** `grep -c "subsection" paper/related_work.tex` → must be ≥ 3
- [ ] **No `\cite{TODO:` left in any file:** `grep -rn "cite{TODO" paper/*.tex` → must return nothing
- [ ] **Stabilization time T* appears consistently:** same formula in introduction and self_stabilization.tex
- [ ] **All figures referenced in text exist in paper/figures/:** `ls paper/figures/`

## 5. Recommended Agent Workflow

To ensure the highest quality submission, follow this specific execution order:

1.  **Targeting Guide**: Understand the venue and tracks.
2.  **Threat Model Agent**: Define precise adversarial boundaries and synchrony.
3.  **Formalism Agent**: Audit the mathematical rigor of definitions and properties.
4.  **Related Work Agent**: Position the paper against classic DS/BFT literature.
5.  **Experimental Validity Agent**: Brutally audit baselines, attacks, and metrics.
6.  **Claims Audit**: Verify that all technical claims match the evidence.
7.  **Reviewer Agent**: Perform a mock SSS peer review.
8.  **Revision Agent (Pass 1)**: Apply major structural and content fixes.
9.  **Revision Agent (Pass 2)**: **(CRITICAL)** Run a second time to polish the integration of all fixes.
10. **Notation Auditor**: Clean up math notation ($n, f, \eta$) and references.
11. **Academic Tone Agent**: Final "hype" scrubbing and AI-filler removal.
12. **Final Submission Checklist**: Last check for LNCS/anonymity compliance.

*End of SUBMISSION_CHECKLIST_AGENT.md*
