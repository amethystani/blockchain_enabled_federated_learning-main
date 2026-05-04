# EXECUTION_ORCHESTRATOR.md
# Master Agent: SSS 2026 Paper Improvement Pipeline

> **START HERE.** This is the entry point. Run this agent first — it sequences all others.
> **Working directory:** `paper/` subfolder of the repository root.
> **Deadline:** May 15, 2026 AoE (Round 2). Today: May 3, 2026. **12 days.**

---

## KNOWN PAPER STATE (as of May 2026 audit)

The following sections ALREADY EXIST and are STRUCTURALLY SOUND — do NOT recreate them:
- `system_model.tex` — formal system model with numbered Assumptions 1–5, threat model, round structure
- `self_stabilization.tex` — formal self-stabilization section with Theorem, Lemma, proof sketches, Dijkstra/Dolev cites
- `algorithm.tex` — pseudocode with Frequent Directions sketching and spectral filter
- `introduction.tex` — good distributed-systems framing, Dijkstra cited, 5 formal contributions
- `main.tex` — LNCS format, double-blind, correct keywords

The following sections are WEAK or MISSING — these are the priority targets:
- `related_work.tex` — **ONLY 3 LINES. This is the biggest gap. Must be written.**
- `conclusion.tex` — **Only 5 lines. Must be expanded.**
- Experiments: **No recovery-over-rounds plot/discussion.** SSS requires this.
- Experiments: **Suspicious baseline results** (FLTrust/FLAME/CRFL/ByzShield all exactly 63.4%)
- `theory.tex` / `convergence.tex` — Check whether proof of self-stabilization bound is complete.
- Abstract: Strong but may still contain hype language — scan and fix.

---

## EXECUTION ORDER

Run agents in this exact sequence. Do not skip steps.

### STEP 1 — Read REVIEW_REPORT.md (if it exists)
```bash
cat paper/REVIEW_REPORT.md 2>/dev/null || echo "No review report yet — will generate in Step 2"
```

### STEP 2 — Run PAPER_REVIEWER_AGENT
```
Run: paper/AGENTS/PAPER_REVIEWER_AGENT.md
Output: paper/REVIEW_REPORT.md
Time estimate: ~15 min
```
Read every .tex file in `paper/`, apply the SSS reviewer rubric, produce REVIEW_REPORT.md.
**Do not proceed to Step 3 until REVIEW_REPORT.md exists.**

### STEP 3 — Run CLAIMS_AUDIT_AGENT
```
Run: paper/AGENTS/CLAIMS_AUDIT_AGENT.md
Output: paper/CLAIMS_AUDIT_REPORT.md
Time estimate: ~10 min
```
Critical: identify the suspicious 63.4% baseline results and all overclaimed theorems.

### STEP 4 — Run SSS_REVISION_AGENT (targeted mode)
```
Run: paper/AGENTS/SSS_REVISION_AGENT.md
Input: paper/REVIEW_REPORT.md + paper/CLAIMS_AUDIT_REPORT.md
Priority targets in order:
  1. related_work.tex (rewrite from 3 lines to full section)
  2. conclusion.tex (expand from 5 lines to full section)
  3. experiments.tex (add recovery-over-rounds discussion, fix baseline framing)
  4. abstract in main.tex (strip remaining hype language)
  5. Probe theory.tex/convergence.tex for incomplete proofs → add TODO markers
```

### STEP 5 — Run RELATED_WORK_AGENT
```
Run: paper/AGENTS/RELATED_WORK_AGENT.md
Focus: Write the missing related_work.tex with 4 citation pillars
Must include: SSS proceedings papers, Dijkstra/Dolev, Byzantine FL baselines, RMT/spectral methods
```

### STEP 6 — Run EXPERIMENTAL_VALIDITY_AGENT
```
Run: paper/AGENTS/EXPERIMENTAL_VALIDITY_AGENT.md
Focus: Investigate the identical 63.4% baseline results and flag for author attention
Output: written into REVIEW_REPORT.md or a separate EXPERIMENT_AUDIT.md
```

### STEP 7 — Run FORMALISM_AGENT
```
Run: paper/AGENTS/FORMALISM_AGENT.md
Focus: Verify self_stabilization.tex proofs are internally consistent
Check: theorem statements match lemmas, notation matches system_model.tex
```

### STEP 8 — Run NOTATION_INTEGRITY_AGENT
```
Run: paper/AGENTS/NOTATION_INTEGRITY_AGENT.md
Focus: Cross-file variable consistency across all .tex files
Critical check: \sigma, f, n, T^*, \varepsilon used consistently everywhere
```

### STEP 9 — Run THREAT_MODEL_AGENT
```
Run: paper/AGENTS/THREAT_MODEL_AGENT.md
Focus: Verify Assumptions 1–5 in system_model.tex are complete and sufficient
Check: adaptive adversary handling, f < n/2 bound justification
```

### STEP 10 — Run ACADEMIC_TONE_AGENT
```
Run: paper/AGENTS/ACADEMIC_TONE_AGENT.md
Focus: Final prose scrub across ALL .tex files
Priority files: introduction.tex, experiments.tex (most likely to have hype language)
```

### STEP 11 — Run SUBMISSION_CHECKLIST_AGENT
```
Run: paper/AGENTS/SUBMISSION_CHECKLIST_AGENT.md
This is the FINAL step — run only after all revisions complete
Output: Go/No-Go decision for May 15 submission
```

---

## TOP 5 HIGHEST IMPACT CHANGES (do these if time is short)

If only 2 hours are available before submission, do these in order:

**1. Write related_work.tex** (30 min)
This is the single biggest gap. SSS reviewers expect distributed-systems citations.
A paper with 3-line related work SIGNALS the authors don't know the venue.
Template in: `paper/AGENTS/RELATED_WORK_AGENT.md`

**2. Investigate and address the 63.4% baseline coincidence** (20 min)
FLTrust, FLAME, CRFL, ByzShield all showing EXACTLY 63.4% looks like a copy-paste error
or overfitted results. Reviewers WILL flag this. Either:
  (a) Verify these are real results from actual experiments, OR
  (b) Add a footnote explaining why they cluster, OR  
  (c) Run the experiments properly and report actual values
Never fabricate results — but also never leave a result that looks like a bug.

**3. Add recovery-over-rounds discussion to experiments.tex** (20 min)
SSS Track D reviewers care about STABILIZATION DYNAMICS, not just final accuracy.
Add: "Figure X shows recovery time R — rounds to return to ε-stable trajectory
after attack onset at round T_attack." Even if the figure already exists, make
sure the CAPTION explains the recovery time explicitly.

**4. Expand conclusion.tex** (15 min)
The current 5-line conclusion ends the paper weakly. Use the template in
SSS_REVISION_AGENT.md Revision 8. Must include: formal property name,
assumptions, and 2+ open problems in distributed-systems terms.

**5. Strip hype from abstract** (10 min)
Check abstract in main.tex for:
- "state-of-the-art" without citation → delete or cite
- "first to provide X" — verify this claim is defensible
- "100% Byzantine detection rate" — this is a strong claim; ensure it's scoped

---

## WORKING DIRECTORY CONVENTIONS

All .tex files are in `paper/` subfolder. When running bash commands, either:
```bash
cd /path/to/repo/paper && <command>
# OR
<command> paper/*.tex
```

Do NOT run from repo root with bare filenames — they won't exist there.

The main LaTeX root file is: `paper/main.tex`
Compile with: `cd paper && latexmk -pdf main.tex`

---

## AGENT OUTPUT FILES

| Agent | Output file |
|---|---|
| PAPER_REVIEWER_AGENT | `paper/REVIEW_REPORT.md` |
| CLAIMS_AUDIT_AGENT | `paper/CLAIMS_AUDIT_REPORT.md` |
| EXPERIMENTAL_VALIDITY_AGENT | `paper/EXPERIMENT_AUDIT.md` |

These files are inputs to SSS_REVISION_AGENT — read them before making revisions.

---

## WHAT NOT TO TOUCH

- `self_stabilization.tex` — formal content is good; only tweak notation if NOTATION_INTEGRITY_AGENT flags issues
- `system_model.tex` — structure is SSS-appropriate; only add missing content flagged by THREAT_MODEL_AGENT
- `algorithm.tex` — pseudocode exists and is correct; only improve if clarity issues flagged
- `introduction.tex` — distributed-systems framing is already present; only strip remaining hype
- The LNCS format parameters in `main.tex` — do not change margins, fonts, or document class options

---

*End of EXECUTION_ORCHESTRATOR.md — proceed to PAPER_REVIEWER_AGENT.md*
