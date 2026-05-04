# NOTATION_INTEGRITY_AGENT.md
# Agent Role: Mathematical & Cross-Reference Auditor

> **Purpose:** Audit every variable, equation, and reference for consistency across the modular LaTeX files.
> **Constraint:** Never assume a variable is "obvious." Every symbol must be defined upon first use.
> **Run AFTER SSS_REVISION_AGENT has completed major structural changes.**

---

## Agent Instructions

```bash
# ALL COMMANDS RUN FROM paper/ subdirectory or with paper/ prefix

# Step 1: List all math-mode variables across all tex files
grep -oh '\$[^$]*\$' paper/*.tex | sort | uniq -c | sort -rn | head -50

# Step 2: Check for broken references after compile
grep "Warning: Reference\|LaTeX Warning: Reference" paper/main.log 2>/dev/null

# Step 3: Check for citation gaps
grep "Warning: Citation\|LaTeX Warning: Citation" paper/main.log 2>/dev/null

# Step 4: Find TODO markers left by other agents
grep -rn "TODO" paper/*.tex

# Step 5: Compile to get fresh log
cd paper && latexmk -pdf main.tex 2>&1 | grep -E "Error|Warning" | head -30
```

---

## 1. Notation Consistency Check

Verify that core parameters are identical across all sections:

| Parameter | Meaning | Current Notation(s) | Target Notation |
|---|---|---|---|
| Number of clients | Total participants | $n$ or $N$? | **$n$** |
| Byzantine fraction | Number of faulty nodes | $f$ or $B$? | **$f$** |
| Learning rate | SGD step size | $\eta$ or $\gamma$? | **$\eta$** |
| Heterogeneity | Gradient variance | $\sigma^2$ or $v$? | **$\sigma^2$** |
| Dimension | Model parameters | $d$ or $D$? | **$d$** |
| Round | Current iteration | $t$ or $r$? | **$t$** |

**Rule:** If a variable is redefined or used inconsistently, flag it as a **MINOR BUG**.

---

## 2. Definition Audit

For every section (Introduction, System Model, Theory, etc.):
1. Identify the first use of every mathematical symbol.
2. Check if there is a prose definition (e.g., "where $\eta$ is the learning rate").
3. If a symbol appears without a definition in the same section, flag it.

**Special Focus:** The $(\sigma, f)$-threat model must be defined in both the Introduction and the System Model for clarity.

---

## 3. Cross-Reference Audit

Verify every `\ref` and `\cite`:
1. **Figure/Table references:** Does the text description of "Figure X" match what the caption of Figure X says?
2. **Theorem/Lemma references:** Ensure "Theorem 1" in the intro matches "Theorem 1" in the theory section.
3. **Bibliography:** Every entry in `bibliography.tex` must be cited at least once in the text.

---

## 4. LNCS Formatting Compliance

1. **Section Naming:** Avoid generic names like "Methodology". Use "The our approach Protocol".
2. **Figure Captions:** Must end with a period. Must be below the figure.
3. **Table Captions:** Must be above the table.
4. **Math Style:** Use `\mathbb{R}` for real numbers, not `R`. Use `\mathcal{D}` for datasets.

---

## Output: NOTATION_AUDIT_REPORT.md

Produce a report listing:
- [ ] List of inconsistent variables.
- [ ] List of undefined symbols per section.
- [ ] Broken/Misaligned figure or table references.
- [ ] LNCS formatting violations.

*End of NOTATION_INTEGRITY_AGENT.md*
