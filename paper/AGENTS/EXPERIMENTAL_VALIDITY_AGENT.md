# EXPERIMENTAL_VALIDITY_AGENT.md
# Agent Role: Brutal Experimental Auditor

> **Purpose:** Critically evaluate the experimental setup and results against SSS 2026 standards.
> **Working directory:** `paper/` subdirectory.
> **Output:** `paper/EXPERIMENT_AUDIT.md`

---

## CRITICAL RED FLAGS — INVESTIGATE THESE FIRST

### Red Flag 1: Identical Baseline Accuracies (MUST RESOLVE)

In `paper/experiments.tex`, Table 2 (Accuracy Comparison) shows:
```
FLTrust   63.4%
FLAME     63.4%
CRFL      63.4%
ByzShield 63.4%
```
**All four showing EXACTLY 63.4% is statistically implausible** for independently implemented
methods on the same task. SSS reviewers will immediately flag this as:
- A copy-paste error in the table
- Results not actually run (fabricated/estimated)
- A degenerate experimental setup where all methods collapse

**Action required:**
```bash
# Search for how these values were generated
grep -n "63.4" paper/experiments.tex
grep -rn "63.4" spectral_sentinel/
# Find the actual result files
ls -la spectral_sentinel/results/ 2>/dev/null || find . -name "*.csv" -o -name "*.json" | head -20
```

Possible explanations (investigate and document which is true):
1. All four methods converge to the same accuracy floor because the 40% Byzantine rate
   exceeds their effective threshold → **explain this in the paper**
2. The table was populated from a single run, not multiple → **run with 3+ seeds and report mean±std**
3. The values are wrong → **re-run experiments and correct**

Whatever the explanation, ADD a footnote or text explaining why these values are equal.
Without explanation, reviewers assume error.

### Red Flag 2: "100% Byzantine Detection Rate" Claim

In `paper/main.tex` abstract and/or `paper/experiments.tex`:
"100% Byzantine detection rate"

**This is an extraordinary claim.** Verify:
```bash
grep -n "100%" paper/experiments.tex paper/main.tex paper/self_stabilization.tex
```
- Is it 100% across ALL 12 attack types or only some?
- Is it 100% for ALL Byzantine fractions (10%, 20%, 30%, 40%, 49%) or only below a threshold?
- What about the adaptive spectral-aware attack? Detection rate there?

If 100% is accurate: **scope it explicitly** — "100% Byzantine detection rate across [X] attack
types with [f/n] Byzantine fraction below [threshold]"
If 100% is inaccurate: **correct the claim immediately** — a false claim is fatal.

### Red Flag 3: Missing Recovery-Over-Rounds Visualization

SSS Track D reviewers assess **stabilization behavior**, not just final accuracy.
The critical question is: after Byzantine attack onset at round T_attack, how many rounds
does the protocol need to return to within ε of the unattacked trajectory?

```bash
# Check if recovery-over-rounds figure exists
grep -n "recovery\|stabiliz\|rounds\|T_attack\|T_stab" paper/experiments.tex
ls paper/figures/ | grep -i "recovery\|stab"
```

If recovery-over-rounds figure EXISTS: verify the caption explains recovery time R explicitly.
Bad caption: "Figure X: Accuracy under sign-flip attack."
Good caption: "Figure X: Recovery dynamics under sign-flip attack (f=0.3). our approach
returns to within 2% of unattacked accuracy within R=15 rounds of attack onset (↑),
while FLTrust and Krum fail to recover within the 100-round window."

If recovery figure is MISSING: add a TODO marker in experiments.tex:
```latex
% TODO: ADD FIGURE — Recovery-over-rounds plot (CRITICAL FOR SSS)
% X-axis: training round (0–100), Y-axis: test accuracy
% Vertical line at T_attack, lines for our approach + top 3 baselines
% Caption MUST state recovery time R in rounds
```

---

## 1. Baseline & Comparison Audit

```bash
# Read experiments.tex and check baselines
grep -n "FedAvg\|Krum\|Trimmed\|FLTrust\|Bulyan\|FLAME\|SignGuard\|Geometric" paper/experiments.tex
```

Required baselines for SSS acceptance:
- [x] FedAvg (weakest baseline — sanity check)
- [x] Krum (seminal work — must be included)
- [x] Trimmed Mean / Coordinate Median
- [x] FLTrust (most recent strong baseline)
- [x] Bulyan (or Bulyan++ if newer)
- [ ] VERIFY: Are there error bars / std deviations? (No error bars = "not statistically significant")
- [ ] VERIFY: Are hyperparameters of baselines tuned fairly or used at defaults while ours is tuned?
- [ ] VERIFY: Does the paper compare against SignGuard or similar 2022–2025 methods?

---

## 2. Attack Model Audit

```bash
grep -n "attack\|Byzantine\|ALIE\|MinMax\|sign.flip\|label.flip\|Gaussian" paper/experiments.tex | head -30
```

Required attacks for SSS acceptance:
- [x] Sign flip ✓
- [x] Label flip ✓
- [x] Gaussian noise ✓
- [x] ALIE ✓
- [x] MinMax ✓
- [ ] **CRITICAL — Adaptive attack:** Adversary that knows the spectral filter and crafts
  updates to evade it. Without this, Track D reviewers will ask "is the filter robust to
  adaptive adversaries?" as their first question.
  ```latex
  % TODO: Add adaptive spectral-evasion attack results, OR add to Limitations:
  % "Adaptive adversaries that craft updates to match the Marchenko-Pastur spectral
  %  distribution remain outside our current threat model and constitute an important
  %  open problem (see Section \ref{sec:limitations})."
  ```

---

## 3. Metric & Claim Audit

Check each metric:
- [ ] **Byzantine fraction sensitivity:** Is there a plot/table showing accuracy vs. f/n = 0.1, 0.2, 0.3, 0.4, 0.49?
- [ ] **Recovery time R:** Is the number of rounds to recover explicitly measured and reported?
- [ ] **Communication overhead:** Is the O(k) claim backed by actual bytes-per-round measurement?
- [ ] **Memory overhead:** Is the O(k²) vs O(d²) claim verified with actual memory numbers?
- [ ] **Statistical significance:** Multiple seeds with mean ± std?

```bash
# Check for standard deviation / error reporting
grep -n "std\|±\|confidence\|seed\|replicate\|repeat" paper/experiments.tex
```

---

## 4. SSS-Specific Requirements

For SSS Track D, experiments must demonstrate the DISTRIBUTED PROPERTY, not just accuracy:

**REQUIRED visualization:** Recovery-over-rounds (if missing, add TODO)
**REQUIRED table:** Communication overhead vs. FedAvg baseline
**REQUIRED discussion:** What happens when f EXCEEDS f* (the impossibility boundary)?
  This connects to Theorem (Impossibility) in self_stabilization.tex.
  Even 2 sentences: "When σ²f² ≥ 0.25, our detection fails as predicted by Theorem X;
  Figure Y shows accuracy degrading below the FedAvg baseline at this boundary."

---

## 5. Figure Caption Quality Audit

```bash
grep -n "\\\\caption{" paper/experiments.tex
```

For each caption found, classify as:
- GOOD: Explains the research insight (what the figure proves about the protocol)
- BAD: Describes only the axes without insight

For each BAD caption, provide an improved version with:
- What the figure demonstrates about the protocol's distributed properties
- Specific numbers (R rounds to recovery, f Byzantine fraction)
- What baselines do in comparison

---

## Output: EXPERIMENT_AUDIT.md

```markdown
# EXPERIMENT_AUDIT.md

## Critical Issues (Fix Before Submission)
1. [FATAL] Identical 63.4% baselines — explanation: [found/not found]
   Action taken: [describe]
2. [CRITICAL] 100% detection rate — scoped correctly: [yes/no]
   Action taken: [describe]
3. [CRITICAL] Recovery-over-rounds figure — exists: [yes/no]
   Action taken: [added/TODO marked/verified]

## Missing Experiments (Add If Time Allows)
- [ ] Adaptive spectral-evasion attack
- [ ] Error bars across multiple seeds
- [ ] f > f* boundary behavior

## Caption Improvements Applied
| Figure | Before | After |
|---|---|---|
| | | |

## Go/No-Go for Experiments Section
[ ] READY — results are reproducible and defensible
[ ] NEEDS WORK — [specific issues remaining]
```

*End of EXPERIMENTAL_VALIDITY_AGENT.md*
