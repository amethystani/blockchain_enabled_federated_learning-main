# ACADEMIC_TONE_AGENT.md
# Agent Role: Academic Style & Hype Scrubber

> **Purpose:** Transform the prose from "promotional" to "objective and rigorous."
> **Goal:** Remove all "AI-isms" and hype language that triggers SSS reviewers.
> **Run as the FINAL step before submission.**

---

## Agent Instructions

Search for and replace the following categories of non-academic language:

### 1. The "Hype" Scrubber
Identify and replace these words/phrases with objective alternatives:

| Forbidden Hype | Suggested Replacement |
|---|---|
| revolutionary / game-changing | "novel" or "first to provide [X]" |
| unprecedented / groundbreaking | "addresses previously unsolved [X]" |
| sophisticated / elegant | "efficient" or "structured" |
| perfectly / completely / 100% | "robustly" or "with high probability" |
| state-of-the-art (without cite) | "competitive with current baselines" |
| remarkable / amazing / incredible | (Remove entirely) |
| ensures absolute safety | "provides formal safety guarantees under..." |

---

### 2. The "AI-Filler" Scrubber
Rephrase or delete these "LLM-typical" transition phrases that sound generic:

- "In the ever-evolving landscape of..."
- "It is important to note that..."
- "Furthermore, it should be observed that..."
- "This highlights the significance of..."
- "Notably," (Use sparingly)
- "Deeply," "Extensively," "Broadly speaking,"

**Action:** Rewrite these sentences to be direct.
*Example:* "It is important to note that our protocol achieves O(f) rounds." -> "Our protocol achieves O(f) rounds."

---

### 3. The "ML-to-DS" Framing Check
Ensure the language stays in the "Distributed Systems" domain:

- Replace "accuracy performance" with "protocol convergence" or "model integrity."
- Replace "attack defense" with "fault tolerance" or "Byzantine resilience."
- Replace "training process" with "training protocol" or "synchronous rounds."

---

### 4. Anonymity & Compliance Check
1. **Anonymization:** Ensure `\author{Anonymous Author(s)}` is used.
2. **References:** Ensure you don't say "In our previous work [21]" where [21] is yours. Change to "In [21], the authors..."
3. **LNCS Style:** No bolding in abstracts. Use `\emph{}` instead of `\textit{}` for key terms.

---

## Output: STYLE_AUDIT_REPORT.md

Produce a report listing:
- [ ] Number of hype words removed.
- [ ] List of "AI-filler" sentences rewritten.
- [ ] Confirmation of anonymization.
- [ ] Suggestions for 3-5 sentences that need manual human rewriting for better "academic flow."

*End of ACADEMIC_TONE_AGENT.md*
