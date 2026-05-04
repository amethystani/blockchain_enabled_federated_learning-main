# SSS2026_TARGETING_GUIDE.md
# our approach → SSS 2026: Deep Venue Analysis & Positioning Guide

> **Researched:** May 2026 | Sources: SSS 2026 CFP (sss2026.conf.lip6.fr), SSS 2025 LNCS 16350,
> SSS 2024 LNCS 14931, DBLP full proceedings records
> **⚠️ DEADLINE ALERT: Round 2 = May 15, 2026 AoE (≈12 days). Round 3 = July 15 (fallback).**
> Read this entire file before touching any `.tex` source.

---

## 1. Verified SSS 2026 Facts

| Field | Verified Fact |
|---|---|
| Full name | 28th International Symposium on Stabilization, Safety, and Security of Distributed Systems |
| Dates | October 9–11, 2026, Gothenburg, Sweden |
| Publisher | Springer LNCS |
| Submission system | HotCRP at sss2026-submission.limos.fr |
| Review model | Double-blind (relaxed) — no names in PDF; arXiv/talks fine |
| General Co-Chairs | Sandeep Kulkarni (Michigan State) + Elad Schiller (Chalmers) |
| Steering Chair | Sébastien Tixeuil (Sorbonne Université) |
| Round 1 deadline | ~~March 31, 2026~~ PASSED |
| **Round 2 deadline** | **May 15, 2026 AoE ← YOUR TARGET** |
| Round 3 deadline | July 15, 2026 AoE ← fallback if Round 2 not ready |
| Regular paper limit | **15 pages** (incl. title, abstract, figures) + unlimited references |
| Brief announcement | 5 pages total including everything |
| Resubmission policy | R1/R2 rejects may resubmit to later round WITH prior reviews forwarded |
| Historical acceptance | SSS 2025: 21 full + 7 short from 59 submissions (~47%) |
|  | SSS 2024: 22 full + 6 short from 69 submissions (~41%) |

---

## 2. The Five SSS 2026 Tracks — Exact Fit Analysis

### TRACK A — Stabilization and Locality in Distributed Computing
**Chair: Stéphane Devismes (Université de Picardie Jules Verne, France)**

Explicit topics per CFP: Stabilizing Systems, Proof labelling schemes, Graph Algorithms,
Graph-theoretic concepts for communication networks, Social and Peer-to-Peer Networks,
LOCAL/CONGEST models, Communication complexity, Game-theory and economical aspects of
distributed computing, Dynamic networks, time-varying graphs, evolving graphs.

**our approach fit: SECONDARY (terminology only, not submission target)**

Track A reviewers include Devismes, Kamei (Hiroshima), Sudo (Hosei), and other
classical self-stabilization theorists. Papers accepted here have formal Dijkstra-style
convergence proofs. They will reject on the spot any paper that uses "self-stabilizing"
loosely.

SSS 2025 Track A example: "Self-stabilizing Mutual Exclusion in Dynamic Networks with
Bounded Temporal Diameter" (Devismes et al.) — formal proof of convergence from any
initial state within a bounded temporal diameter graph.

Strategy: Use Track A vocabulary (stabilization, recovery invariant, fault-containment)
in abstract and introduction to signal venue fluency. DO NOT submit to Track A unless
you have a formal classical self-stabilization proof.

---

### TRACK B — Time, Safety, and Security in Distributed Computing
**Chair: John Augustine (IIT Delhi, India)**

Explicit topics per CFP: Concurrent and fault-tolerant algorithms, Synchronization
protocols, Shared and transactional memory, **Blockchain technologies and cryptocurrencies**,
Formal methods semantics and verification, **Secure multi-party computation and cryptographic
distributed protocols**, Privacy-enhancing technologies and anonymity, Post-quantum and
information theoretic cryptography and security.

**our approach fit: SECONDARY (blockchain checkpointing + BFT algorithms)**

Recent SSS Track B acceptances directly relevant to our approach:
- SSS 2025: "Improving the Hu-Toueg Construction of a Byzantine Linearizable SWMR Register"
- SSS 2025: "Deterministic Causal Order Under Byzantine Sybil Tolerance: Techniques and Limitations"
- SSS 2025: "Byzantine Reliable Broadcast in Wireless Networks" (Lu, Liu, Ren)
- SSS 2024: "Byzantine Reliable Broadcast with One Trusted Monotonic Counter" (Amoussou-Guenou et al.)
- SSS 2024: "TRAIL: Cross-Shard Validation for Byzantine Shard Protection" (Oglio et al.)
- SSS 2025: "Near-Optimal Stability for Distributed Transaction Processing in Blockchain Sharding"

The blockchain checkpointing component and Byzantine fault tolerance formalism fit Track B.
Cross-reference this vocabulary in paper framing. If the FL component is removed and the
checkpointing protocol alone is analyzed, it could be a Track B paper.

---

### TRACK C — Moving and Computing
**Chair: Anissa Lamani (University of Strasbourg, France)**

Topics: Mobile agents, robots, sensor networks, population protocols, nature-inspired computing.

**our approach fit: NONE. Do not reference this track at all.**

---

### TRACK D — Distributed Artificial Intelligence and Machine Learning
**Chair: Anish Arora (Ohio State University) — SSS Steering Committee member**

Explicit topics per CFP:
- **Fault-tolerant distributed training** ← PRIMARY FIT
- Scheduling and resource allocation in ML clusters
- Consistency models for distributed ML
- **Communication efficient distributed machine learning** ← gradient sketching fit
- Edge Cloud co-training
- ML-aware storage systems
- Data flow optimization of ML pipelines
- Energy efficient distributed ML
- **Learning augmented consensus** ← spectral detection + aggregation fit
- Predictive failure detection ← Byzantine detection component fits
- **Convergence of asynchronous optimization** ← stabilization/recovery argument fits
- Lower bounds for distributed learning

**our approach fit: PRIMARY — submit here**

Understanding Track D chair Anish Arora is critical:
Arora is a foundational SSS community member (Ohio State, steering committee). His SSS 2025
paper with Chen and Lin ("Knowledge-Guided Machine Learning for Stabilizing Near-Shortest Path
Routing") uses ML predictions to improve stabilization time of a distributed routing protocol.
The paper is NOT about ML accuracy — it is about a distributed guarantee (stabilization time)
being improved by ML. This is exactly the lens through which our approach must be framed:
Byzantine robustness and recovery stability as distributed guarantees, with spectral ML as the
mechanism that achieves them.

SSS 2025 keynote on decentralised ML: Sonia Ben Mokhtar gave the keynote "On the Safety and
Security of Decentralised Machine Learning" — explicit community signal that secure/robust
distributed ML is welcome at SSS.

SSS 2025 accepted FL paper: "Label Leakage in Regression Federated Learning Using Cryptographic
Tools" (Jobic, Mayoue, Tucci-Piergiovanni) — direct federated learning precedent in proceedings.
This paper uses cryptographic tools + formal security definitions, consistent with SSS standards.

---

### TRACK E — Practical & Open Problems
**Chair: Magnus Almgren (Chalmers University of Technology, Sweden)**

Topics: Industrial deployment experiments, real world case studies, open problems in
large-scale configuration management, security and privacy failures in deployed systems.

**our approach fit: WEAK — only if you have real deployment data or an open-problems
framing. Could be a brief announcement fallback for Round 3.**

---

## 3. Primary Submission Decision

**Submit to: Track D (Distributed AI and Machine Learning)**

**Cross-reference vocabulary from: Track A (stabilization) + Track B (Byzantine fault tolerance,
blockchain)**

This combination — Track D paper with Track A/B vocabulary — is exactly what SSS 2025's
accepted ML papers did. It signals that the authors understand they are at a distributed
systems conference, not an ML conference.

---

## 4. The Self-Stabilization Question — The Most Critical Issue

SSS takes its name from self-stabilization. Dijkstra's 1974 paper "Self-Stabilizing Systems in
Spite of Distributed Control" is the founding document. The community includes Dijkstra Prize
winners and lifelong stabilization researchers. Misuse of "self-stabilizing" is detected
immediately and will cause rejection.

### What classical self-stabilization requires (Dijkstra/Dolev definition):
- System starts in ANY arbitrary configuration (including fully corrupted state)
- System autonomously reaches a legitimate/correct configuration
- Within a finite (ideally bounded) number of rounds/steps
- Using only local information and the protocol's own rules
- WITHOUT external reset or intervention

Reference: Dolev, S. (2000). Self-Stabilization. MIT Press. — This is what reviewers cite.

### Four honest framings for our approach (choose one and defend it):

**FRAMING 1 — Classical self-stabilization**
Claim: "our approach is self-stabilizing: starting from any model state corrupted by
Byzantine clients, the protocol autonomously recovers to a correct training trajectory within
O(R) rounds under assumptions A1–A4."
Requirement: Formal proof that for ANY starting configuration, the system reaches the
invariant within bounded rounds. Requires proving the sketch captures sufficient structure
from any starting point. HIGH BAR — only claim this if you have the proof.

**FRAMING 2 — Practical stabilization (RECOMMENDED for May 15 submission)**
Claim: "our approach provides practical stabilization: after any finite sequence of
Byzantine-contaminated rounds, the system re-enters a ε-bounded training trajectory within
R rounds, provided the Byzantine fraction remains below f* and honest client gradients
satisfy distributional assumption D1."
Requirement: Prove bounded recovery time R from contaminated state. Lower bar than classical
but still formal. Matches precedent of SSS 2025's "Knowledge-Guided ML for Stabilizing
Near-Shortest Path Routing."

**FRAMING 3 — Fault containment**
Claim: "our approach provides Byzantine fault containment: the influence of at most f
Byzantine clients on the aggregate update is bounded by ε per round, and the protocol
returns to a correct trajectory after Byzantine perturbation ceases."
Requirement: Prove the influence bound ε. Weakest claim but most honest if proofs are
incomplete. Still legitimate for Track D/B.

**FRAMING 4 — WRONG (will cause rejection)**
Claim: "Our system is fully self-stabilizing and guarantees convergence under arbitrary
Byzantine attacks." — This combines "fully self-stabilizing" with "arbitrary Byzantine
attacks" without impossibility arguments or fraction assumptions. Track A reviewers will
know this is impossible and reject the paper.

**For Round 2 (May 15): Use Framing 2 or 3. Upgrade to Framing 1 for Round 3 if proofs
are completed before July 15.**

---

## 5. SSS Reviewer Expectations — Evidence-Based Analysis

Based on systematic analysis of SSS 2023, 2024, 2025 accepted papers:

### For ALL SSS papers:
- Precise definitions before use — every technical concept explicitly defined
- Stated numbered assumptions — not buried in prose
- Scoped claims — "under assumptions A1–A4, we show..."
- Algorithm pseudocode — not just prose description
- Related work that contrasts, not just lists

### For Track D papers specifically (based on SSS 2025 ML papers):
- ML component must improve a DISTRIBUTED GUARANTEE (stabilization time, fault tolerance
  bound, convergence round count) — not just accuracy
- Experimental results show the distributed property improving
- Comparison to distributed baseline (not just ML baselines)
- Anish Arora will look for: round structure, fault model, recovery property

### For Byzantine-focused papers (Tracks B, D):
- Precise adversary model: fraction f, capabilities, knowledge
- Safety property and liveness property stated separately
- Proof or proof sketch (even informal) of key properties
- Comparison to prior Byzantine-tolerant algorithms

### What ALL accepted SSS papers have in common:
1. The problem is framed as a distributed systems problem first
2. ML/blockchain/cryptography are TOOLS that solve the distributed problem
3. Claims are scoped by explicit assumptions
4. There is formal content (algorithm + property statement) even if proofs are sketched
5. Related work includes SSS-community papers

---

## 6. Language Transformation Table

Every sentence in the paper must pass through this filter:

| ML-paper language (REJECT) | SSS-paper language (USE) |
|---|---|
| "We train a global model" | "The protocol executes T distributed training rounds" |
| "Byzantine clients send bad gradients" | "An adaptive adversary controls at most f clients and may inject arbitrary update vectors per round" |
| "Our defense improves accuracy by X%" | "The protocol bounds aggregate deviation from the honest centroid by ε under assumption A2" |
| "The model converges faster" | "The protocol reaches a stable training trajectory within R rounds after Byzantine contamination" |
| "We use sketching to save memory" | "Count-sketch reduces per-round communication complexity from O(d) to O(k) for sketch dimension k << d" |
| "Spectral filtering detects attackers" | "The spectral filter partitions the client update set into trusted subset T and suspect subset S based on eigenvalue structure of the sketch matrix M" |
| "Blockchain makes it more secure" | "A distributed ledger provides tamper-evident, verifiable checkpointing of the global model state, enabling Byzantine-detected rollback" |
| "Our system is robust" | "Under Assumption A3 (f < f* Byzantine clients) the protocol maintains safety invariant I throughout training" |
| "It works well in practice" | "Empirical evaluation over R attack scenarios demonstrates recovery within [bound] rounds" |
| "We propose a novel FL defense" | "We present a distributed training protocol with Byzantine fault tolerance and recovery guarantees" |

---

## 7. Section-by-Section Rewrite Strategy

### Abstract (target: 150–200 words, max 250)

Required content in order:
1. Federated learning as a Byzantine-adversarial distributed protocol (1 sentence)
2. The fault model: at most f Byzantine clients (1 sentence)
3. The three protocol stages: sketching → spectral filter → aggregation → checkpoint (2 sentences)
4. The stabilization/recovery property with its assumptions (1–2 sentences)
5. Evidence: datasets, attack types, baselines beaten (1–2 sentences)

Required vocabulary: "distributed protocol", "Byzantine fault", "self-stabilizing" OR
"practical stabilization", "verifiable checkpoint", "recovery"

Forbidden as lead claim: "accuracy", "novel defense", "deep learning"

EXAMPLE OPENING: "Federated learning, viewed as a distributed protocol operating under an
adversarial environment, must provide fault tolerance guarantees analogous to those of
Byzantine-resilient distributed systems..."

---

### Introduction (target: 1.5–2 pages)

Required paragraph structure:
1. Federated learning as a distributed protocol with Byzantine adversarial model
2. Why Byzantine attacks break existing distributed training protocols
3. Why existing robust aggregators (Krum, Trimmed Mean, FLTrust, Bulyan) are insufficient
   — frame as limitations of the distributed protocol, not just ML defense failures
4. The our approach protocol — describe the pipeline as a distributed round structure
5. Stabilization framing — define carefully what "stabilization" means here, cite Dolev 2000
6. Numbered contributions — each must be a falsifiable distributed-systems claim
7. Paper roadmap

FORBIDDEN opening: "In recent years, machine learning..." or "Deep learning has..."
REQUIRED opening register: distributed systems, fault tolerance, adversarial environment

---

### System Model (ADD THIS SECTION if missing)
Target: 0.5–1 page. This section cannot be skipped for an SSS paper.

Required elements:
- N clients C = {c_1, ..., c_N}, one server S
- Round structure: synchronous / partially synchronous / asynchronous (SPECIFY ONE)
- What happens in each round: broadcast model, local computation, upload sketch, server aggregates
- Byzantine subset B ⊆ C, |B| ≤ f (state whether f is static or dynamic)
- Honest client model: local dataset D_i, gradient computation g_i = ∇L(w_t; D_i)
- Adversary knowledge: does Byzantine adversary know honest updates? Know the filter?
- Server capabilities: what can the server see, compute, store?
- Communication model: what is sent, when, in what format

---

### Threat Model (ADD THIS SECTION if missing)
Target: 0.5 page. Required for any SSS Byzantine paper.

Required elements:
- Byzantine client capabilities: arbitrary gradient vectors, coordination among Byzantine clients
- Attack types explicitly considered: sign-flip, label-flip, Gaussian noise, scaling, model-replacement
- What the adversary CANNOT do: cannot compromise server, cannot exceed f clients, cannot break
  cryptographic primitives (if blockchain used)
- Threat boundary statement: "This protocol does NOT protect against..." (be honest)
- Whether attacks are static (fixed Byzantine set) or adaptive (adversary can change strategy)

---

### Algorithm Section (target: 1–1.5 pages)

Required:
- Algorithm pseudocode (LaTeX algorithm2e or algorithmicx package)
- Round structure labeled: Client Phase / Server Phase
- Count-sketch construction: parameters d (dimension), k (sketch size), hash functions
- Spectral filter: how eigenvalue decomposition is applied, threshold τ, how trusted set T is selected
- Robust aggregation: which aggregator (geometric median? trimmed mean?), applied to T only
- Checkpoint/commit: what is stored, how verified
- Next-round model broadcast

---

### Stabilization Argument (ADD THIS SECTION if missing)
Target: 0.5–1 page. This is the core SSS-identity section.

Structure:
- Define legitimate configuration / stable training trajectory formally
- State safety invariant: "In every round t, if |B| ≤ f, then the aggregate update deviates
  from the honest centroid by at most ε"
- State recovery property: "After at most R rounds following Byzantine perturbation, the
  protocol returns to a stable trajectory"
- Lemma 1: Count-sketch preserves directional structure [proof or TODO]
- Lemma 2: Spectral filter separates Byzantine updates when perturbation exceeds τ [proof or TODO]
- Lemma 3: Aggregation over trusted set bounds deviation [proof or TODO]
- Theorem: Under assumptions A1–A4, our approach is practically stabilizing

If proofs are not complete, use: "Theorem (empirically supported, formal proof in progress)"
and cite your experimental recovery results. This is honest and acceptable for Round 2.

---

### Evaluation (target: 2–3 pages)

Required axes for SSS acceptance:
1. Byzantine fraction sensitivity: f/N = 0.1, 0.2, 0.3, 0.4 (more is better)
2. Attack diversity: sign-flip, label-flip, Gaussian noise, scaling, model-replacement
3. RECOVERY OVER ROUNDS — plot accuracy vs. round number with attack at round T_start
   This is the critical SSS-specific metric. Shows stabilization behavior.
4. Baselines: FedAvg, Krum, Multi-Krum, Trimmed Mean, Coordinate Median, FLTrust, Bulyan
5. Ablation: without sketching / without spectral filter / without checkpoint
6. Communication overhead: bytes per round vs. FedAvg baseline
7. Runtime overhead: wall-clock time per round
8. Non-IID sensitivity: different α values in Dirichlet distribution
9. Recovery time R: how many rounds to return to ε-neighborhood of clean trajectory

Figure captions MUST explain the research insight, not just describe axes.
BAD: "Figure 3: Accuracy vs. rounds under sign-flip attack."
GOOD: "Figure 3: Recovery dynamics under sign-flip attack with f=0.3 Byzantine clients.
our approach returns to within 2% of unattacked accuracy within 15 rounds of attack
onset (marked ↑), while FLTrust and Krum fail to recover within the 100-round window."

---

### Limitations (target: 0.25–0.5 page)
Required — SSS reviewers trust papers more when limitations are honest:
- What happens when f exceeds f* (Byzantine threshold)
- Adaptive spectral attack: adversary that learns eigenvalue structure and crafts undetectable updates
- Non-IID severity: when honest gradients are very heterogeneous, does the spectral filter fail?
- Formal proofs: explicitly state which results are empirical vs. proved
- Blockchain overhead: scalability at large N

---

### Conclusion (target: 0.25–0.5 page)
Must restate:
- The distributed-systems contribution in one sentence
- The stabilization/recovery guarantee with its assumptions explicitly named
- One or two concrete open problems in distributed-systems terms (e.g., "extending to
  asynchronous round models remains open" or "closing the gap between empirical and
  formal convergence bounds is future work")

---

## 8. Citation Strategy

### Must cite (SSS community):
- Dijkstra, E.W. (1974). Self-stabilizing systems in spite of distributed control. CACM.
- Dolev, S. (2000). Self-Stabilization. MIT Press.
- SSS 2025: Chen, Lin, Arora — "Knowledge-Guided Machine Learning for Stabilizing Near-Shortest Path Routing" (LNCS 16350)
- SSS 2025: Jobic, Mayoue, Tucci-Piergiovanni — "Label Leakage in Regression Federated Learning" (LNCS 16350)
- SSS 2024: Amoussou-Guenou et al. — "Byzantine Reliable Broadcast with One Trusted Monotonic Counter"
- SSS 2024: Oglio, Nesterenko, Sharma — "TRAIL: Cross-Shard Validation for Byzantine Shard Protection"
- SSS 2025: Lu, Liu, Ren — "Byzantine Reliable Broadcast in Wireless Networks"

### Must cite (BFT FL foundations):
- Blanchard et al. (2017) — Krum / Byzantine-Tolerant ML (NeurIPS)
- Yin et al. (2018) — Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates (ICML)
- El Mhamdi et al. (2018) — The Hidden Vulnerability of Distributed Learning in Byzantium
- Cao et al. (2020) — FLTrust (NDSS)
- Guerraoui et al. (2018) — Bulyan

### Must cite (sketching/compression):
- Charikar et al. (2002) — Count-Sketch
- Ivanov et al. — sketching for federated learning (search recent)

### Forbidden citation strategy:
- Do NOT cite only ML conferences (NeurIPS, ICML, ICLR, CVPR)
- Every ML citation should be balanced with a distributed-systems citation
- The related work section must show awareness of the SSS proceedings

---

## 9. Double-Blind Compliance Checklist

SSS 2026 uses relaxed double-blind. Required:
- Remove all author names and affiliations from PDF
- Remove `\author{}` content from LNCS template
- Remove acknowledgments that identify the lab/grant
- Do not write "In our previous work [X]" — write "In [X], the authors show..."
- Do not include project URLs or lab websites in paper
- arXiv posting is fine — just anonymize the submission PDF itself

---

## 10. LNCS Format Requirements

- Use Springer LNCS template: llncs.cls
- 15 pages maximum (title + abstract + body + figures + tables; references excluded)
- References can be unlimited length — do not compress the reference list
- Figures must be legible in grayscale (some reviewers print in B&W)
- Algorithm pseudocode: use algorithm2e or algorithmicx (not custom verbatim)
- Theorem/Lemma/Definition: use standard LNCS theorem environments
- Do not redefine \section or change margins
- Check: latexmk -pdf main.tex compiles cleanly with no undefined references

---

## 11. Submission Framing on HotCRP

When submitting to HotCRP:
- Title: Keep "our approach: Self-Stabilizing Byzantine-Robust Federated Learning"
  OR consider: "our approach: A Practically Stabilizing Byzantine-Robust Federated
  Learning Protocol" (more precise, safer claim)
- Track selection: Track D — Distributed Artificial Intelligence and Machine Learning
- Abstract on HotCRP: same as paper abstract (ensure it uses distributed-systems vocabulary)
- Keywords: include "Byzantine fault tolerance", "self-stabilizing", "federated learning",
  "spectral filtering", "fault-tolerant distributed training"
- Student paper flag: check if eligible for best student paper

---

## 12. Round 2 vs Round 3 Decision Framework

**Submit Round 2 (May 15) if:**
- System model and threat model sections exist (even if concise)
- Algorithm pseudocode is complete
- Stabilization argument exists (even informal with TODO markers on proofs)
- At least 4 baselines compared
- Recovery-over-rounds plot exists
- Abstract and introduction use distributed-systems language

**Wait for Round 3 (July 15) if:**
- Any of the above are completely missing
- Experiments are incomplete (less than 3 attack types)
- The paper still reads primarily as an ML paper with no system model

**Strategic advantage of Round 2:** If rejected, you receive reviews AND can resubmit to
Round 3 with those reviews incorporated. The three-round structure is designed for this.
Submitting Round 2 with a borderline paper is better than waiting if you can fix the core
framing issues in 12 days.

---

*End of SSS2026_TARGETING_GUIDE.md — proceed to PAPER_REVIEWER_AGENT.md*
