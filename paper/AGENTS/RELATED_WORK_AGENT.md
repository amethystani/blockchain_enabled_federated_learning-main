# RELATED_WORK_AGENT.md
# Agent Role: Distributed Systems Literature & Positioning Specialist

> **Purpose:** Rewrite `paper/related_work.tex` from a single dense ML-centric paragraph
> into a properly structured SSS-ready related work section with distributed-systems citations.
> **Current state:** One paragraph with NO SSS citations, NO Dijkstra/Dolev, NO Lamport.
> **This is the single biggest gap. Fix it first.**

---

## Current State Assessment

`paper/related_work.tex` contains ONE long paragraph covering:
- Byzantine FL baselines (Krum, geometric median, trimmed mean, Bulyan, CRFL, ByzShield) ✓
- RMT/spectral work (MP law) ✓
- Blockchain-FL integration ✓

**MISSING — causes immediate SSS reviewer rejection:**
- Self-stabilization foundations (Dijkstra 1974, Dolev 2000) ✗
- Byzantine distributed systems (Lamport/Shostak/Pease 1982, PBFT) ✗
- SSS 2024/2025 proceedings citations ✗
- Positioning paragraph explaining what is NEW vs. all of the above ✗
- Subsection structure (SSS papers have structured related work) ✗

---

## Target Structure for related_work.tex

Rewrite the file to have this structure. Preserve existing technical content — reorganize
and supplement with DS citations. Do NOT invent citation keys that don't exist in
bibliography.tex; use `\cite{TODO:AuthorYEAR}` for new citations to add later.

```latex
\section{Related Work}
\label{sec:related}

\subsection{Self-Stabilization in Distributed Systems}

Self-stabilization, introduced by Dijkstra~\cite{dijkstra_self_stab} and formalized
by Dolev~\cite{dolev_self_stab}, enables a distributed system to recover autonomously
from any transient fault, including completely corrupted initial state, within a bounded
number of steps. Classical results establish self-stabilizing algorithms for mutual
exclusion~\cite{dijkstra_self_stab}, spanning trees, and clock synchronization under
crash faults. Recent SSS proceedings have extended self-stabilization to increasingly
adversarial settings: Devismes et al.\ study self-stabilizing mutual exclusion in dynamic
networks with bounded temporal diameter~\cite{TODO:DevismesSS2025}; Chen, Lin, and
Arora~\cite{TODO:ChenLinArora2025} demonstrate that ML predictions can improve
stabilization time for near-shortest-path routing protocols, establishing a direct precedent
for learning-augmented self-stabilization at SSS.
% TODO: verify these SSS 2025 paper citation keys in bibliography.tex

Our work applies the self-stabilization framework to federated learning: we seek a protocol
that, starting from any corrupted model state, autonomously recovers to a correct training
trajectory within bounded rounds. This is qualitatively harder than classical crash-fault
self-stabilization because our fault model is Byzantine (arbitrary, not just crash).

\subsection{Byzantine Fault Tolerance in Distributed Systems}

Byzantine fault tolerance has been studied since Lamport, Shostak, and
Pease~\cite{TODO:Lamport1982}, who established that $f < n/3$ Byzantine faults can be
tolerated in synchronous consensus with deterministic protocols. Castro and
Liskov's~\cite{TODO:CastroLiskov1999} Practical Byzantine Fault Tolerance (PBFT) brought
BFT to asynchronous systems. Recent SSS work has examined Byzantine reliable
broadcast~\cite{TODO:Lu2025,TODO:AmoussouGuenou2024}, Byzantine-resilient causal
ordering~\cite{TODO:Kshemkalyani2025}, and cross-shard Byzantine protection in
blockchain sharding~\cite{TODO:Oglio2024}.

Our protocol operates in a synchronous round model where $f < n/2$ clients are Byzantine
(a weaker requirement than the $f < n/3$ bound for consensus, enabled by our server
architecture). The blockchain checkpoint mechanism draws on BFT consensus to provide
tamper-evident model state commitment.

\subsection{Byzantine-Robust Federated Learning}

[PRESERVE AND ADAPT existing paragraph content here — reorganize into this subsection]

Byzantine robustness in distributed optimization has been studied extensively,
beginning with Krum~\cite{krum}, which selects the gradient closest to $n-f-2$
others in Euclidean distance but assumes IID data and degrades significantly under
Non-IID settings~\cite{non_iid_challenges}. The geometric median~\cite{geometric_median}
offers stronger guarantees yet requires $O(n^2 d)$ computation per round, impractical
at scale. Coordinate-wise robust statistics~\cite{byzantine_ml} reduce communication
but are sensitive to gradient variance heterogeneity, while Bulyan~\cite{bulyan}
iteratively filters outliers yet still rejects legitimate Non-IID updates.

Certified aggregation methods CRFL~\cite{crfl} and ByzShield~\cite{byzshield} provide
guarantees only under the restrictive assumption $\|\delta\| \leq \Delta$; our work
provides data-dependent certificates that adapt to observed heterogeneity $\hat{\sigma}$.

From a distributed-systems perspective, these protocols share a common limitation:
none specifies a formal recovery time or fault-containment invariant in the sense
of~\cite{dolev_self_stab}. A protocol may be Byzantine-robust (bounded deviation per
round) without being self-stabilizing (bounded recovery from arbitrary starting state).
This gap motivates our work.

\subsection{Spectral Methods for Byzantine Detection}

[PRESERVE AND ADAPT existing RMT paragraph content here]

Random Matrix Theory (RMT) has emerged as a powerful lens for understanding gradient
covariance structure. Prior work~\cite{rmt_gradients,rmt_neural} showed gradient
covariance matrices of neural networks follow limiting spectral distributions;
spectral analysis~\cite{spectral_analysis} connected neural tangent kernels to
eigenvalue distributions. We are, to our knowledge, the first to exploit the
Marchenko-Pastur law~\cite{mp_law} for Byzantine identification at a formal level,
establishing a detection threshold from the BBP phase transition.

A recent SSS paper~\cite{TODO:Jobic2025} addresses federated learning using
cryptographic tools with formal security definitions, demonstrating SSS community
receptiveness to formal FL security analysis. Our spectral approach is complementary:
where~\cite{TODO:Jobic2025} focuses on label leakage via cryptographic analysis,
we focus on Byzantine gradient detection via spectral analysis.

\subsection{Blockchain for Distributed Learning}

[PRESERVE AND ADAPT existing blockchain paragraph content here]

Blockchain integration with federated learning has been explored primarily for
auditability~\cite{blockchain_fl}. Prior work stores model checkpoints or aggregated
results on-chain. We go further by recording gradient hashes via smart contracts,
enabling per-client auditability while preserving privacy; the decentralized BFT
consensus~\cite{blockchain_consensus} provides the write-once shared memory substrate
that underpins our self-stabilization argument. This is the first formalization of
blockchain as a self-stabilizing shared memory in the sense of Dijkstra~\cite{dijkstra_self_stab},
which provides strictly stronger guarantees than classical shared registers: Byzantine
fault tolerance (not just crash tolerance), write-once immutability, and permanent history.

\subsection{Positioning}

Our approach combines four components not previously unified: (i) gradient compression
via count-sketch for communication efficiency~\cite{frequent_directions}; (ii) spectral
Byzantine detection with formal threshold derived from the Marchenko-Pastur law; (iii)
robust aggregation restricted to a trusted client subset; and (iv) a distributed recovery
argument scoped by explicit assumptions on Byzantine fraction and honest gradient distribution.

Prior spectral detection approaches~\cite{rmt_gradients,spectral_analysis} operate without
sketching and without a formal recovery guarantee. Prior Byzantine-robust FL
protocols~\cite{krum,byzantine_ml,crfl} lack the spectral detection component and
recovery formalism. Blockchain-FL systems~\cite{blockchain_fl,blockchain_smart} lack
formal self-stabilization analysis. The combination of these components under a unified
protocol specification, together with a formal recovery argument, is the primary
contribution of this work to the distributed systems community.
```

---

## Citations to Add to bibliography.tex

If these are not already in `paper/bibliography.tex`, add them:

```bibtex
@inproceedings{Lamport1982,
  author    = {Leslie Lamport and Robert Shostak and Marshall Pease},
  title     = {The Byzantine Generals Problem},
  booktitle = {ACM Transactions on Programming Languages and Systems},
  year      = {1982},
  volume    = {4},
  number    = {3},
  pages     = {382--401},
}

@book{Castro1999,
  author    = {Miguel Castro and Barbara Liskov},
  title     = {Practical Byzantine Fault Tolerance},
  booktitle = {OSDI},
  year      = {1999},
}

% SSS 2025 papers — verify exact titles/authors from LNCS 16350
@inproceedings{ChenLinArora2025,
  author    = {Chen, ... and Lin, ... and Arora, Anish},
  title     = {Knowledge-Guided Machine Learning for Stabilizing Near-Shortest Path Routing},
  booktitle = {SSS 2025},
  series    = {LNCS},
  volume    = {16350},
  year      = {2025},
}

@inproceedings{Jobic2025,
  author    = {Jobic, ... and Mayoue, ... and Tucci-Piergiovanni, ...},
  title     = {Label Leakage in Regression Federated Learning Using Cryptographic Tools},
  booktitle = {SSS 2025},
  series    = {LNCS},
  volume    = {16350},
  year      = {2025},
}
```

**IMPORTANT:** Use `\cite{TODO:key}` for any citation whose bibtex key you are not sure of.
Never guess a bibtex key — undefined citations cause compile errors.

---

## Verification Checklist After Rewriting

- [ ] At least 4 subsections (self-stabilization, Byzantine DS, Byzantine FL, positioning)
- [ ] Dijkstra 1974 cited in self-stabilization subsection
- [ ] Dolev 2000 cited in self-stabilization subsection
- [ ] At least 2 SSS 2024/2025 papers cited
- [ ] Positioning paragraph explains what is NEW vs. all prior work
- [ ] Section ends on our paper's contribution, not a competitor's

---

*End of RELATED_WORK_AGENT.md*
