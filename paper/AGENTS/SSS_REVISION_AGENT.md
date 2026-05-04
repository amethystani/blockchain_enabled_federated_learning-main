# SSS_REVISION_AGENT.md
# Agent Role: SSS 2026 Paper Revision Specialist

> **Purpose:** Directly revise `.tex` files to make our approach competitive for
> SSS 2026 Track D. Run AFTER PAPER_REVIEWER_AGENT.md has produced REVIEW_REPORT.md.
> **Constraint:** Never invent results. Never fabricate citations. Never delete content
> without preserving it in a comment. Mark all missing content with TODO:.

---

## KNOWN PAPER STATE — Skip Re-Creating These

The following sections are ALREADY WRITTEN and structurally SSS-appropriate.
DO NOT recreate them — only targeted edits based on REVIEW_REPORT.md findings:

| File | Status | What to do |
|---|---|---|
| `paper/system_model.tex` | ✓ Strong | Only edit if REVIEW_REPORT flags specific gaps |
| `paper/self_stabilization.tex` | ✓ Strong | Only edit if notation inconsistencies found |
| `paper/algorithm.tex` | ✓ Adequate | Only edit if pseudocode clarity issues flagged |
| `paper/introduction.tex` | ✓ Good | Strip any remaining hype; verify contributions match paper |
| `paper/related_work.tex` | ✗ WEAK | REWRITE using RELATED_WORK_AGENT template |
| `paper/conclusion.tex` | ✗ WEAK | EXPAND using Revision 8 template below |
| `paper/experiments.tex` | ~ Mixed | Add recovery-over-rounds, fix identical baselines |
| `paper/limitations.tex` | ✓ Exists | Verify adaptive adversary limitation is stated |

**PRIORITY ORDER (do in order, stop if time runs out):**
1. `paper/related_work.tex` — REWRITE (see RELATED_WORK_AGENT.md)
2. `paper/conclusion.tex` — EXPAND
3. `paper/experiments.tex` — Add recovery discussion, investigate 63.4% coincidence
4. `paper/main.tex` (abstract) — Strip hype, scope "100% detection" claim
5. `paper/introduction.tex` — Verify contributions numbered and match actual paper

## Pre-Revision Setup

```bash
# Step 1: Confirm paper directory structure
find paper/ -name "*.tex" | sort
ls paper/figures/

# Step 2: Attempt a clean compile
cd paper && latexmk -pdf main.tex 2>&1 | tail -30
cd ..

# Step 3: Read REVIEW_REPORT.md and EXPERIMENT_AUDIT.md (if they exist)
cat paper/REVIEW_REPORT.md 2>/dev/null | head -100
cat paper/EXPERIMENT_AUDIT.md 2>/dev/null | head -50

# Step 4: Save git state
git status

# Step 4: Read REVIEW_REPORT.md
cat REVIEW_REPORT.md
```

---

## File Detection Logic

The paper may use any file structure. Claude Code must:

```bash
# Detect which files exist
for candidate in abstract.tex introduction.tex intro.tex related.tex related_work.tex \
    relatedwork.tex model.tex system_model.tex threat_model.tex algorithm.tex \
    methodology.tex method.tex analysis.tex proof.tex experiments.tex evaluation.tex \
    eval.tex experiment.tex conclusion.tex conclusions.tex main.tex paper.tex; do
  [ -f "$candidate" ] && echo "FOUND: $candidate"
done
```

Map detected files to revision targets below. If the paper is a monolithic `main.tex`,
work section by section within that file.

---

## Revision 1: Abstract

**File:** Detect from above (often in main.tex or abstract.tex)

**Read the current abstract first:**
```bash
# Find abstract content
grep -n "\\\\begin{abstract}" main.tex
# Read 20 lines from that point
```

**Revision rules:**

PRESERVE: The core technical claims — what the protocol does mechanically.
TRANSFORM: The framing — from ML defense to distributed protocol.
ADD: Stabilization/recovery property statement.
ADD: Explicit fault model (f Byzantine clients).
REMOVE: Any sentence that leads with accuracy as the primary result.
REMOVE: Hype language.

**Required abstract structure (do NOT deviate):**
```
Sentence 1: Federated learning as a distributed protocol in adversarial environments
Sentence 2: The challenge — Byzantine clients can inject arbitrary updates, breaking protocol correctness
Sentence 3: Existing approaches [Krum / FLTrust / etc.] lack [specific distributed property]
Sentences 4–5: The our approach protocol — three stages described as protocol pipeline
Sentence 6: The stabilization/recovery guarantee (scoped by assumptions)
Sentence 7–8: Empirical validation — datasets, attack types, baselines
```

**LaTeX revision approach:**
```latex
% BEFORE (ML framing — DO NOT KEEP):
% "We propose a novel Byzantine-robust federated learning method that uses
%  spectral analysis to detect malicious clients, achieving X% accuracy..."

% AFTER (SSS framing — USE THIS):
\begin{abstract}
Federated learning, when operated in environments where clients may behave
arbitrarily, constitutes a Byzantine-adversarial distributed protocol requiring
formal fault tolerance guarantees. A Byzantine adversary controlling up to $f$
clients may inject crafted gradient updates that compromise model integrity and
prevent convergence. Existing robust aggregation rules—including Krum, Trimmed
Mean, and FLTrust—lack explicit distributed recovery guarantees and do not
specify the rounds required to stabilize after Byzantine perturbation.

We present \emph{our approach}, a fault-tolerant federated learning
protocol in which: (i) clients compress local gradients into compact
count-sketches; (ii) the server applies a spectral filter to the sketch matrix,
separating update directions associated with Byzantine behavior from those of
honest clients; (iii) trusted updates are robustly aggregated; and (iv) the
resulting model state is committed through a verifiable distributed checkpoint.

Under the assumption that the Byzantine fraction satisfies $f < f^*$ and honest
client gradients satisfy distributional condition~\ref{assum:honest},
our approach provides a bounded recovery guarantee: after any sequence of
contaminated rounds, the protocol returns to an $\varepsilon$-stable training
trajectory within $R$ rounds. % TODO: state R explicitly once proved/measured

Experiments on [DATASETS] under [ATTACK TYPES] demonstrate [RESULT FRAMED AS
RECOVERY ROUNDS AND DEVIATION BOUND, NOT JUST ACCURACY].
\end{abstract}
```

**If the current abstract is close to the SSS framing:** Make targeted word substitutions
only. Do not rewrite if 70%+ is already correct.

---

## Revision 2: Introduction

**File:** introduction.tex or intro.tex or Section 1 of main.tex

**Read current introduction:**
```bash
# Find \section{Introduction} and read ~100 lines
grep -n "\\\\section{" main.tex | head -20
```

**Required structural changes:**

### Opening paragraph (most critical — must change if ML-framed)

FORBIDDEN opening pattern:
```
"In recent years, machine learning [OR deep learning OR federated learning] has 
emerged as [revolutionary / transformative / powerful]..."
```

REQUIRED opening pattern:
```latex
% Opening must frame FL as a distributed protocol problem
Federated learning constitutes a distributed protocol in which $n$ clients
collaboratively optimize a shared model over $T$ rounds, communicating
compressed updates to a central server without exposing local data~\cite{McMahan2017}.
When clients may behave arbitrarily---sending crafted gradient vectors to
corrupt the global model---the protocol must provide Byzantine fault tolerance
guarantees analogous to those studied in distributed computing since Lamport,
Shostak, and Pease's foundational work~\cite{Lamport1982}.
```

### Motivation paragraph — reframe existing work gaps in DS terms

Add this framing to the motivation:
```latex
% TODO: Insert after describing the gap in existing FL defenses
Robust aggregation rules such as Krum~\cite{Blanchard2017}, Trimmed
Mean~\cite{Yin2018}, and FLTrust~\cite{Cao2020} improve Byzantine resilience
empirically, but lack formal distributed recovery guarantees: they do not specify
the number of rounds required to return to a stable training trajectory after
Byzantine contamination, nor do they provide a fault-containment invariant
bounding Byzantine influence per round. In the language of distributed fault
tolerance~\cite{Dolev2000}, these protocols do not self-stabilize.
```

### Contributions — must be rewritten to be precise and falsifiable

FORBIDDEN contribution style:
```
"1. We propose a novel spectral Byzantine detection mechanism."
"2. We achieve state-of-the-art performance."
"3. We conduct extensive experiments."
```

REQUIRED contribution style:
```latex
\textbf{Contributions.} This paper makes the following contributions:
\begin{enumerate}
  \item \textbf{Distributed protocol specification.} We present our approach,
    a fault-tolerant federated learning protocol with an explicit round structure,
    formal system model, and Byzantine adversary model (Section~\ref{sec:model}).
    
  \item \textbf{Spectral Byzantine filter.} We define a spectral filtering
    procedure over the count-sketch matrix that partitions client updates into
    trusted and suspect subsets based on eigenvalue structure, with formal
    threshold $\tau$ (Section~\ref{sec:algorithm}).
    
  \item \textbf{Stabilization argument.} We provide a [formal / semi-formal]
    recovery argument showing that under assumptions~\ref{assum:byzantine}
    and~\ref{assum:honest}, the protocol returns to an $\varepsilon$-stable
    training trajectory within $R$ rounds of Byzantine perturbation
    (Section~\ref{sec:analysis}). % TODO: strengthen to formal proof if possible
    
  \item \textbf{Verifiable checkpointing.} We specify a distributed checkpoint
    mechanism that provides tamper-evident model state commitment between rounds,
    enabling Byzantine-detected rollback (Section~\ref{sec:algorithm}).
    
  \item \textbf{Experimental validation.} We evaluate our approach under
    [N attack types] across [M datasets] with [K baselines], measuring recovery
    time $R$ and communication overhead alongside accuracy
    (Section~\ref{sec:experiments}).
\end{enumerate}
```

### Stabilization positioning paragraph (ADD THIS — critical for SSS)

Add before the roadmap paragraph:
```latex
\paragraph{On self-stabilization.}
% CHOOSE ONE of the following based on what is provable:

% OPTION A — if formal proof exists:
our approach satisfies a practical self-stabilization property in the
tradition of Dijkstra~\cite{Dijkstra1974} and Dolev~\cite{Dolev2000}: starting
from any model state corrupted by Byzantine perturbation, the protocol
autonomously recovers to a correct training trajectory within bounded rounds
under stated assumptions, without external intervention.

% OPTION B — if only empirical support exists:
We adopt the term \emph{practically stabilizing} to describe our approach's
recovery behavior: empirical results show that the protocol consistently returns
to a stable training trajectory within $R$ rounds after Byzantine attack onset,
suggesting a stabilization property analogous---though not formally equivalent---to
classical self-stabilization~\cite{Dijkstra1974,Dolev2000}. Establishing formal
recovery bounds under general distributional assumptions remains future work.
```

---

## Revision 3: System Model (ADD SECTION)

**If system model section is missing, CREATE it before the algorithm section.**

```latex
\section{System Model}
\label{sec:model}

\subsection{Participants and Communication}
We consider a synchronous distributed system comprising $n$ clients
$\mathcal{C} = \{c_1, \ldots, c_n\}$ and a single server $S$.
% TODO: justify synchrony assumption — if asynchronous, change to:
% "We consider a partially synchronous model with known bound \Delta on
%  message delay after the Global Stabilization Time (GST)."

Training proceeds in discrete rounds $t = 1, 2, \ldots, T$.
In each round $t$, the server broadcasts the current global model $w_t$,
each client computes a local gradient estimate and produces a compressed
representation (count-sketch), and transmits it to the server.
The server applies the spectral filter, performs robust aggregation, and
updates the global model to $w_{t+1}$.

\subsection{Byzantine Fault Model}
\label{subsec:byzantine}
An adversary $\mathcal{A}$ controls an unknown subset $\mathcal{B} \subseteq \mathcal{C}$
of Byzantine clients, with $|\mathcal{B}| \leq f$.
Byzantine clients may deviate arbitrarily from the protocol: they may send any
vector $\tilde{g}_i \in \mathbb{R}^d$ in place of their true gradient, coordinate
with other Byzantine clients, and observe the current model $w_t$.

\begin{assumption}[Byzantine Fraction]
\label{assum:byzantine}
The number of Byzantine clients satisfies $f < f^*$, where $f^*$ is the
Byzantine threshold determined by the spectral filter (Definition~\ref{def:filter}).
\end{assumption}

% TODO: determine f* analytically from spectral gap condition

\subsection{Honest Client Model}
\label{subsec:honest}
Each honest client $c_i \in \mathcal{C} \setminus \mathcal{B}$ holds a local
dataset $D_i$ and computes a stochastic gradient estimate
$g_i^t = \nabla \mathcal{L}(w_t; \xi_i^t)$ where $\xi_i^t \sim D_i$.

\begin{assumption}[Honest Gradient Distribution]
\label{assum:honest}
Honest client gradients satisfy: (i) $\mathbb{E}[g_i^t] = \nabla \mathcal{L}(w_t; D_i)$
(unbiasedness); (ii) $\mathrm{Var}[g_i^t] \leq \sigma^2$ (bounded variance);
(iii) TODO: add any distributional condition needed for spectral separability.
\end{assumption}

\subsection{Adversary Knowledge and Capabilities}
The adversary $\mathcal{A}$ is adaptive: it may change the attack strategy
between rounds, observe the current model $w_t$, but cannot:
(i) compromise the server $S$;
(ii) control more than $f < f^*$ clients;
(iii) break the cryptographic commitments used in the checkpoint mechanism
(assuming a collision-resistant hash function).

\subsection{Round Structure}
Each round $t$ proceeds as follows:
\begin{enumerate}
  \item \textbf{Broadcast:} Server $S$ broadcasts $w_t$ to all clients.
  \item \textbf{Local computation:} Each client $c_i$ computes $g_i^t$ and
    constructs count-sketch $\tilde{g}_i^t \in \mathbb{R}^k$ (Algorithm~\ref{alg:sketch}).
  \item \textbf{Upload:} Each client transmits $\tilde{g}_i^t$ to $S$.
  \item \textbf{Spectral filtering:} $S$ applies the spectral filter
    (Algorithm~\ref{alg:filter}) to partition clients into trusted set
    $\mathcal{T}_t$ and suspect set $\mathcal{S}_t$.
  \item \textbf{Aggregation:} $S$ computes $\bar{g}_t$ from
    $\{\tilde{g}_i^t\}_{i \in \mathcal{T}_t}$.
  \item \textbf{Update and checkpoint:} $S$ updates $w_{t+1} = w_t - \eta \bar{g}_t$
    and commits checkpoint $\mathrm{ckpt}_t = \mathrm{Hash}(w_{t+1}, t)$ to
    the distributed ledger.
\end{enumerate}
```

---

## Revision 4: Algorithm Section

**Find the algorithm section and check for pseudocode.**

If pseudocode is missing, ADD the following template (fill in actual parameter values):

```latex
\section{The our approach Protocol}
\label{sec:algorithm}

\subsection{Gradient Compression via Count-Sketch}

Each client compresses its gradient $g_i^t \in \mathbb{R}^d$ into a sketch
$\tilde{g}_i^t \in \mathbb{R}^k$ using a count-sketch with $k \ll d$.

\begin{algorithm}[t]
\caption{Client-side Count-Sketch (Round $t$)}
\label{alg:sketch}
\DontPrintSemicolon
\KwIn{Gradient $g_i^t \in \mathbb{R}^d$, sketch dimension $k$, hash functions $h_1, h_2$}
\KwOut{Sketch $\tilde{g}_i^t \in \mathbb{R}^k$}
Initialize $\tilde{g}_i^t \leftarrow \mathbf{0} \in \mathbb{R}^k$\;
\For{$j = 1$ \KwTo $d$}{
  $\tilde{g}_i^t[h_1(j)] \mathrel{+}= h_2(j) \cdot g_i^t[j]$\;
}
\Return $\tilde{g}_i^t$\;
\end{algorithm}

\subsection{Spectral Byzantine Filter}

The server constructs the sketch matrix $M_t \in \mathbb{R}^{n \times k}$
where row $i$ is the sketch $\tilde{g}_i^t$ of client $c_i$.

\begin{algorithm}[t]
\caption{Server-side Spectral Byzantine Filter (Round $t$)}
\label{alg:filter}
\DontPrintSemicolon
\KwIn{Sketch matrix $M_t \in \mathbb{R}^{n \times k}$, threshold $\tau$}
\KwOut{Trusted set $\mathcal{T}_t \subseteq \mathcal{C}$, suspect set $\mathcal{S}_t$}
Compute SVD: $M_t = U \Sigma V^\top$\;
Let $\lambda_1 \geq \lambda_2 \geq \ldots$ be singular values of $M_t$\;
Compute outlier score $s_i = \|M_t[i,:] - \Pi_k M_t[i,:]\|_2$ for each client $c_i$\;
% where \Pi_k projects onto the top-k principal components
% TODO: specify k and exact projection formula
$\mathcal{T}_t \leftarrow \{c_i : s_i \leq \tau\}$\;
$\mathcal{S}_t \leftarrow \mathcal{C} \setminus \mathcal{T}_t$\;
\Return $\mathcal{T}_t$, $\mathcal{S}_t$\;
\end{algorithm}

\noindent\textbf{Remark on threshold $\tau$.}
% TODO: specify how tau is set — fixed hyperparameter? Adaptive? Theoretically derived?
The threshold $\tau$ is set to [TODO: specify]. 
% If adaptive: "The threshold adapts to the empirical singular value gap..."
% If theoretical: "Theorem X derives the threshold from the spectral gap condition..."

\subsection{Robust Aggregation over Trusted Set}

\begin{algorithm}[t]
\caption{Server-side Robust Aggregation and Checkpoint (Round $t$)}
\label{alg:aggregate}
\DontPrintSemicolon
\KwIn{Trusted sketches $\{\tilde{g}_i^t\}_{i \in \mathcal{T}_t}$, step size $\eta$, current model $w_t$}
\KwOut{Updated model $w_{t+1}$, checkpoint $\mathrm{ckpt}_t$}
Recover approximate gradients: $\hat{g}_i^t \leftarrow \mathrm{CountSketchQuery}(\tilde{g}_i^t)$ for $i \in \mathcal{T}_t$\;
% TODO: specify aggregation rule:
$\bar{g}_t \leftarrow \mathrm{[RobustAggRule]}(\{\hat{g}_i^t\}_{i \in \mathcal{T}_t})$\;
% Options: geometric median, coordinate-wise trimmed mean, FedAvg if T is clean
$w_{t+1} \leftarrow w_t - \eta \bar{g}_t$\;
$\mathrm{ckpt}_t \leftarrow \mathrm{H}(w_{t+1} \| t \| |\mathcal{T}_t|)$\;
% H is a collision-resistant hash function
\textbf{Commit} $\mathrm{ckpt}_t$ to distributed ledger\;
\Return $w_{t+1}$\;
\end{algorithm}
```

---

## Revision 5: Stabilization Argument (ADD SECTION)

**ADD this section after the algorithm and before experiments. This is the SSS identity section.**

```latex
\section{Stabilization and Recovery Analysis}
\label{sec:analysis}

We analyze the fault-tolerance properties of our approach.
We adopt the \emph{practical stabilization} framework~\cite{Dolev2000}:
the protocol is practically stabilizing if, after any finite sequence of
Byzantine-contaminated rounds, it autonomously returns to a correct training
trajectory within bounded rounds, without external intervention.

\begin{definition}[Stable Training Trajectory]
\label{def:stable}
A training trajectory $\{w_t\}$ is $\varepsilon$-stable at round $t$ if
$\|w_t - w_t^*\| \leq \varepsilon$, where $w_t^*$ is the model that would
result from running the protocol with no Byzantine clients.
% TODO: formalize w_t^* — this requires specifying the reference protocol
\end{definition}

\begin{definition}[Recovery Time]
\label{def:recovery}
The recovery time $R$ of the protocol is the maximum number of rounds
required to return to an $\varepsilon$-stable trajectory after Byzantine
perturbation ceases, over all adversary strategies satisfying
Assumption~\ref{assum:byzantine}.
\end{definition}

\begin{lemma}[Sketch Fidelity]
\label{lem:sketch}
Under Assumption~\ref{assum:honest}, the count-sketch preserves the
directional structure of honest gradients: for any honest client $c_i$,
$\|\mathrm{CountSketchQuery}(\tilde{g}_i^t) - g_i^t\|_2 \leq \delta$
with probability $\geq 1 - \beta$, for parameters $(k, \delta, \beta)$
satisfying [TODO: condition on k, d, sigma].
\end{lemma}
\begin{proof}
% TODO: cite standard count-sketch accuracy bounds (Charikar et al. 2002)
Follows from standard count-sketch error bounds~\cite{Charikar2002}.
Full derivation in Appendix~\ref{app:sketch_proof}.
\end{proof}

\begin{lemma}[Spectral Separability]
\label{lem:spectral}
Under Assumptions~\ref{assum:byzantine} and~\ref{assum:honest}, and
assuming the spectral gap condition [TODO: state gap condition formally],
the spectral filter (Algorithm~\ref{alg:filter}) places all Byzantine
clients into the suspect set $\mathcal{S}_t$ with probability $\geq 1 - \gamma$:
$\mathcal{B} \subseteq \mathcal{S}_t$.
\end{lemma}
\begin{proof}
% TODO: formal proof or cite related spectral detection result
% Proof sketch: Byzantine updates form a low-rank perturbation of the
% honest gradient matrix; SVD identifies principal directions of deviation.
Proof sketch in Appendix~\ref{app:spectral_proof}. \textbf{[TODO: complete proof.]}
\end{proof}

\begin{lemma}[Aggregation Bound]
\label{lem:aggregation}
Conditioned on the event in Lemma~\ref{lem:spectral}, the robust aggregate
$\bar{g}_t$ over the trusted set $\mathcal{T}_t$ satisfies:
$\|\bar{g}_t - \bar{g}_t^*\|_2 \leq \varepsilon_{\mathrm{agg}}$,
where $\bar{g}_t^*$ is the mean over all honest clients.
\end{lemma}
\begin{proof}
% TODO: depends on choice of aggregation rule
Follows from standard properties of the [geometric median / trimmed mean]
aggregator~\cite{[cite]}. See Appendix~\ref{app:agg_proof}.
\end{proof}

\begin{theorem}[Practical Stabilization of our approach]
\label{thm:stabilization}
Under Assumptions~\ref{assum:byzantine} and~\ref{assum:honest},
let $f < f^*$ be the Byzantine fraction and let the spectral gap condition
of Lemma~\ref{lem:spectral} hold. Then our approach is practically
stabilizing with recovery time $R = O([TODO: express in terms of eta, epsilon,
spectral gap])$ and stable trajectory deviation $\varepsilon = [TODO]$.
\end{theorem}
\begin{proof}
% TODO: combine Lemmas 1–3 to prove the recovery bound
By Lemmas~\ref{lem:sketch}, \ref{lem:spectral}, and~\ref{lem:aggregation},
in each round with $|\mathcal{B}| \leq f < f^*$, the aggregate deviates
from the honest centroid by at most $\varepsilon_{\mathrm{agg}} + \delta$.
After Byzantine perturbation ceases, standard SGD convergence results imply
recovery within $R$ rounds. \textbf{[TODO: complete formal argument.
Empirical recovery time is measured in Section~\ref{sec:experiments},
Figure~\ref{fig:recovery}.]}
\end{proof}

\paragraph{Remark on classical vs.\ practical self-stabilization.}
% CHOOSE AND FILL IN:
% OPTION A (if classical proof exists): "Theorem~\ref{thm:stabilization} establishes
% classical self-stabilization in the sense of Dijkstra~\cite{Dijkstra1974}..."
% OPTION B (honest): "We note that Theorem~\ref{thm:stabilization} establishes a
% weaker property than classical Dijkstra self-stabilization~\cite{Dijkstra1974},
% which would require recovery from ANY starting configuration. We establish
% recovery from Byzantine-perturbed configurations under stated assumptions.
% Establishing full classical self-stabilization remains an open question."
```

---

## Revision 6: Related Work

**Find related work section. Apply these transformations:**

```bash
grep -n "\\\\section{Related" main.tex
```

**ADD if missing — SSS opening paragraph:**
```latex
\paragraph{Self-stabilizing distributed systems.}
Self-stabilization, introduced by Dijkstra~\cite{Dijkstra1974} and
formalized by Dolev~\cite{Dolev2000}, enables distributed systems to recover
from arbitrary transient faults without external intervention. Recent SSS
proceedings include work on self-stabilizing mutual exclusion in dynamic
networks~\cite{Devismes2025}, Byzantine-resilient distributed
broadcast~\cite{Lu2025,Amoussou2024}, and learning-augmented stabilization
of routing protocols~\cite{Chen2025}. our approach connects to this
tradition by providing a formally scoped recovery guarantee for a distributed
training protocol operating in an adversarial environment.
```

**ADD if missing — SSS Byzantine papers paragraph:**
```latex
\paragraph{Byzantine fault-tolerant distributed algorithms.}
Byzantine agreement and reliable broadcast under Byzantine faults have been
studied since~\cite{Lamport1982}. Recent SSS work includes Byzantine reliable
broadcast with monotonic counters~\cite{Amoussou2024}, cross-shard Byzantine
protection in blockchain sharding~\cite{Oglio2024}, and causal ordering under
Sybil Byzantine faults~\cite{Kshemkalyani2025}. Our work applies Byzantine
fault tolerance principles to the federated training protocol context.
```

**REVISE the FL related work paragraph to frame limitations in DS terms:**
```latex
% BEFORE (typical ML framing):
% "Krum [X] selects the gradient closest to its neighbors. FLTrust [Y] uses
%  a trusted dataset. These methods achieve high accuracy but..."

% AFTER (SSS framing):
Robust aggregation rules for federated learning include Krum~\cite{Blanchard2017},
which selects the gradient minimizing sum of distances to neighbors;
Trimmed Mean and Coordinate Median~\cite{Yin2018}, which apply coordinate-wise
statistics; Bulyan~\cite{ElMhamdi2018}; and FLTrust~\cite{Cao2020}, which
uses a trusted root dataset at the server. From a distributed-systems
perspective, these protocols share a common limitation: none provides an
explicit distributed recovery guarantee specifying the number of rounds
required to stabilize after Byzantine contamination, nor do they specify a
fault-containment invariant in the sense of~\cite{Dolev2000}.
our approach addresses this gap by providing a formally scoped
recovery argument alongside empirical evidence.
```

**ADD positioning paragraph at end of related work:**
```latex
\paragraph{Positioning.}
To the best of our knowledge, our approach is the first federated
learning protocol to combine: (i) gradient compression via count-sketch
for communication efficiency; (ii) spectral Byzantine detection with a
formally specified threshold; (iii) robust aggregation restricted to the
trusted client subset; and (iv) a distributed recovery argument scoped by
explicit assumptions on Byzantine fraction and honest gradient distribution.
Prior spectral detection approaches~\cite{[cite]} operate without the
sketching layer and without a formal recovery guarantee. Prior
Byzantine-robust FL protocols~\cite{Blanchard2017,Yin2018,Cao2020} lack
the spectral detection component and the recovery formalism. The combination
of these components under a unified distributed protocol specification is
the primary contribution of this work.
```

---

## Revision 7: Experiments Section

**Read current experiments section:**
```bash
grep -n "\\\\section{Exp\|\\\\section{Eval" main.tex
```

**Required additions if missing:**

```latex
% ADD: Recovery over rounds figure description
% (If the figure exists, improve the caption; if not, add a TODO)

% TODO: Add Figure: "Accuracy vs. Training Round" with attack injected at round T_attack.
% This is the critical SSS-specific visualization showing stabilization behavior.
% X-axis: training round (0 to T)
% Y-axis: test accuracy
% Lines: our approach, FedAvg, Krum, FLTrust
% Vertical line at T_attack marking attack onset
% Caption: "Recovery dynamics under [attack type] with f=[f] Byzantine clients.
%   our approach returns to within [X]% of unattacked accuracy within [R] rounds
%   of attack onset (marked ↑), while [baselines] fail to recover within [T] rounds."

% ADD: Communication overhead table
\begin{table}[t]
\caption{Per-round communication overhead. $d$: gradient dimension, $k$: sketch size.}
\label{tab:comm}
\centering
\begin{tabular}{lcc}
\hline
Method & Client$\to$Server & Overhead vs.\ FedAvg \\
\hline
FedAvg & $O(d)$ & 1$\times$ \\
Krum & $O(d)$ & 1$\times$ \\
our approach & $O(k)$, $k \ll d$ & $k/d$ \\
\hline
\end{tabular}
\end{table}

% ADD: Recovery time table
\begin{table}[t]
\caption{Recovery time $R$ (rounds to return to $\varepsilon$-stable trajectory)
  under three attack types with $f/n = 0.3$.}
\label{tab:recovery}
\centering
\begin{tabular}{lccc}
\hline
Method & Sign-flip & Label-flip & Scaling \\
\hline
our approach & [R] & [R] & [R] \\
Krum & [R or DNR] & & \\
FLTrust & & & \\
\hline
\multicolumn{4}{l}{DNR = Did Not Recover within $T=100$ rounds.}
\end{tabular}
\end{table}
% TODO: Fill in actual values from experiments.
```

**Improve ALL figure captions in the paper:**
For each `\caption{...}` in the experiments section, verify it explains the research
insight. If it only describes axes, expand it to include what the figure proves.

---

## Revision 8: Conclusion

**Rewrite conclusion to end on distributed-systems terms:**

```latex
\section{Conclusion}
\label{sec:conclusion}

We presented our approach, a Byzantine-robust federated learning protocol
that provides [practical stabilization / fault containment] guarantees under a
Byzantine fraction $f < f^*$ and distributional assumptions on honest client
gradients. The protocol combines gradient compression via count-sketch,
spectral Byzantine filtering, robust aggregation over the trusted client subset,
and verifiable distributed checkpointing into a unified protocol pipeline with
an explicit round structure.

[Practically stabilizing / Fault-containing] behavior is demonstrated empirically
across $[N]$ attack types: the protocol consistently returns to within $\varepsilon$
of the unattacked training trajectory within $R$ rounds of attack onset.
Closing the gap between the empirical recovery evidence and a fully formal
convergence proof under general distributional assumptions remains an important
open problem.

Future work includes: (i) extending the protocol to asynchronous round models;
(ii) proving formal self-stabilization in the classical Dijkstra sense under
stronger distributional assumptions; (iii) analyzing adaptive adversaries that
craft updates to evade the spectral filter; and (iv) scaling the checkpoint
mechanism to large-N deployments.
```

---

## Final Compile Check

```bash
# After all revisions, compile and check
latexmk -pdf main.tex 2>&1 | grep -E "Error|Warning|Overfull|Underfull" | head -40

# Check for undefined references
grep "LaTeX Warning: Reference" main.log

# Check for undefined citations  
grep "LaTeX Warning: Citation" main.log

# Check page count
pdfinfo main.pdf | grep Pages
# Must be <= 15 pages (excluding references)

# If over 15 pages, identify where to trim:
# 1. Algorithm pseudocode — check for redundancy
# 2. Experiments — move some plots to appendix
# 3. Proofs — move to appendix (mark "proof in appendix")
# Do NOT trim: system model, adversary model, stabilization argument, contributions
```

---

## Agent Constraints (NEVER VIOLATE)

1. **Never invent experimental results.** If a number is missing, write `TODO: [description]`
2. **Never fabricate citations.** If a citation is needed, write `\cite{TODO:authorYEAR}` with a comment describing the needed paper
3. **Never delete technical content.** Comment it out: `% REMOVED: [reason] — [original text]`
4. **Never change the paper's core technical claims** — only the framing and presentation
5. **Never add formal theorem statements that cannot be proven** — use "empirically supported" or "conjecture" if needed
6. **Always maintain LNCS format compliance** — do not change margins, fonts, or sectioning structure
7. **Mark every uncertain edit** with `% REVISION NOTE: [what was changed and why]`

---

*End of SSS_REVISION_AGENT.md*
