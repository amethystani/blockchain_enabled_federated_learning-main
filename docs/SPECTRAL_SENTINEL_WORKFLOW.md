# Spectral Sentinel: Complete Workflow Diagram

## Full Algorithm Workflow

```mermaid
flowchart TB
    Start([Start Federated Learning]) --> Config[Configure System<br/>Dataset, Clients, Attack Type]
    
    Config --> DataLoad[Load & Partition Data<br/>Non-IID Dirichlet Distribution]
    DataLoad --> ModelInit[Initialize Global Model<br/>CNN/ResNet/LeNet5]
    
    ModelInit --> CreateClients{Create Clients}
    CreateClients --> HonestClients[Honest Clients<br/>Train with clean data]
    CreateClients --> ByzantineClients[Byzantine Clients<br/>With attack strategy]
    
    HonestClients --> ClientPool[Client Pool]
    ByzantineClients --> ClientPool
    
    ClientPool --> InitServer[Initialize Federated Server<br/>+ Spectral Sentinel Aggregator]
    
    InitServer --> RoundStart{Start Round r}
    
    RoundStart --> SelectClients[Select Participating Clients]
    SelectClients --> BroadcastModel[Broadcast Global Model<br/>to Selected Clients]
    
    BroadcastModel --> ParallelTrain[/Parallel Local Training/]
    
    ParallelTrain --> HonestTrain[Honest Clients:<br/>Local SGD Training<br/>E epochs on local data]
    ParallelTrain --> ByzantineTrain[Byzantine Clients:<br/>Apply Attack<br/>MinMax/ALIE/LabelFlip/etc]
    
    HonestTrain --> CollectGrad[Server Collects Gradients<br/>G = {g₁, g₂, ..., gₙ}]
    ByzantineTrain --> CollectGrad
    
    CollectGrad --> SpectralSentinel[SPECTRAL SENTINEL AGGREGATOR]
    
    SpectralSentinel --> GradMatrix[1. Convert to Matrix<br/>X ∈ ℝⁿˣᵈ<br/>n=clients, d=parameters]
    
    GradMatrix --> Sketching{Use Sketching?}
    Sketching -->|Yes, Large Model| FreqDir[Apply Frequent Directions<br/>Reduce to X̃ ∈ ℝⁿˣᵏ<br/>k << d, O(k²) memory]
    Sketching -->|No, Small Model| FullMatrix[Use Full Matrix X]
    
    FreqDir --> Covariance[2. Compute Covariance<br/>Σ = XᵀX / n]
    FullMatrix --> Covariance
    
    Covariance --> Eigendecomp[3. Eigendecomposition<br/>λ₁ ≥ λ₂ ≥ ... ≥ λₙ]
    
    Eigendecomp --> RMTAnalysis[4. RMT Analysis]
    
    RMTAnalysis --> FitMP[Fit Marchenko-Pastur Law<br/>ρ(λ) with γ = n/d]
    FitMP --> ComputeDensity[Compute Spectral Density<br/>Empirical vs Theoretical]
    
    ComputeDensity --> KSTest[5. KS Test<br/>H₀: λ follows MP law<br/>D_KS = sup|F_emp - F_MP|]
    
    KSTest --> KSDecision{p-value < α?}
    KSDecision -->|No| NoAnomaly[No Byzantine Detected<br/>via KS Test]
    KSDecision -->|Yes| AnomalyDetected[Anomaly Detected!]
    
    AnomalyDetected --> TailAnalysis[6. Tail Anomaly Detection<br/>Check eigenvalues > λ_max]
    NoAnomaly --> TailAnalysis
    
    TailAnalysis --> TailDecision{Tail eigenvalues<br>> threshold?}
    TailDecision -->|Yes| IdentifyByz[Identify Byzantine Clients]
    TailDecision -->|No| AllHonest[All Clients Honest]
    
    IdentifyByz --> EigenvectorProj[Project Gradients onto<br/>Top-k Eigenvectors<br/>anomalous directions]
    
    EigenvectorProj --> RankClients[Rank Clients by<br/>Projection Magnitude]
    
    RankClients --> FlagByzantine[Flag Clients with<br/>Largest Projections<br/>as Byzantine]
    
    FlagByzantine --> FilterGrad[7. Filter Byzantine Gradients<br/>Keep only honest G_honest]
    AllHonest --> FilterGrad
    
    FilterGrad --> AggregateHonest[8. Aggregate Honest Gradients<br/>ḡ = (1/|G_honest|) Σ g_i]
    
    AggregateHonest --> UpdateGlobal[9. Update Global Model<br/>θ ← θ - η · ḡ]
    
    UpdateGlobal --> Evaluate[Evaluate on Test Set<br/>Accuracy & Loss]
    
    Evaluate --> Stats[Update Statistics:<br/>- Detection Rate<br/>- Accuracy<br/>- KS Statistics<br/>- Phase Transition Metric]
    
    Stats --> CheckRounds{Round r < R?}
    CheckRounds -->|Yes| RoundStart
    CheckRounds -->|No| Visualize
    
    Visualize[Generate Visualizations:<br/>- Training Curves<br/>- Detection Metrics<br/>- Spectral Analysis]
    
    Visualize --> SaveModel[Save Final Model & Results]
    SaveModel --> End([End])
    
    style SpectralSentinel fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    style RMTAnalysis fill:#4ecdc4,stroke:#0a7c6f,stroke-width:2px
    style KSTest fill:#ffe66d,stroke:#d4a574,stroke-width:2px
    style IdentifyByz fill:#ff6b6b,stroke:#c92a2a,stroke-width:2px
    style AggregateHonest fill:#51cf66,stroke:#2f9e44,stroke-width:2px
```

## Detailed Byzantine Detection Pipeline

```mermaid
flowchart LR
    subgraph Input
        G[Gradients<br/>g₁...gₙ]
    end
    
    subgraph "Matrix Construction"
        M[Matrix X<br/>n × d]
        G --> M
    end
    
    subgraph "RMT Analysis"
        E[Eigenvalues<br/>λ₁...λₙ]
        MP[Marchenko-Pastur<br/>Distribution]
        M --> E
        E --> MP
    end
    
    subgraph "Detection Tests"
        KS[KS Test<br/>D, p-value]
        Tail[Tail Anomaly<br/>λ > λ_max]
        MP --> KS
        E --> Tail
    end
    
    subgraph "Client Identification"
        Proj[Eigenvector<br/>Projection]
        Rank[Ranking &<br/>Thresholding]
        KS --> Proj
        Tail --> Proj
        Proj --> Rank
    end
    
    subgraph Output
        Honest[Honest<br/>Clients]
        Byz[Byzantine<br/>Clients]
        Rank --> Honest
        Rank --> Byz
    end
    
    style "RMT Analysis" fill:#e3f2fd
    style "Detection Tests" fill:#fff3e0
    style "Client Identification" fill:#fce4ec
```

## Component Architecture

```mermaid
graph TB
    subgraph "Experiments Layer"
        Exp[simulate_basic.py<br/>Main Entry Point]
    end
    
    subgraph "Federated Layer"
        Server[FederatedServer<br/>Orchestration]
        HClient[HonestClient<br/>Clean Training]
        BClient[ByzantineClient<br/>Attack Execution]
        DataLoader[DataLoader<br/>Non-IID Partition]
    end
    
    subgraph "Aggregation Layer"
        SS[SpectralSentinel<br/>Main Aggregator]
        Baselines[Baselines<br/>FedAvg, Krum, etc]
    end
    
    subgraph "RMT Core"
        SA[SpectralAnalyzer<br/>Byzantine Detection]
        MP_Law[MarchenkoPastur<br/>Theoretical Distribution]
        SD[SpectralDensity<br/>Empirical Analysis]
    end
    
    subgraph "Sketching Layer"
        FD[FrequentDirections<br/>O(k²) Memory]
        LW[LayerWiseSketch<br/>For Transformers]
    end
    
    subgraph "Attack Layer"
        Attacks[8 Attack Types:<br/>MinMax, ALIE, LabelFlip<br/>Adaptive, SignFlip, etc]
    end
    
    subgraph "Utils"
        Models[Neural Network<br/>Models]
        Metrics[Visualization<br/>& Metrics]
    end
    
    Exp --> Server
    Server --> HClient
    Server --> BClient
    Server --> DataLoader
    Server --> SS
    Server --> Baselines
    
    BClient --> Attacks
    
    SS --> SA
    SS --> FD
    
    SA --> MP_Law
    SA --> SD
    SA --> FD
    
    FD --> LW
    
    HClient --> Models
    Exp --> Metrics
    
    style SS fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    style SA fill:#4ecdc4,stroke:#0a7c6f,stroke-width:2px
    style Attacks fill:#ffd43b,stroke:#fab005,stroke-width:2px
```

## Key Algorithms

### Marchenko-Pastur Law
For n clients, d parameters, aspect ratio γ = n/d:

```
ρ_MP(λ) = (1/(2πσ²λ)) √[(λ_max - λ)(λ - λ_min)]

where:
  λ_min = σ²(1 - √γ)²
  λ_max = σ²(1 + √γ)²
```

### Byzantine Detection Criterion
1. **KS Test**: `D_KS = sup |F_empirical(λ) - F_MP(λ)| < threshold`
2. **Tail Test**: `#{λ_i > λ_max} / n < tail_threshold`
3. **Phase Transition**: `σ²f² < 0.25` for reliable detection

### Aggregation
```
G_honest = {g_i : i ∉ Byzantine_detected}
θ_new = θ_old - η · (1/|G_honest|) Σ_{i ∈ G_honest} g_i
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Detection Rate** | ~95% @ 40% Byzantine |
| **Memory Complexity** | O(k²) with sketching vs O(d²) |
| **Time Complexity** | O(nk²) per round |
| **Scalability** | Up to 1.5B parameters |
| **Robustness** | Works under Non-IID data |

## Attack Types Handled

1. **Min-Max**: Maximize/minimize loss
2. **ALIE**: Inner product manipulation
3. **Label Flipping**: Flip training labels
4. **Adaptive Spectral**: Attack aware of defense
5. **Sign Flip**: Reverse gradient direction
6. **Zero Gradient**: Send zeros
7. **Gaussian Noise**: Add random noise
8. **Model Poisoning**: Corrupt model weights

---

**Legend:**
- 🟥 Red: Byzantine detection & filtering
- 🟦 Blue: RMT analysis
- 🟨 Yellow: Statistical tests
- 🟩 Green: Aggregation & model update
