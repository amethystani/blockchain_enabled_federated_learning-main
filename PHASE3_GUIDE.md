# Phase 3 Guide: Scaled-Down Real-World Experiments

Complete guide for running Phase 3 experiments with all three deployment scales.

## Overview

Phase 3 implements scaled-down versions of the paper's real-world experiments:
- **Phase 3A**: Medium-Scale (ResNet-50, ~25M params, FEMNIST)
- **Phase 3B**: Large-Scale (ViT-Small, ~22M params, Tiny ImageNet)
- **Phase 3C**: Foundation Models (GPT-2-Medium, ~345M params, placeholder)

## Hardware Requirements

| Phase | VRAM | RAM | Runtime |
|-------|------|-----|---------|
| 3A | 8-16GB | 16GB+ | 2-4 hours |
| 3B | 8-16GB | 16GB+ | 3-6 hours |
| 3C | 12-16GB | 24GB+ | Manual setup |

## Phase 3A: Medium-Scale

### Quick Start

```bash
# Single aggregator test
python spectral_sentinel/experiments/medium_scale_experiment.py \
  --aggregator spectral_sentinel \
  --num_clients 50 \
  --byzantine_ratio 0.4 \
  --num_rounds 30

# Compare all aggregators
python spectral_sentinel/experiments/medium_scale_experiment.py \
  --compare_all \
  --num_rounds 20
```

### What It Tests
- **Model**: ResNet-50 (~25M parameters)
- **Dataset**: FEMNIST (62 classes, 50 clients)
- **Attack**: Min-max attack (40% Byzantine)
- **Comparison**: Spectral Sentinel vs FLTrust, FLAME, Bulyan, Krum, FedAvg

### Expected Results
| Aggregator | Target Accuracy |
|------------|----------------|
| Spectral Sentinel | 75-80% |
| FLTrust | 60-70% |
| FLAME | 60-70% |
| Bulyan | 55-65% |
| Krum | 50-60% |
| FedAvg | 20-30% |

### Memory Usage
- Full covariance: ~10GB (estimated)
- Sketched (k=256): ~260MB ✓

## Phase 3B: Large-Scale

### Quick Start

```bash
python spectral_sentinel/experiments/large_scale_experiment.py \
  --aggregator spectral_sentinel \
  --num_clients 32 \
  --byzantine_ratio 0.3 \
  --num_rounds 25
```

### What It Tests
- **Model**: ViT-Small (~22M parameters)
- **Dataset**: Tiny ImageNet (CIFAR-100 placeholder, 100 classes)
- **Attack**: ALIE attack (30% Byzantine)
- **Sketching**: Enabled (k=256)

### Expected Results
- Target accuracy: 65-75%
- Detection recall: >85%
- Memory: ~260MB (vs ~8GB full)

## Phase 3C: Foundation Models

### Status
⚠️ **Placeholder implementation** - requires manual setup

### Requirements
```bash
pip install transformers datasets
```

### Manual Setup Steps
1. Load pretrained GPT-2-Medium from HuggingFace
2. Create text dataset (WikiText-103 or similar)
3. Implement tokenization pipeline
4. Add perplexity metric
5. Integrate with federated learning framework

See `foundation_model_experiment.py` for details.

## Installation

### Basic Requirements
```bash
pip install -r requirements_spectral.txt
```

### Phase 3 Additional Requirements
```bash
# Already installed with base requirements:
# - torch >= 2.0
# - torchvision
# - numpy, scipy

# Optional for Phase 3C:
pip install transformers datasets  # For GPT-2
```

## Running All Phases

### Sequential Execution
```bash
# Phase 3A (2-4 hours)
python spectral_sentinel/experiments/medium_scale_experiment.py --compare_all

# Phase 3B (3-6 hours)
python spectral_sentinel/experiments/large_scale_experiment.py

# Phase 3C (manual)
python spectral_sentinel/experiments/foundation_model_experiment.py
```

### Quick Validation (Reduced Rounds)
```bash
# Fast testing (20 min total)
python spectral_sentinel/experiments/medium_scale_experiment.py --num_rounds 10
python spectral_sentinel/experiments/large_scale_experiment.py --num_rounds 10
```

## Results Structure

After running experiments:
```
results/
├── phase3a_medium_scale/
│   ├── spectral_sentinel/
│   ├── fltrust/
│   ├── flame/
│   └── comparison.png
├── phase3b_large_scale/
│   └── spectral_sentinel/
└── phase3c_foundation/
    └── (manual)
```

## Key Features Demonstrated

### Phase 3A
- [x] Scaled-down medium-scale model (ResNet-50)
- [x] FEMNIST natural heterogeneity
- [x] TV distance computation
- [x] FLTrust and FLAME baselines
- [x] Memory-efficient sketching

### Phase 3B
- [x] Vision Transformer architecture
- [x] Larger image size (64x64)
- [x] More sophisticated attack (ALIE)
- [x] Sketching validation

### Phase 3C
- [x] Foundation model architecture (GPT-2-Medium)
- [ ] Language modeling task
- [ ] Perplexity metric
- [ ] Layer-wise sketching for decoders

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
--batch_size 8

# Use CPU (slow but works)
--device cpu

# Reduce clients
--num_clients 20
```

### FEMNIST Download Issues
EMNIST dataset downloads automatically. If it fails:
- Check internet connection
- Clear `./data` directory
- Manually download from torchvision

### Slow Training
- Reduce `--num_rounds`
- Use fewer clients
- Enable GPU if available

## Next Steps

After Phase 3:
1. Run full Phase 2 validation to compare with Phase 1 baselines
2. Generate final comparison plots
3. Write up results
4. (Optional) Implement remaining features:
   - Geo-distributed simulation
   - Combined attacks
   - Full foundation model integration

## Citation

If using this implementation:
```bibtex
@article{spectral_sentinel_2025,
  title={Spectral Sentinel: Byzantine-Robust Federated Learning via Random Matrix Theory},
  note={Phase 3 scaled-down implementation}
}
```

---

**Status**: Phase 3A ✅ Ready | Phase 3B ✅ Ready | Phase 3C ⚠️ Manual Setup Required
