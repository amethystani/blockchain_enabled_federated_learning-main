# 🚀 QUICK START GUIDE

## Super Quick Start (5 minutes)

### Option 1: Interactive Menu
```bash
python app.py
```
Then select `[1] Quick Validation` from the menu.

### Option 2: Direct Command
```bash
python app.py --quick
```

### Option 3: Direct Script
```bash
python spectral_sentinel/experiments/quick_validation.py
```

---

## What You'll See

The quick validation will:
1. Load MNIST dataset (10 clients)
2. Train ResNet-18 for 10 rounds
3. Test against Byzantine attacks (30% ratio)
4. Compare Spectral Sentinel vs baselines

**Expected Output**:
```
✓ Spectral Sentinel:  ~85% accuracy
✓ Detection Rate:     >90%
✓ Baseline (FedAvg):  ~30% accuracy
```

**Runtime**: ~5 minutes on CPU, ~2 minutes on GPU

---

## Running Specific Experiments

### Using app.py (Recommended)
```bash
# Interactive menu
python app.py

# Quick validation
python app.py --quick

# Complete benchmark
python app.py --benchmark

# Run everything
python app.py --all
```

### Using Scripts Directly

```bash
# Phase 2: Validation
python spectral_sentinel/experiments/quick_validation.py
python spectral_sentinel/experiments/compare_aggregators.py
python spectral_sentinel/experiments/test_all_attacks.py

# Phase 3: Scaled experiments
python spectral_sentinel/experiments/medium_scale_experiment.py

# Phase 4: Advanced analysis
python spectral_sentinel/experiments/game_theoretic_experiment.py
python spectral_sentinel/experiments/ablation_studies.py

# Phase 5: Benchmarks
python spectral_sentinel/experiments/complete_benchmark.py
python spectral_sentinel/experiments/limitations_analysis.py
```

---

## Run Everything (8-12 hours)

### Option 1: Using app.py
```bash
python app.py --all
```

### Option 2: Using bash script
```bash
chmod +x run_all.sh
./run_all.sh
```

---

## Docker Deployment

### Quick Test
```bash
# Build
docker build -t spectral_sentinel .

# Run quick validation
docker run --gpus all spectral_sentinel \
    python3 spectral_sentinel/experiments/quick_validation.py
```

### Multi-Node Simulation
```bash
# Start 4 nodes
docker-compose up

# Scale to 16 nodes
docker-compose up --scale worker=16
```

---

## Common Commands

```bash
# Check GPU
python -c "from spectral_sentinel.utils.multi_gpu import print_gpu_info; print_gpu_info()"

# Quick demo (10 clients, 5 rounds)
python -c "from spectral_sentinel.experiments.quick_validation import run_quick_experiment; run_quick_experiment(num_clients=10, num_rounds=5)"

# Generate report
python app.py
# Then select [21] Generate Report
```

---

## Results Location

All results are saved to:
```
results/
├── phase2_validation/
├── phase3a_medium_scale/
├── phase4_game_theory/
└── phase5_benchmark/
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size or use CPU
python spectral_sentinel/experiments/quick_validation.py --device cpu
```

### Missing Dependencies
```bash
pip install -r requirements_spectral.txt
pip install -r requirements_phase3.txt
```

### GPU Not Found
```bash
# Run on CPU (slower but works)
export CUDA_VISIBLE_DEVICES=""
python app.py --quick
```

---

## Next Steps After Running

1. **Check Results**: Look in `./results/` directory
2. **Review Plots**: PNG files show accuracy curves
3. **Read CSV**: Detailed metrics in CSV files
4. **Compare**: Match results against WHATWEHAVETOIMPLEMENT.MD

---

## Full Documentation

- **SPECTRAL_SENTINEL_COMPLETE.md**: Complete feature list
- **PHASE3_GUIDE.md**: Scaled experiments guide
- **walkthrough.md**: Implementation details
- **WHATWEHAVETOIMPLEMENT.MD**: Original requirements

---

**TL;DR**: Just run `python app.py` and select what you want! ✨
