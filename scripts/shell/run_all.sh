#!/bin/bash
# Run all Spectral Sentinel experiments sequentially
# Estimated time: 8-12 hours

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        SPECTRAL SENTINEL - RUN ALL EXPERIMENTS                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  This will run ALL experiments sequentially"
echo "⏱️  Estimated time: 8-12 hours"
echo ""

read -p "Continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "🚀 Starting all experiments..."
echo ""

# Create results directory
mkdir -p results

# Phase 2: Validation (1-2 hours)
echo "═══════════════════════════════════════════════════"
echo "PHASE 2: VALIDATION SUITE"
echo "═══════════════════════════════════════════════════"

echo "→ Quick Validation (5 min)"
python spectral_sentinel/experiments/quick_validation.py

echo ""
echo "→ Aggregator Comparison (30 min)"
python spectral_sentinel/experiments/compare_aggregators.py

echo ""
echo "→ All Attacks Test (45 min)"
python spectral_sentinel/experiments/test_all_attacks.py

echo ""
echo "→ Phase Transition Validation (20 min)"
python spectral_sentinel/experiments/validate_phase_transition.py

echo ""
echo "✓ Phase 2 Complete!"
echo ""

# Phase 3: Scaled Experiments (6-10 hours)
echo "═══════════════════════════════════════════════════"
echo "PHASE 3: SCALED EXPERIMENTS"
echo "═══════════════════════════════════════════════════"

echo "→ Medium-Scale (ResNet-50 + FEMNIST, 3 hours)"
python spectral_sentinel/experiments/medium_scale_experiment.py

echo ""
echo "→ Large-Scale (ViT-Small + Tiny ImageNet, 4 hours)"
python spectral_sentinel/experiments/large_scale_experiment.py

echo ""
echo "✓ Phase 3 Complete!"
echo ""

# Phase 4: Advanced Analysis (3-4 hours)
echo "═══════════════════════════════════════════════════"
echo "PHASE 4: ADVANCED ANALYSIS"
echo "═══════════════════════════════════════════════════"

echo "→ Game-Theoretic Analysis (30 min)"
python spectral_sentinel/experiments/game_theoretic_experiment.py

echo ""
echo "→ Ablation Studies (1 hour)"
python spectral_sentinel/experiments/ablation_studies.py

echo ""
echo "→ Certified Defense Comparison (1 hour)"
python spectral_sentinel/experiments/certified_defense_comparison.py

echo ""
echo "✓ Phase 4 Complete!"
echo ""

# Phase 5: Benchmarks (2-3 hours)
echo "═══════════════════════════════════════════════════"
echo "PHASE 5: BENCHMARKS & LIMITATIONS"
echo "═══════════════════════════════════════════════════"

echo "→ Complete Benchmark (12×11, 2 hours)"
python spectral_sentinel/experiments/complete_benchmark.py

echo ""
echo "→ Limitations Analysis (30 min)"
python spectral_sentinel/experiments/limitations_analysis.py

echo ""
echo "✓ Phase 5 Complete!"
echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  ALL EXPERIMENTS COMPLETE! 🎉                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: ./results/"
echo ""
echo "Key directories:"
echo "  • results/phase2_validation/"
echo "  • results/phase3a_medium_scale/"
echo "  • results/phase4_game_theory/"
echo "  • results/phase5_benchmark/"
echo ""
echo "Next steps:"
echo "  1. Review results in ./results/"
echo "  2. Check plots and CSV files"
echo "  3. Compare with paper claims (docs/WHATWEHAVETOIMPLEMENT.MD)"
echo ""
echo "For detailed analysis, see:"
echo "  • docs/SPECTRAL_SENTINEL_COMPLETE.md"
echo "  • docs/SPECTRAL_SENTINEL_README.md"
echo ""
echo "✨ Done! ✨"
