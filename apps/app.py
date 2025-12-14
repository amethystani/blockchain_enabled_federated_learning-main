#!/usr/bin/env python3
"""
Spectral Sentinel - Main Application

Interactive CLI to run all experiments and features.
Run this to access the complete Spectral Sentinel system.

Usage:
    python apps/app.py                    # Interactive menu
    python apps/app.py --quick            # Quick validation
    python apps/app.py --benchmark        # Full benchmark
    python apps/app.py --all              # Run everything
"""

import sys
import os
import argparse
from pathlib import Path

# Add repo root to path (so `spectral_sentinel` is importable even from `apps/`)
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

def print_banner():
    """Print welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🛡️  SPECTRAL SENTINEL 🛡️                        ║
    ║                                                               ║
    ║     Byzantine-Robust Federated Learning via RMT               ║
    ║                                                               ║
    ║     12 Attacks | 11 Aggregators | 100% Complete              ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """Print main menu."""
    menu = """
    ╔═════════════════════════════════════════════════════════════╗
    ║                        MAIN MENU                            ║
    ╠═════════════════════════════════════════════════════════════╣
    ║                                                             ║
    ║  QUICK START (5-10 minutes)                                ║
    ║  ─────────────────────────                                  ║
    ║  [1]  Quick Validation          (Sanity check, 5 min)      ║
    ║  [2]  Demo Experiment           (Mini demo, 10 min)        ║
    ║                                                             ║
    ║  PHASE 2: VALIDATION (1-2 hours)                           ║
    ║  ─────────────────────────────                              ║
    ║  [3]  Aggregator Comparison     (9 aggregators)            ║
    ║  [4]  All Attacks Test          (10 attacks)               ║
    ║  [5]  Phase Transition          (Validate σ²f²)            ║
    ║  [6]  Run All Phase 2           (Complete validation)      ║
    ║                                                             ║
    ║  PHASE 3: SCALED EXPERIMENTS (3-6 hours each)              ║
    ║  ─────────────────────────────────────────                  ║
    ║  [7]  Medium-Scale              (ResNet-50 + FEMNIST)      ║
    ║  [8]  Large-Scale               (ViT-Small + TinyImageNet) ║
    ║  [9]  Foundation Model          (GPT-2-Medium)             ║
    ║                                                             ║
    ║  PHASE 4: ADVANCED ANALYSIS (1-2 hours each)               ║
    ║  ──────────────────────────────────────────                 ║
    ║  [10] Game-Theoretic Analysis   (Nash equilibrium)         ║
    ║  [11] Ablation Studies          (4 design choices)         ║
    ║  [12] Certified Defenses        (CRFL, ByzShield)          ║
    ║                                                             ║
    ║  PHASE 5: BENCHMARKS & LIMITS (2-3 hours each)             ║
    ║  ────────────────────────────────────────────               ║
    ║  [13] Complete Benchmark        (12×11 evaluation)         ║
    ║  [14] Limitations Analysis      (5 theoretical tests)      ║
    ║                                                             ║
    ║  DEPLOYMENT                                                 ║
    ║  ──────────                                                 ║
    ║  [15] Docker Build              (Build container)          ║
    ║  [16] Docker Run                (Run in container)         ║
    ║  [17] Multi-GPU Info            (Check GPU setup)          ║
    ║                                                             ║
    ║  SPECIAL                                                    ║
    ║  ───────                                                    ║
    ║  [20] Run Everything            (All experiments)          ║
    ║  [21] Generate Report           (Summary of results)       ║
    ║                                                             ║
    ║  [0]  Exit                                                  ║
    ║                                                             ║
    ╚═════════════════════════════════════════════════════════════╝
    """
    print(menu)


def run_quick_validation():
    """Run quick validation."""
    print("\n🚀 Running Quick Validation (5 minutes)...")
    print("="*70)
    os.system("python3 spectral_sentinel/experiments/quick_validation.py")


def run_demo():
    """Run demo experiment."""
    print("\n🎯 Running Demo Experiment...")
    print("="*70)
    # Quick MNIST experiment with 10 clients
    cmd = 'python3 -c "from spectral_sentinel.experiments.quick_validation import run_quick_experiment; run_quick_experiment(num_clients=10, num_rounds=5, dataset=\'mnist\')"'
    os.system(cmd)


def run_aggregator_comparison():
    """Run aggregator comparison."""
    print("\n📊 Comparing Aggregators...")
    os.system("python spectral_sentinel/experiments/compare_aggregators.py")


def run_all_attacks():
    """Test all attacks."""
    print("\n⚔️  Testing All Attacks...")
    os.system("python spectral_sentinel/experiments/test_all_attacks.py")


def run_phase_transition():
    """Validate phase transition."""
    print("\n🔬 Validating Phase Transition...")
    os.system("python spectral_sentinel/experiments/validate_phase_transition.py")


def run_all_phase2():
    """Run all Phase 2 validation."""
    print("\n🏃 Running Complete Phase 2 Validation...")
    os.system("bash spectral_sentinel/experiments/validation/run_phase2_validation.sh")


def run_medium_scale():
    """Run medium-scale experiment."""
    print("\n📈 Medium-Scale Experiment (ResNet-50 + FEMNIST)...")
    os.system("python spectral_sentinel/experiments/medium_scale_experiment.py")


def run_large_scale():
    """Run large-scale experiment."""
    print("\n🚀 Large-Scale Experiment (ViT-Small + Tiny ImageNet)...")
    os.system("python spectral_sentinel/experiments/large_scale_experiment.py")


def run_foundation():
    """Run foundation model experiment."""
    print("\n🤖 Foundation Model Experiment (GPT-2-Medium)...")
    os.system("python spectral_sentinel/experiments/foundation_model_experiment.py")


def run_game_theory():
    """Run game-theoretic analysis."""
    print("\n🎮 Game-Theoretic Analysis...")
    os.system("python spectral_sentinel/experiments/game_theoretic_experiment.py")


def run_ablations():
    """Run ablation studies."""
    print("\n🔬 Ablation Studies...")
    os.system("python spectral_sentinel/experiments/ablation_studies.py")


def run_certified():
    """Run certified defense comparison."""
    print("\n🛡️  Certified Defense Comparison...")
    os.system("python spectral_sentinel/experiments/certified_defense_comparison.py")


def run_benchmark():
    """Run complete benchmark."""
    print("\n⚡ Complete 12×11 Benchmark...")
    os.system("python spectral_sentinel/experiments/complete_benchmark.py")


def run_limitations():
    """Run limitations analysis."""
    print("\n⚠️  Limitations Analysis...")
    os.system("python spectral_sentinel/experiments/limitations_analysis.py")


def docker_build():
    """Build Docker image."""
    print("\n🐳 Building Docker Image...")
    os.system("docker build -t spectral_sentinel .")


def docker_run():
    """Run in Docker."""
    print("\n🐳 Running in Docker...")
    os.system("docker run --gpus all spectral_sentinel python3 spectral_sentinel/experiments/quick_validation.py")


def check_gpu():
    """Check GPU setup."""
    print("\n🖥️  GPU Information...")
    os.system("python -c 'from spectral_sentinel.utils.multi_gpu import print_gpu_info; print_gpu_info()'")


def run_everything():
    """Run all experiments sequentially."""
    print("\n🚀 RUNNING ALL EXPERIMENTS")
    print("="*70)
    print("⚠️  This will take 8-12 hours total!")
    print("="*70)
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    experiments = [
        ("Quick Validation", run_quick_validation),
        ("All Attacks", run_all_attacks),
        ("Aggregator Comparison", run_aggregator_comparison),
        ("Phase Transition", run_phase_transition),
        ("Game Theory", run_game_theory),
        ("Ablations", run_ablations),
        ("Certified Defenses", run_certified),
        ("Limitations", run_limitations),
        ("Complete Benchmark", run_benchmark),
    ]
    
    for name, func in experiments:
        print(f"\n\n{'='*70}")
        print(f"Starting: {name}")
        print(f"{'='*70}")
        func()
        print(f"\n✓ Completed: {name}")
    
    print("\n\n🎉 ALL EXPERIMENTS COMPLETE!")
    generate_report()


def generate_report():
    """Generate summary report."""
    print("\n📄 Generating Summary Report...")
    
    report = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║              SPECTRAL SENTINEL - EXECUTION REPORT             ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Experiments Completed:
    • Quick Validation
    • Phase 2 Validation Suite
    • Phase 3 Scaled Experiments
    • Phase 4 Advanced Analysis
    • Phase 5 Benchmarks & Limitations
    
    Results saved to: ./results/
    
    Key Files:
    • results/phase2_validation/
    • results/phase3a_medium_scale/
    • results/phase4_game_theory/
    • results/phase5_benchmark/
    
    Next Steps:
    1. Review results in ./results/
    2. Check plots and CSV files
    3. Compare with paper claims
    
    For detailed analysis, see:
    • docs/SPECTRAL_SENTINEL_COMPLETE.md
    • docs/SPECTRAL_SENTINEL_README.md
    """
    
    print(report)
    
    # Save report
    with open('EXECUTION_REPORT.txt', 'w') as f:
        f.write(report)
    
    print("✓ Report saved to: EXECUTION_REPORT.txt")


def main():
    """Main application loop."""
    parser = argparse.ArgumentParser(description='Spectral Sentinel Main Application')
    parser.add_argument('--quick', action='store_true', help='Run quick validation')
    parser.add_argument('--benchmark', action='store_true', help='Run complete benchmark')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--docker', action='store_true', help='Build and run in Docker')
    
    args = parser.parse_args()
    
    # Handle command-line args
    if args.quick:
        print_banner()
        run_quick_validation()
        return
    
    if args.benchmark:
        print_banner()
        run_benchmark()
        return
    
    if args.all:
        print_banner()
        run_everything()
        return
    
    if args.docker:
        print_banner()
        docker_build()
        docker_run()
        return
    
    # Interactive menu
    print_banner()
    
    while True:
        print_menu()
        choice = input("\n👉 Enter choice [0-21]: ").strip()
        
        if choice == '0':
            print("\n✨ Thank you for using Spectral Sentinel! ✨\n")
            break
        
        elif choice == '1':
            run_quick_validation()
        elif choice == '2':
            run_demo()
        elif choice == '3':
            run_aggregator_comparison()
        elif choice == '4':
            run_all_attacks()
        elif choice == '5':
            run_phase_transition()
        elif choice == '6':
            run_all_phase2()
        elif choice == '7':
            run_medium_scale()
        elif choice == '8':
            run_large_scale()
        elif choice == '9':
            run_foundation()
        elif choice == '10':
            run_game_theory()
        elif choice == '11':
            run_ablations()
        elif choice == '12':
            run_certified()
        elif choice == '13':
            run_benchmark()
        elif choice == '14':
            run_limitations()
        elif choice == '15':
            docker_build()
        elif choice == '16':
            docker_run()
        elif choice == '17':
            check_gpu()
        elif choice == '20':
            run_everything()
        elif choice == '21':
            generate_report()
        else:
            print("\n❌ Invalid choice. Please try again.")
        
        input("\n\n⏸️  Press Enter to continue...")
        print("\n" * 2)


if __name__ == '__main__':
    main()
