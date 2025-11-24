#!/usr/bin/env python3
"""
Phase Transition Validation

Tests the theoretical phase transition at σ²f² = 0.25.
Sweeps Byzantine ratio and tracks detection degradation.

Expected runtime: ~25 minutes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from spectral_sentinel.config import Config
from spectral_sentinel.experiments.simulate_basic import run_experiment

# Byzantine ratios to test
BYZANTINE_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.49]

def main():
    print("="*70)
    print("⚡ PHASE TRANSITION VALIDATION")
    print("="*70)
    print(f"\nTesting Byzantine ratios: {BYZANTINE_RATIOS}")
    print("Expected phase transition: σ²f² ≥ 0.25")
    print()
    
    # Create results directory
    results_dir = Path('./results/phase_transition')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for results
    all_results = {}
    
    # Common configuration
    base_config = Config(
        dataset='mnist',
        num_clients=10,
        aggregator='spectral_sentinel',
        attack_type='minmax',
        num_rounds=5,
        local_epochs=1,
        batch_size=32,
        seed=42,
        save_model=False,
        visualize=False
    )
    
    # Run each Byzantine ratio
    for i, byz_ratio in enumerate(BYZANTINE_RATIOS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(BYZANTINE_RATIOS)}] Testing f = {byz_ratio:.2f} ({int(byz_ratio*100)}% Byzantine)")
        print(f"{'='*70}")
        
        # Update config
        config = base_config
        config.byzantine_ratio = byz_ratio
        config.exp_name = f"phase_f{int(byz_ratio*100)}"
        
        try:
            # Run experiment
            stats = run_experiment(config)
            
            # Extract phase transition metrics
            agg_stats = stats.get('aggregator_stats', {})
            phase_metrics = agg_stats.get('phase_transition_metric', [])
            
            if len(phase_metrics) > 0:
                avg_phase_metric = np.mean(phase_metrics[-10:])  # Last 10 rounds
                max_phase_metric = np.max(phase_metrics)
            else:
                avg_phase_metric = 0
                max_phase_metric = 0
            
            # Detection metrics
            num_byzantine = config.num_byzantine
            detected_counts = stats.get('byzantine_detected_per_round', [])
            
            if len(detected_counts) > 0:
                true_positives = sum(min(d, num_byzantine) for d in detected_counts)
                total_possible = num_byzantine * len(detected_counts)
                recall = true_positives / total_possible if total_possible > 0 else 0
            else:
                recall = 0
            
            # Store results
            all_results[byz_ratio] = {
                'f': byz_ratio,
                'avg_sigma2_f2': avg_phase_metric,
                'max_sigma2_f2': max_phase_metric,
                'final_accuracy': stats['final_accuracy'],
                'detection_recall': recall * 100,
                'detectable': avg_phase_metric < 0.25
            }
            
            status = "✅ Detectable" if avg_phase_metric < 0.25 else "⚠️  Near/Beyond Transition"
            print(f"\n{status}")
            print(f"  σ²f² = {avg_phase_metric:.4f}")
            print(f"  Detection Recall = {recall*100:.1f}%")
            print(f"  Final Accuracy = {stats['final_accuracy']:.1f}%")
            
        except Exception as e:
            print(f"\n❌ f={byz_ratio} failed: {e}")
            all_results[byz_ratio] = {'error': str(e)}
    
    # Generate results table
    print(f"\n{'='*70}")
    print("📊 PHASE TRANSITION RESULTS")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame([
        {
            'Byzantine %': int(res['f']*100),
            'f': res['f'],
            'σ²f² (avg)': res.get('avg_sigma2_f2', 0),
            'Detectable': '✅' if res.get('detectable', False) else '❌',
            'Detection Recall (%)': res.get('detection_recall', 0),
            'Accuracy (%)': res.get('final_accuracy', 0),
        }
        for byz_ratio, res in all_results.items()
        if 'error' not in res
    ])
    
    print(results_df.to_string(index=False))
    
    # Theoretical analysis
    print(f"\n{'='*70}")
    print("🧪 THEORETICAL VALIDATION")
    print(f"{'='*70}")
    
    phase_transition_observed = False
    for _, row in results_df.iterrows():
        if row['σ²f² (avg)'] >= 0.20:
            phase_transition_observed = True
            print(f"\n⚠️  Phase transition effects observed at f={row['f']:.2f}:")
            print(f"    σ²f² = {row['σ²f² (avg)']:.4f} (threshold: 0.25)")
            print(f"    Detection recall dropped to {row['Detection Recall (%)']:.1f}%")
            
            if row['σ²f² (avg)'] >= 0.25:
                print(f"    ❌ BEYOND PHASE TRANSITION - Detection theoretically impossible!")
    
    if not phase_transition_observed:
        print("\n✅ All tested ratios below phase transition threshold")
    
    # Save CSV
    csv_path = results_dir / 'phase_transition.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        # Plot: Detection Rate vs σ²f²
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Detection Recall vs f
        ax1.plot(results_df['f'], results_df['Detection Recall (%)'], 
                'o-', linewidth=2, markersize=8, color='#3498db')
        ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% target')
        ax1.set_xlabel('Byzantine Fraction (f)')
        ax1.set_ylabel('Detection Recall (%)')
        ax1.set_title('Detection Rate vs Byzantine Fraction')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Right: Detection Recall vs σ²f²
        ax2.plot(results_df['σ²f² (avg)'], results_df['Detection Recall (%)'],
                'o-', linewidth=2, markersize=8, color='#e74c3c')
        ax2.axvline(x=0.25, color='red', linestyle='--', linewidth=2, 
                   label='Phase transition (σ²f²=0.25)')
        ax2.set_xlabel('σ²f² (Phase Transition Metric)')
        ax2.set_ylabel('Detection Recall (%)')
        ax2.set_title('Detection Rate vs Phase Transition Metric')
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / 'phase_transition_analysis.png', dpi=150)
        print(f"📊 Plot saved: phase_transition_analysis.png")
        
        plt.close('all')
        
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")
    
    print(f"\n{'='*70}")
    print("✅ PHASE TRANSITION VALIDATION COMPLETE!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
