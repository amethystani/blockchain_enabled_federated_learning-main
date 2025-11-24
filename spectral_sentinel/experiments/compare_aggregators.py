#!/usr/bin/env python3
"""
Aggregator Comparison Suite

Compares all 7 aggregators under Byzantine attack.
Generates comparison plots and CSV results.

Expected runtime: ~30 minutes
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

# Aggregators to compare
AGGREGATORS = [
    'spectral_sentinel',
    'fedavg',
    'krum',
    'geometric_median',
    'trimmed_mean',
    'median',
    'bulyan',
    'signguard'
]

def main():
    print("="*70)
    print("📊 AGGREGATOR COMPARISON SUITE")
    print("="*70)
    print(f"\nComparing {len(AGGREGATORS)} aggregators:")
    for i, agg in enumerate(AGGREGATORS, 1):
        print(f"  {i}. {agg}")
    print()
    
    # Create results directory
    results_dir = Path('./results/aggregator_comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for results
    all_results = {}
    
    # Common configuration
    base_config = Config(
        dataset='mnist',
        num_clients=5,
        byzantine_ratio=0.4,
        attack_type='minmax',
        num_rounds=5,
        local_epochs=1,
        batch_size=32,
        seed=42,
        save_model=False,
        visualize=False
    )
    
    # Run each aggregator
    for i, aggregator in enumerate(AGGREGATORS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(AGGREGATORS)}] Testing {aggregator.upper()}")
        print(f"{'='*70}")
        
        # Update config
        config = base_config
        config.aggregator = aggregator
        config.exp_name = f"comp_{aggregator}"
        
        try:
            # Run experiment
            stats = run_experiment(config)
            
            # Store results
            all_results[aggregator] = {
                'final_accuracy': stats['final_accuracy'],
                'max_accuracy': stats['max_accuracy'],
                'final_loss': stats['final_loss'],
                'avg_byzantine_detected': stats.get('avg_byzantine_per_round', 0),
                'total_byzantine_detected': stats.get('total_byzantine_detected', 0),
                'accuracy_curve': stats['round_accuracies'],
                'loss_curve': stats['round_losses']
            }
            
            print(f"\n✅ {aggregator}: Final Accuracy = {stats['final_accuracy']:.2f}%")
            
        except Exception as e:
            print(f"\n❌ {aggregator} failed: {e}")
            all_results[aggregator] = {'error': str(e)}
    
    # Generate comparison table
    print(f"\n{'='*70}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame([
        {
            'Aggregator': agg,
            'Final Acc (%)': res.get('final_accuracy', 0),
            'Max Acc (%)': res.get('max_accuracy', 0),
            'Final Loss': res.get('final_loss', 0),
            'Avg Byz Detected': res.get('avg_byzantine_detected', 0),
            'Status': 'Success' if 'error' not in res else 'Failed'
        }
        for agg, res in all_results.items()
    ]).sort_values('Final Acc (%)', ascending=False)
    
    print(results_df.to_string(index=False))
    
    # Save CSV
    csv_path = results_dir / 'comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        # Plot 1: Final Accuracy Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        sorted_results = results_df.sort_values('Final Acc (%)', ascending=True)
        colors = ['#2ecc71' if agg == 'spectral_sentinel' else '#3498db' 
                 for agg in sorted_results['Aggregator']]
        ax.barh(sorted_results['Aggregator'], sorted_results['Final Acc (%)'], color=colors)
        ax.set_xlabel('Final Accuracy (%)')
        ax.set_title('Aggregator Comparison: Final Accuracy (40% Byzantine, Min-Max Attack)')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / 'final_accuracy_comparison.png', dpi=150)
        print(f"📊 Plot saved: final_accuracy_comparison.png")
        
        # Plot 2: Training Curves
        fig, ax = plt.subplots(figsize=(14, 7))
        for agg, res in all_results.items():
            if 'accuracy_curve' in res:
                rounds = range(1, len(res['accuracy_curve']) + 1)
                linewidth = 3 if agg == 'spectral_sentinel' else 2
                alpha = 1.0 if agg == 'spectral_sentinel' else 0.7
                ax.plot(rounds, res['accuracy_curve'], label=agg, linewidth=linewidth, alpha=alpha)
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Aggregator Comparison: Training Curves')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / 'training_curves_comparison.png', dpi=150)
        print(f"📊 Plot saved: training_curves_comparison.png")
        
        plt.close('all')
        
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
