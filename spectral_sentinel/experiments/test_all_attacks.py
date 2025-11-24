#!/usr/bin/env python3
"""
Attack Robustness Test Suite

Tests Spectral Sentinel against all 10 attack types.
Measures detection effectiveness for each attack.

Expected runtime: ~40 minutes
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

# All attack types
ATTACKS = [
    ('minmax', 'Easy'),
    ('signflip', 'Easy'),
    ('zero', 'Easy'),
    ('gaussian', 'Easy'),
    ('labelflip', 'Medium'),
    ('model_poisoning', 'Medium'),
    ('inversion', 'Medium'),
    ('alie', 'Hard'),
    ('adaptive', 'Hard'),
    ('backdoor', 'Hard'),
]

def main():
    print("="*70)
    print("🎯 ATTACK ROBUSTNESS TEST SUITE")
    print("="*70)
    print(f"\nTesting {len(ATTACKS)} attack types:")
    for i, (attack, difficulty) in enumerate(ATTACKS, 1):
        print(f"  {i}. {attack:20s} ({difficulty})")
    print()
    
    # Create results directory
    results_dir = Path('./results/attack_robustness')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for results
    all_results = {}
    
    # Common configuration  
    base_config = Config(
        dataset='mnist',
        num_clients=5,
        byzantine_ratio=0.4,
        aggregator='spectral_sentinel',
        num_rounds=5,
        local_epochs=1,
        batch_size=32,
        seed=42,
        save_model=False,
        visualize=False
    )
    
    # Run each attack
    for i, (attack_type, difficulty) in enumerate(ATTACKS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(ATTACKS)}] Testing {attack_type.upper()} ({difficulty})")
        print(f"{'='*70}")
        
        # Update config
        config = base_config
        config.attack_type = attack_type
        config.exp_name = f"attack_{attack_type}"
        
        try:
            # Run experiment
            stats = run_experiment(config)
            
            # Calculate detection metrics
            num_byzantine = config.num_byzantine
            total_rounds = config.num_rounds
            
            if 'byzantine_detected_per_round' in stats:
                detected_counts = stats['byzantine_detected_per_round']
                true_positives = sum(min(d, num_byzantine) for d in detected_counts)
                false_positives = sum(max(0, d - num_byzantine) for d in detected_counts)
                false_negatives = sum(max(0, num_byzantine - d) for d in detected_counts)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision, recall, f1 = 0, 0, 0
            
            # Store results
            all_results[attack_type] = {
                'difficulty': difficulty,
                'final_accuracy': stats['final_accuracy'],
                'max_accuracy': stats['max_accuracy'],
                'avg_byzantine_detected': stats.get('avg_byzantine_per_round', 0),
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1 * 100
            }
            
            print(f"\n✅ {attack_type}: Accuracy = {stats['final_accuracy']:.2f}%, "
                  f"Detection Recall = {recall*100:.1f}%")
            
        except Exception as e:
            print(f"\n❌ {attack_type} failed: {e}")
            all_results[attack_type] = {'error': str(e), 'difficulty': difficulty}
    
    # Generate results table
    print(f"\n{'='*70}")
    print("📊 ATTACK ROBUSTNESS RESULTS")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame([
        {
            'Attack': attack,
            'Difficulty': res.get('difficulty', ''),
            'Accuracy (%)': res.get('final_accuracy', 0),
            'Precision (%)': res.get('precision', 0),
            'Recall (%)': res.get('recall', 0),
            'F1 Score (%)': res.get('f1_score', 0),
            'Status': 'Success' if 'error' not in res else 'Failed'
        }
        for attack, res in all_results.items()
    ])
    
    # Sort by difficulty then recall
    difficulty_order = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    results_df['_sort'] = results_df['Difficulty'].map(difficulty_order)
    results_df = results_df.sort_values(['_sort', 'Recall (%)'], ascending=[True, False]).drop('_sort', axis=1)
    
    print(results_df.to_string(index=False))
    
    # Summary stats
    print(f"\n{'='*70}")
    print("📈 SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Average Accuracy: {results_df['Accuracy (%)'].mean():.2f}%")
    print(f"Average Detection Recall: {results_df['Recall (%)'].mean():.2f}%")
    print(f"Average F1 Score: {results_df['F1 Score (%)'].mean():.2f}%")
    print(f"\nBy Difficulty:")
    for diff in ['Easy', 'Medium', 'Hard']:
        subset = results_df[results_df['Difficulty'] == diff]
        print(f"  {diff:8s}: Avg Recall = {subset['Recall (%)'].mean():.1f}%, "
              f"Avg Accuracy = {subset['Accuracy (%)'].mean():.1f}%")
    
    # Save CSV
    csv_path = results_dir / 'attack_robustness.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        # Plot: Detection Recall by Attack Type
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = {'Easy': '#2ecc71', 'Medium': '#f39c12', 'Hard': '#e74c3c'}
        bar_colors = [colors[d] for d in results_df['Difficulty']]
        
        ax.barh(results_df['Attack'], results_df['Recall (%)'], color=bar_colors)
        ax.set_xlabel('Detection Recall (%)')
        ax.set_title('Spectral Sentinel: Detection Rate by Attack Type (40% Byzantine)')
        ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5, label='90% target')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / 'detection_by_attack.png', dpi=150)
        print(f"📊 Plot saved: detection_by_attack.png")
        
        plt.close('all')
        
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")
    
    print(f"\n{'='*70}")
    print("✅ ATTACK ROBUSTNESS TEST COMPLETE!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
