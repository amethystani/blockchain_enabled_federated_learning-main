"""
Metrics and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os


def plot_training_curves(stats: Dict, 
                        save_path: Optional[str] = None,
                        title: str = "Training Results"):
    """
    Plot training curves (accuracy and loss).
    
    Args:
        stats: Statistics dict from server
        save_path: Path to save figure
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = range(1, len(stats['round_accuracies']) + 1)
    
    # Plot accuracy
    ax1.plot(rounds, stats['round_accuracies'], 'b-', linewidth=2, label='Accuracy')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'{title} - Accuracy')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Plot loss
    ax2.plot(rounds, stats['round_losses'], 'r-', linewidth=2, label='Loss')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{title} - Loss')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_detection_metrics(stats: Dict,
                          num_byzantine: int,
                          save_path: Optional[str] = None):
    """
    Plot Byzantine detection metrics.
    
    Args:
        stats: Statistics dict from server
        num_byzantine: True number of Byzantine clients
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = range(1, len(stats['byzantine_detected_per_round']) + 1)
    detected = stats['byzantine_detected_per_round']
    
    # Plot detected Byzantine clients per round
    ax1.plot(rounds, detected, 'r-', linewidth=2, label='Detected')
    ax1.axhline(y=num_byzantine, color='g', linestyle='--', 
               linewidth=2, label=f'True Byzantine ({num_byzantine})')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Number of Byzantine Clients Detected')
    ax1.set_title('Byzantine Detection Over Rounds')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Compute detection metrics
    true_positives = []
    false_positives = []
    
    for agg_info in stats['aggregation_info']:
        detected_byz = len(agg_info['byzantine_clients'])
        # This is simplified - in reality we'd need ground truth client IDs
        # For now, assume detection is accurate if count matches
        tp = min(detected_byz, num_byzantine)
        fp = max(0, detected_byz - num_byzantine)
        true_positives.append(tp)
        false_positives.append(fp)
    
    # Plot TP and FP
    ax2.plot(rounds, true_positives, 'g-', linewidth=2, label='True Positives')
    ax2.plot(rounds, false_positives, 'r-', linewidth=2, label='False Positives')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Count')
    ax2.set_title('Detection Quality')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved detection metrics to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(results_dict: Dict[str, Dict],
                   metric: str = 'accuracy',
                   save_path: Optional[str] = None):
    """
    Plot comparison of multiple aggregators.
    
    Args:
        results_dict: Dict mapping aggregator names to their stats
        metric: 'accuracy' or 'loss'
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    for agg_name, stats in results_dict.items():
        rounds = range(1, len(stats[f'round_{metric}s']) + 1)
        values = stats[f'round_{metric}s']
        plt.plot(rounds, values, linewidth=2, label=agg_name, marker='o', markersize=3)
    
    plt.xlabel('Round')
    plt.ylabel(metric.capitalize() + (' (%)' if metric == 'accuracy' else ''))
    plt.title(f'Aggregator Comparison - {metric.capitalize()}')
    plt.grid(alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_detection_stats(stats: Dict, 
                           true_byzantine_ids: List[int]) -> Dict:
    """
    Compute detection statistics (precision, recall, F1).
    
    Args:
        stats: Server statistics
        true_byzantine_ids: Ground truth Byzantine client IDs
        
    Returns:
        Detection statistics dict
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for agg_info in stats['aggregation_info']:
        detected = set(agg_info['byzantine_clients'])
        true_byz = set(true_byzantine_ids)
        
        tp = len(detected & true_byz)
        fp = len(detected - true_byz)
        fn = len(true_byz - detected)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }


def print_summary(stats: Dict, config: 'Config'):
    """
    Print experiment summary.
    
    Args:
        stats: Server statistics
        config: Experiment configuration
    """
    print("\n" + "="*60)
    print("📊 EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Model: {config.model_type}")
    print(f"  Clients: {config.num_clients} ({config.num_honest} honest, {config.num_byzantine} Byzantine)")
    print(f"  Attack: {config.attack_type}")
    print(f"  Aggregator: {config.aggregator}")
    print(f"  Rounds: {config.num_rounds}")
    
    print(f"\nResults:")
    print(f"  Final Accuracy: {stats['final_accuracy']:.2f}%")
    print(f"  Max Accuracy: {stats['max_accuracy']:.2f}%")
    print(f"  Final Loss: {stats['final_loss']:.4f}")
    
    if 'total_byzantine_detected' in stats:
        print(f"\nDetection:")
        print(f"  Total Byzantine Detected: {stats['total_byzantine_detected']}")
        print(f"  Avg per Round: {stats['avg_byzantine_per_round']:.2f}")
    
    print("\n" + "="*60 + "\n")
