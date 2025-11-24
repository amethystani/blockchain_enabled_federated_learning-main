"""
Phase 3A: Medium-Scale Experiment (Scaled-Down Version)

ResNet-50 (~25M params) on FEMNIST with 50 clients, 40% Byzantine ratio.
Tests Spectral Sentinel vs FLTrust, FLAME, and all baseline aggregators.
"""

import sys
import os
import torch
import numpy as np
import time
from typing import Dict, List
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spectral_sentinel.config import Config
from spectral_sentinel.federated.data_loader import load_federated_data, compute_tv_distance
from spectral_sentinel.utils.models import get_model
from spectral_sentinel.federated.client import Client
from spectral_sentinel.federated.server import Server
from spectral_sentinel.aggregators.baselines import get_aggregator
from spectral_sentinel.attacks.attacks import get_attack
from spectral_sentinel.utils.metrics import plot_training_curves, save_results

def run_medium_scale_experiment(
    aggregator_name: str = 'spectral_sentinel',
    num_clients: int = 50,
    byzantine_ratio: float = 0.4,
    num_rounds: int = 30,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Run medium-scale experiment.
    
    Args:
        aggregator_name: Aggregator to test
        num_clients: Number of clients (50 for scaled-down)
        byzantine_ratio: Fraction of Byzantine clients
        num_rounds: Number of FL rounds
        device: Device to use
    """
    print("\n" + "="*80)
    print("🚀 Phase 3A: Medium-Scale Experiment (Scaled-Down)")
    print("="*80)
    print(f"Model: ResNet-50 (~25M params)")
    print(f"Dataset: FEMNIST (62 classes)")
    print(f"Clients: {num_clients}")
    print(f"Byzantine ratio: {byzantine_ratio:.1%}")
    print(f"Aggregator: {aggregator_name}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Configuration
    config = Config(
        dataset='femnist',
        model_type='resnet50',
        num_clients=num_clients,
        byzantine_ratio=byzantine_ratio,
        attack_type='minmax',
        aggregator=aggregator_name,
        num_rounds=num_rounds,
        local_epochs=2,
        batch_size=32,
        learning_rate=0.01,
        use_sketching=True,
        sketch_size=256,
        device=device
    )
    
    # Load FEMNIST dataset
    print("📦 Loading FEMNIST dataset...")
    client_datasets, test_dataset = load_federated_data(
        'femnist',
        num_clients=num_clients,
        non_iid_alpha=0.3,  # High heterogeneity
        data_dir='./data'
    )
    
    # Compute heterogeneity (TV distance)
    print("\n📊 Computing heterogeneity metrics...")
    # Get global label distribution
    if hasattr(test_dataset, 'targets'):
        global_labels = np.array(test_dataset.targets)
    elif hasattr(test_dataset, 'labels'):
        global_labels = np.array(test_dataset.labels)
    else:
        global_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
    
    tv_distances = [compute_tv_distance(ds, global_labels) for ds in client_datasets]
    avg_tv = np.mean(tv_distances)
    print(f"Average TV distance: {avg_tv:.3f} (higher = more heterogeneous)")
    
    # Create model
    print("\n🏗️  Creating ResNet-50 model...")
    model = get_model('resnet50', num_classes=62, input_channels=1).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.1f}M")
    
    # Create clients
    print(f"\n👥 Creating {num_clients} clients...")
    num_byzantine = int(num_clients * byzantine_ratio)
    attack = get_attack('minmax', attack_strength=1.0)
    
    clients = []
    for i in range(num_clients):
        is_byzantine = i < num_byzantine
        client = Client(
            client_id=i,
            dataset=client_datasets[i],
            model=model,
            is_byzantine=is_byzantine,
            attack=attack if is_byzantine else None,
            config=config
        )
        clients.append(client)
    
    print(f"✓ {num_clients - num_byzantine} honest clients")
    print(f"✗ {num_byzantine} Byzantine clients ({byzantine_ratio:.1%})")
    
    # Create aggregator
    print(f"\n🎯 Creating {aggregator_name} aggregator...")
    aggregator_kwargs = {}
    if aggregator_name == 'krum' or aggregator_name == 'bulyan':
        aggregator_kwargs['num_byzantine'] = num_byzantine
    
    aggregator = get_aggregator(aggregator_name, **aggregator_kwargs)
    
    # Create server
    print("🖥️  Creating server...")
    server = Server(
        model=model,
        clients=clients,
        test_dataset=test_dataset,
        aggregator=aggregator,
        config=config
    )
    
    # Training
    print(f"\n🏋️  Starting federated training ({num_rounds} rounds)...")
    print("-" * 80)
    
    results = server.train(num_rounds=num_rounds)
    
    # Print final results
    print("\n" + "="*80)
    print("📊 Final Results")
    print("="*80)
    print(f"Final Test Accuracy: {results['test_accuracies'][-1]:.2f}%")
    print(f"Best Test Accuracy: {max(results['test_accuracies']):.2f}%")
    
    if 'detection_stats' in results:
        print(f"\nDetection Statistics:")
        print(f"  Average Detection Recall: {np.mean(results['detection_stats']['recall']):.2%}")
        print(f"  Average Detection Precision: {np.mean(results['detection_stats']['precision']):.2%}")
        print(f"  Average F1 Score: {np.mean(results['detection_stats']['f1']):.2%}")
    
    # Memory usage
    if config.use_sketching:
        sketch_memory = config.sketch_size ** 2 * 4 / 1024 / 1024  # MB
        print(f"\nMemory Usage (sketching enabled):")
        print(f"  Sketch size: {config.sketch_size}")
        print(f"  Estimated memory: ~{sketch_memory:.0f}MB")
    
    print("="*80)
    
    # Save results
    results_dir = f'results/phase3a_medium_scale/{aggregator_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(
        results,
        save_dir=results_dir,
        experiment_name=f'medium_scale_{aggregator_name}'
    )
    
    print(f"\n💾 Results saved to: {results_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 3A: Medium-Scale Experiment')
    parser.add_argument('--aggregator', type=str, default='spectral_sentinel',
                       help='Aggregator to use')
    parser.add_argument('--num_clients', type=int, default=50,
                       help='Number of clients')
    parser.add_argument('--byzantine_ratio', type=float, default=0.4,
                       help='Byzantine client ratio')
    parser.add_argument('--num_rounds', type=int, default=30,
                       help='Number of rounds')
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare all aggregators')
    
    args = parser.parse_args()
    
    if args.compare_all:
        # Compare all aggregators
        aggregators = ['spectral_sentinel', 'fltrust', 'flame', 'bulyan', 'krum', 'fedavg']
        all_results = {}
        
        for agg in aggregators:
            print(f"\n\n{'='*80}")
            print(f"Testing aggregator: {agg}")
            print(f"{'='*80}\n")
            
            results = run_medium_scale_experiment(
                aggregator_name=agg,
                num_clients=args.num_clients,
                byzantine_ratio=args.byzantine_ratio,
                num_rounds=args.num_rounds
            )
            all_results[agg] = results
        
        # Plot comparison
        print("\n📊 Generating comparison plots...")
        plot_training_curves(
            all_results,
            save_path='results/phase3a_medium_scale/comparison.png',
            title= 'Phase 3A: Medium-Scale Comparison (ResNet-50 on FEMNIST)'
        )
    else:
        # Single aggregator
        run_medium_scale_experiment(
            aggregator_name=args.aggregator,
            num_clients=args.num_clients,
            byzantine_ratio=args.byzantine_ratio,
            num_rounds=args.num_rounds
        )


if __name__ == '__main__':
    main()
