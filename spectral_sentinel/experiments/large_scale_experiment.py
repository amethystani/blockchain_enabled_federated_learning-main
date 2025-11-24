"""
Phase 3B: Large-Scale Experiment (Scaled-Down Version)

ViT-Small (~22M params) on Tiny ImageNet with 32 clients, 30% Byzantine ratio.
Tests Spectral Sentinel with sketching for memory efficiency.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spectral_sentinel.config import Config
from spectral_sentinel.federated.data_loader import load_federated_data
from spectral_sentinel.utils.models import get_model
from spectral_sentinel.federated.client import Client
from spectral_sentinel.federated.server import Server
from spectral_sentinel.aggregators.baselines import get_aggregator
from spectral_sentinel.attacks.attacks import get_attack
from spectral_sentinel.utils.metrics import save_results

def run_large_scale_experiment(
    aggregator_name: str = 'spectral_sentinel',
    num_clients: int = 32,
    byzantine_ratio: float = 0.3,
    num_rounds: int = 25,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Run large-scale experiment with ViT-Small."""
    
    print("\n" + "="*80)
    print("🚀 Phase 3B: Large-Scale Experiment (Scaled-Down)")
    print("="*80)
    print(f"Model: ViT-Small (~22M params)")
    print(f"Dataset: Tiny ImageNet (CIFAR-100 placeholder, 100 classes)")
    print(f"Clients: {num_clients}")
    print(f"Byzantine ratio: {byzantine_ratio:.1%}")
    print(f"Aggregator: {aggregator_name}")
    print(f"Sketching: enabled (k=256)")
    print("="*80 + "\n")
    
    config = Config(
        dataset='tiny_imagenet',
        model_type='vit_small',
        num_clients=num_clients,
        byzantine_ratio=byzantine_ratio,
        attack_type='alie',  # More sophisticated attack
        aggregator=aggregator_name,
        num_rounds=num_rounds,
        local_epochs=2,
        batch_size=16,  # Smaller batch for ViT
        learning_rate=0.001,
        use_sketching=True,
        sketch_size=256,
        device=device
    )
    
    # Load dataset
    print("📦 Loading Tiny ImageNet...")
    client_datasets, test_dataset = load_federated_data(
        'tiny_imagenet',
        num_clients=num_clients,
        non_iid_alpha=0.5,
        data_dir='./data'
    )
    
    # Create ViT-Small model (100 classes for CIFAR-100)
    print("\n🏗️  Creating ViT-Small model...")
    model = get_model('vit_small', num_classes=100).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.1f}M")
    
    # Create clients
    num_byzantine = int(num_clients * byzantine_ratio)
    attack = get_attack('alie', attack_strength=1.0)
    
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
    
    print(f"\n👥 Created {num_clients} clients:")
    print(f"  ✓ {num_clients - num_byzantine} honest")
    print(f"  ✗ {num_byzantine} Byzantine")
    
    # Create aggregator
    aggregator_kwargs = {}
    if aggregator_name in ['krum', 'bulyan']:
        aggregator_kwargs['num_byzantine'] = num_byzantine
    
    aggregator = get_aggregator(aggregator_name, **aggregator_kwargs)
    
    # Create server
    server = Server(
        model=model,
        clients=clients,
        test_dataset=test_dataset,
        aggregator=aggregator,
        config=config
    )
    
    # Training
    print(f"\n🏋️  Starting training ({num_rounds} rounds)...")
    print("-" * 80)
    
    results = server.train(num_rounds=num_rounds)
    
    # Results
    print("\n" + "="*80)
    print("📊 Final Results")
    print("="*80)
    print(f"Final Accuracy: {results['test_accuracies'][-1]:.2f}%")
    print(f"Best Accuracy: {max(results['test_accuracies']):.2f}%")
    print(f"Sketch Memory: ~{config.sketch_size**2 * 4 / 1024 / 1024:.0f}MB")
    print("="*80)
    
    # Save
    results_dir = f'results/phase3b_large_scale/{aggregator_name}'
    os.makedirs(results_dir, exist_ok=True)
    save_results(results, results_dir, f'large_scale_{aggregator_name}')
    print(f"\n💾 Results saved to: {results_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 3B: Large-Scale Experiment')
    parser.add_argument('--aggregator', type=str, default='spectral_sentinel')
    parser.add_argument('--num_clients', type=int, default=32)
    parser.add_argument('--byzantine_ratio', type=float, default=0.3)
    parser.add_argument('--num_rounds', type=int, default=25)
    args = parser.parse_args()
    
    run_large_scale_experiment(
        aggregator_name=args.aggregator,
        num_clients=args.num_clients,
        byzantine_ratio=args.byzantine_ratio,
        num_rounds=args.num_rounds
    )


if __name__ == '__main__':
    main()
