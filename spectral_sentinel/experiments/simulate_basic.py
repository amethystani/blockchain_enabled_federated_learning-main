#!/usr/bin/env python3
"""
Spectral Sentinel: Basic Simulation

End-to-end simulation on MNIST/CIFAR-10 to demonstrate Byzantine detection.
"""

import torch
import numpy as np
import random
import argparse
import os
from pathlib import Path

# Spectral Sentinel imports
from spectral_sentinel.config import Config
from spectral_sentinel.federated.data_loader import load_federated_data
from spectral_sentinel.federated.client import HonestClient, ByzantineClient
from spectral_sentinel.federated.server import FederatedServer
from spectral_sentinel.attacks.attacks import get_attack
from spectral_sentinel.aggregators.spectral_sentinel import SpectralSentinelAggregator
from spectral_sentinel.aggregators.baselines import get_aggregator
from spectral_sentinel.utils.models import get_model
from spectral_sentinel.utils.metrics import (
    plot_training_curves,
    plot_detection_metrics,
    print_summary
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_clients(config: Config, 
                  client_datasets: list,
                  model: torch.nn.Module) -> tuple:
    """
    Create honest and Byzantine clients.
    
    Returns:
        (all_clients, honest_clients, byzantine_clients, byzantine_ids)
    """
    clients = []
    byzantine_ids = []
    
    # Create attack instance for Byzantine clients
    attack = get_attack(config.attack_type, attack_strength=1.0)
    
    # Determine which clients are Byzantine
    all_client_ids = list(range(config.num_clients))
    byzantine_client_ids = random.sample(all_client_ids, config.num_byzantine)
    byzantine_ids = byzantine_client_ids
    
    print(f"\n🎯 Client Assignment:")
    print(f"  Byzantine clients: {byzantine_client_ids}")
    print(f"  Attack type: {config.attack_type}")
    
    # Create clients
    for i in range(config.num_clients):
        if i in byzantine_client_ids:
            client = ByzantineClient(
                client_id=i,
                model=get_model(config.model_type, config.num_classes,
                              1 if config.dataset == 'mnist' else 3),
                dataset=client_datasets[i],
                attack=attack,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                device=config.device,
                num_classes=config.num_classes
            )
        else:
            client = HonestClient(
                client_id=i,
                model=get_model(config.model_type, config.num_classes,
                              1 if config.dataset == 'mnist' else 3),
                dataset=client_datasets[i],
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                device=config.device
            )
        
        clients.append(client)
    
    honest_clients = [c for c in clients if isinstance(c, HonestClient)]
    byzantine_clients = [c for c in clients if isinstance(c, ByzantineClient)]
    
    return clients, honest_clients, byzantine_clients, byzantine_ids


def run_experiment(config: Config):
    """
    Run a complete federated learning experiment.
    
    Args:
        config: Experiment configuration
    """
    print("\n" + "="*70)
    print("🚀 SPECTRAL SENTINEL: Byzantine-Robust Federated Learning")
    print("="*70)
    
    # Set random seed
    set_seed(config.seed)
    
    # Load and partition data
    print(f"\n📁 Loading {config.dataset.upper()} dataset...")
    client_datasets, test_dataset = load_federated_data(
        config.dataset,
        config.num_clients,
        config.non_iid_alpha
    )
    
    # Create model
    print(f"\n🏗️  Creating {config.model_type} model...")
    input_channels = 1 if config.dataset == 'mnist' else 3
    model = get_model(config.model_type, config.num_classes, input_channels)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create aggregator
    print(f"\n⚙️  Initializing {config.aggregator} aggregator...")
    if config.aggregator == 'spectral_sentinel':
        aggregator = SpectralSentinelAggregator(
            ks_threshold=config.mp_threshold,
            tail_threshold=config.tail_threshold,
            use_sketching=config.use_sketching,
            sketch_size=config.sketch_size,
            window_size=config.sliding_window_size,
            clip_threshold=config.clip_threshold
        )
    else:
        aggregator = get_aggregator(
            config.aggregator,
            num_byzantine=config.num_byzantine
        )
    
    # Create clients
    clients, honest_clients, byzantine_clients, byzantine_ids = create_clients(
        config, client_datasets, model
    )
    
    # Create server
    print(f"\n🖥️  Initializing federated server...")
    server = FederatedServer(
        model=model,
        aggregator=aggregator,
        test_dataset=test_dataset,
        device=config.device
    )
    
    # Run training
    print(f"\n{'='*70}")
    print("🎓 FEDERATED TRAINING")
    print(f"{'='*70}")
    
    stats = server.train(
        clients=clients,
        num_rounds=config.num_rounds,
        num_local_epochs=config.local_epochs,
        clients_per_round=config.clients_per_round,
        verbose=True
    )
    
    # Print summary
    print_summary(stats, config)
    
    # Save results
    if config.save_model or config.visualize:
        os.makedirs(config.save_path, exist_ok=True)
    
    # Visualizations
    if config.visualize:
        print("📊 Generating visualizations...")
        
        # Training curves
        plot_training_curves(
            stats,
            save_path=os.path.join(config.save_path, 'training_curves.png'),
            title=f"{config.aggregator} on {config.dataset}"
        )
        
        # Detection metrics
        if len(stats['byzantine_detected_per_round']) > 0:
            plot_detection_metrics(
                stats,
                num_byzantine=config.num_byzantine,
                save_path=os.path.join(config.save_path, 'detection_metrics.png')
            )
        
        # Spectral visualization (if using Spectral Sentinel)
        if config.aggregator == 'spectral_sentinel':
            aggregator.visualize_detection(
                save_path=os.path.join(config.save_path, 'spectral_analysis.png')
            )
    
    # Save model
    if config.save_model:
        model_path = os.path.join(config.save_path, 'final_model.pt')
        torch.save(server.model.state_dict(), model_path)
        print(f"💾 Saved model to {model_path}")
    
    print("\n✅ Experiment complete!\n")
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Spectral Sentinel Simulation')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10', 'cifar100'],
                       help='Dataset to use')
    
    # Federated learning
    parser.add_argument('--num_clients', type=int, default=20,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local epochs per round')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5,
                       help='Dirichlet alpha for Non-IID (lower = more skew)')
    
    # Byzantine settings
    parser.add_argument('--byzantine_ratio', type=float, default=0.4,
                       help='Fraction of Byzantine clients')
    parser.add_argument('--attack_type', type=str, default='minmax',
                       choices=['minmax', 'labelflip', 'alie', 'adaptive', 
                               'signflip', 'zero', 'gaussian', 'backdoor', 'model_poisoning'],
                       help='Attack type')
    
    # Aggregator
    parser.add_argument('--aggregator', type=str, default='spectral_sentinel',
                       choices=['spectral_sentinel', 'fedavg', 'krum', 
                               'geometric_median', 'trimmed_mean', 'median',
                               'bulyan', 'signguard'],
                       help='Aggregation method')
    
    # Spectral Sentinel specific
    parser.add_argument('--use_sketching', action='store_true',
                       help='Enable Frequent Directions sketching')
    parser.add_argument('--sketch_size', type=int, default=256,
                       help='Sketch size k')
    
    # Model
    parser.add_argument('--model_type', type=str, default='simple_cnn',
                       choices=['simple_cnn', 'lenet5', 'resnet18'],
                       help='Model architecture')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--save_path', type=str, default='./results',
                       help='Path to save results')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable visualizations')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        non_iid_alpha=args.non_iid_alpha,
        byzantine_ratio=args.byzantine_ratio,
        attack_type=args.attack_type,
        aggregator=args.aggregator,
        use_sketching=args.use_sketching,
        sketch_size=args.sketch_size,
        model_type=args.model_type,
        seed=args.seed,
        device=args.device if args.device != 'auto' else 
               ('cuda' if torch.cuda.is_available() else 'cpu'),
        save_path=args.save_path,
        visualize=not args.no_visualize
    )
    
    # Run experiment
    run_experiment(config)


if __name__ == '__main__':
    main()
