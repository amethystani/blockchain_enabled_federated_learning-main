#!/usr/bin/env python3
"""
Minimal On-Chain Experiment - Gas Efficient Version

Runs a single round with minimal blockchain operations to test the system.
Use this when balance is low, then scale up once verified.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import json
from datetime import datetime
import os
from dotenv import load_dotenv

from spectral_sentinel.blockchain import BlockchainConnector, BlockchainConfig
from spectral_sentinel.federated.server import FederatedServer
from spectral_sentinel.federated.client import HonestClient, ByzantineClient
from spectral_sentinel.federated.data_loader import load_federated_data, create_data_loader
from spectral_sentinel.aggregators.spectral_sentinel import SpectralSentinelAggregator
from spectral_sentinel.aggregators.baselines import FedAvgAggregator
from spectral_sentinel.utils.models import SimpleCNN
from spectral_sentinel.attacks.attacks import MinMaxAttack

load_dotenv()

def run_minimal_experiment(aggregator_name='spectral_sentinel', use_blockchain=True):
    """Run minimal experiment with optional blockchain."""
    
    print("="*70)
    print("🧪 MINIMAL ON-CHAIN EXPERIMENT")
    print("="*70)
    print(f"Aggregator: {aggregator_name}")
    print(f"Blockchain: {'Enabled' if use_blockchain else 'Disabled (simulation only)'}")
    print("="*70)
    
    # Load data
    print("\n📊 Loading MNIST...")
    train_datasets, test_dataset = load_federated_data(
        dataset_name='mnist',
        num_clients=3,
        non_iid_alpha=0.5,
        data_dir='./data'
    )
    print(f"✅ Loaded {len(train_datasets)} client datasets")
    
    # Create model
    model = SimpleCNN(num_classes=10, input_channels=1)
    
    # Create aggregator
    if aggregator_name == 'spectral_sentinel':
        aggregator = SpectralSentinelAggregator()
    else:
        aggregator = FedAvgAggregator()
    
    # Create server
    server = FederatedServer(
        model=model,
        aggregator=aggregator,
        test_dataset=test_dataset,
        device='cpu',
        batch_size=32
    )
    
    # Create clients
    clients = []
    for i in range(3):
        client_model = SimpleCNN(num_classes=10, input_channels=1)
        if i == 0:  # One Byzantine
            attack = MinMaxAttack()
            client = ByzantineClient(
                client_id=i,
                model=client_model,
                dataset=train_datasets[i],
                attack=attack,
                device='cpu'
            )
        else:
            client = HonestClient(
                client_id=i,
                dataset=train_datasets[i],
                model=client_model,
                device='cpu'
            )
        clients.append(client)
    
    # Blockchain setup (if enabled)
    blockchain = None
    if use_blockchain:
        print("\n🔗 Connecting to blockchain...")
        config = BlockchainConfig(
            network="amoy",
            rpc_url=os.getenv("AMOY_RPC_URL") or "https://rpc-amoy.polygon.technology/",
            private_key=os.getenv("PRIVATE_KEY"),
            contract_address=os.getenv("CONTRACT_ADDRESS"),
        )
        blockchain = BlockchainConnector(config)
        
        # Check balance
        balance_wei = blockchain.w3.eth.get_balance(blockchain.account.address)
        balance_matic = blockchain.w3.from_wei(balance_wei, 'ether')
        print(f"💰 Balance: {balance_matic:.4f} MATIC")
        
        if balance_matic < 0.01:
            print("⚠️  Low balance! Running simulation only (no blockchain writes)")
            use_blockchain = False
    
    # Run ONE round
    print("\n🚀 Running Round 1...")
    
    # Off-chain: Run federated round
    print("   🏋️  Training (off-chain)...")
    round_info = server.federated_round(clients, num_local_epochs=1, verbose=False)
    
    # Evaluate
    accuracy, loss = server.evaluate()
    print(f"   📈 Accuracy: {accuracy:.2f}% | Loss: {loss:.4f}")
    
    # On-chain: Only if enabled and sufficient balance
    if use_blockchain and blockchain:
        try:
            # Start round
            print("   🔗 Starting round on blockchain...")
            current_round = blockchain.get_current_round()
            if current_round == 0:
                tx_hash = blockchain.start_round()
                current_round = blockchain.get_current_round()
                print(f"   ✅ Round {current_round} started")
            
            # Submit ONE model (most important one - aggregated)
            print("   📤 Submitting aggregated model to blockchain...")
            aggregated_model = server.get_model_parameters()
            tx_hash = blockchain.submit_model_update(
                aggregated_model,
                client_id=999,  # Special ID for aggregated
                round_num=current_round
            )
            print(f"   ✅ Model submitted: {blockchain.get_transaction_url(tx_hash)}")
            
            # Finalize
            print("   ✅ Finalizing round...")
            tx_hash = blockchain.finalize_round(aggregated_model, current_round)
            print(f"   ✅ Round finalized: {blockchain.get_transaction_url(tx_hash)}")
            
        except Exception as e:
            print(f"   ⚠️  Blockchain operation failed: {e}")
            print("   ℹ️  Continuing with off-chain results only...")
    
    # Results
    results = {
        'aggregator': aggregator_name,
        'accuracy': accuracy,
        'loss': loss,
        'byzantine_detected': round_info.get('byzantine_detected', 0),
        'blockchain_enabled': use_blockchain,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save
    output_path = Path("./results/blockchain_experiments")
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"minimal_{aggregator_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path / filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Experiment complete!")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Loss: {loss:.4f}")
    print(f"   Byzantine detected: {round_info.get('byzantine_detected', 0)}")
    print(f"   Results saved to: {output_path / filename}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregator', default='spectral_sentinel', 
                       choices=['spectral_sentinel', 'fedavg'])
    parser.add_argument('--no-blockchain', action='store_true', 
                       help='Run simulation only (no blockchain)')
    args = parser.parse_args()
    
    run_minimal_experiment(
        aggregator_name=args.aggregator,
        use_blockchain=not args.no_blockchain
    )

