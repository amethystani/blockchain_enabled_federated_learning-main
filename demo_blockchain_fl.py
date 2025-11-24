#!/usr/bin/env python3
"""
Simple Blockchain FL Demo

Demonstrates federated learning with blockchain integration.
This is a minimal example showing the core workflow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from spectral_sentinel.blockchain import BlockchainConnector, BlockchainConfig
from spectral_sentinel.utils.models import SimpleCNN


def simple_federated_round():
    """Run a simple federated learning round on blockchain."""
    
    print("""
╔════════════════════════════════════════════════════════╗
║                                                        ║
║        🔗 BLOCKCHAIN FEDERATED LEARNING DEMO          ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    print("📡 Configuring blockchain connection...")
    config = BlockchainConfig(
        network="local",  # Change to "mumbai" for testnet
        rpc_url="http://127.0.0.1:8545",
        # Default Hardhat account #0
        private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    )
    
    # Initialize connector
    print("\n🔌 Connecting to blockchain...")
    connector = BlockchainConnector(config)
    
    # Simulate 3 clients
    num_clients = 3
    client_ids = list(range(num_clients))
    
    # Register clients
    print(f"\n👥 Registering {num_clients} clients...")
    connector.register_clients_batch(client_ids)
    
    # Start round
    print("\n🚀 Starting federated learning round...")
    connector.start_round()
    current_round = connector.get_current_round()
    
    # Simulate local training and submission
    print(f"\n🏋️  Clients training locally and submitting to blockchain...")
    models = []
    
    for client_id in client_ids:
        # Simulate local training (in reality, this would be actual training)
        model = SimpleCNN(num_classes=10, input_channels=1)
        # Add some random noise to simulate different local updates
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.01
        
        model_dict = model.state_dict()
        models.append(model_dict)
        
        # Submit to blockchain
        tx_hash = connector.submit_model_update(model_dict, client_id, current_round)
        print(f"   ✅ Client {client_id} submitted (tx: {tx_hash[:10]}...)")
    
    # Wait for all submissions
    print(f"\n⏳ Waiting for all submissions...")
    success = connector.wait_for_submissions(current_round, num_clients, timeout=60)
    
    if not success:
        print("❌ Not all clients submitted in time!")
        return False
    
    # Server retrieves and aggregates
    print(f"\n📥 Server retrieving models from blockchain...")
    retrieved_models = []
    
    for client_id in client_ids:
        model_dict = connector.get_model_update(current_round, client_id)
        if model_dict:
            retrieved_models.append(model_dict)
            print(f"   ✅ Retrieved model from client {client_id}")
        else:
            print(f"   ❌ Failed to retrieve model from client {client_id}")
    
    # Simple averaging (FedAvg)
    print(f"\n🔄 Aggregating models (FedAvg)...")
    aggregated_model = {}
    for key in retrieved_models[0].keys():
        aggregated_model[key] = torch.stack([m[key] for m in retrieved_models]).mean(dim=0)
    
    print(f"   ✅ Aggregation complete")
    
    # Finalize round
    print(f"\n💾 Finalizing round on blockchain...")
    tx_hash = connector.finalize_round(aggregated_model, current_round)
    
    # Get round info
    round_info = connector.get_round_info(current_round)
    
    print(f"\n" + "=" * 60)
    print("✅ ROUND COMPLETE!")
    print("=" * 60)
    print(f"Round Number:     {round_info['round_number']}")
    print(f"Submissions:      {round_info['num_submissions']}/{num_clients}")
    print(f"Finalized:        {round_info['finalized']}")
    print(f"Model Hash:       {round_info['aggregated_model_hash'][:16]}...")
    
    # Show storage stats
    storage_stats = connector.storage.get_storage_stats()
    print(f"\n📊 Storage Statistics:")
    print(f"Files stored:     {storage_stats['num_files']}")
    print(f"Total size:       {storage_stats['total_size_mb']:.2f} MB")
    
    # Show cost estimate
    print(f"\n💰 Cost Analysis:")
    cost_estimate = connector.estimate_experiment_cost(num_clients=20, num_rounds=50)
    print(f"For 20 clients × 50 rounds:")
    print(f"Total transactions: {cost_estimate['total_transactions']}")
    print(f"Estimated cost:     {cost_estimate['total_cost']}")
    
    print(f"\n🎉 Success! Federated learning round completed on blockchain!\n")
    
    return True


if __name__ == "__main__":
    try:
        success = simple_federated_round()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
