#!/usr/bin/env python3
"""
Blockchain Integration Test

Tests the blockchain connector with a local Hardhat network.
Run this after deploying the contract locally.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from spectral_sentinel.blockchain import BlockchainConnector, BlockchainConfig
from spectral_sentinel.utils.models import SimpleCNN  # Using existing model


def create_test_model():
    """Create a simple test model."""
    model = SimpleCNN(num_classes=10, input_channels=1)
    return model.state_dict()


def test_blockchain_integration():
    """Test end-to-end blockchain integration."""
    
    print("=" * 70)
    print("🧪 BLOCKCHAIN INTEGRATION TEST")
    print("=" * 70)
    
    # Configuration (assumes local Hardhat network is running)
    config = BlockchainConfig(
        network="local",
        rpc_url="http://127.0.0.1:8545",
        # Use first Hardhat account (default private key)
        private_key="ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        # CONTRACT_ADDRESS will be loaded from env or deployment.json
    )
    
    try:
        # Initialize connector
        print("\n📡 Step 1: Initializing blockchain connector...")
        connector = BlockchainConnector(config)
        
        # Test 1: Register clients
        print("\n👥 Step 2: Registering clients...")
        num_clients = 5
        client_ids = list(range(num_clients))
        
        tx_hash = connector.register_clients_batch(client_ids)
        print(f"   Transaction: {connector.get_transaction_url(tx_hash)}")
        
        # Verify registration
        for client_id in client_ids:
            is_registered = connector.is_client_registered()
            assert is_registered, f"Client {client_id} not registered!"
        print(f"   ✅ All {num_clients} clients registered successfully")
        
        # Test 2: Start round
        print("\n🚀 Step 3: Starting Round 1...")
        tx_hash = connector.start_round()
        current_round = connector.get_current_round()
        assert current_round == 1, "Round not started!"
        print(f"   ✅ Round {current_round} started")
        
        # Test 3: Submit model updates
        print("\n📤 Step 4: Submitting model updates...")
        test_models = {}
        
        for client_id in client_ids:
            model_dict = create_test_model()
            test_models[client_id] = model_dict
            
            tx_hash = connector.submit_model_update(
                model_dict=model_dict,
                client_id=client_id,
                round_num=current_round
            )
        
        print(f"   ✅ All {num_clients} model updates submitted")
        
        # Test 4: Verify submissions
        print("\n🔍 Step 5: Verifying submissions on blockchain...")
        submissions = connector.get_round_submissions(current_round)
        assert submissions == num_clients, f"Expected {num_clients} submissions, got {submissions}"
        print(f"   ✅ {submissions}/{num_clients} submissions verified")
        
        # Test 5: Retrieve model updates
        print("\n📥 Step 6: Retrieving model updates...")
        retrieved_models = []
        
        for client_id in client_ids:
            model_dict = connector.get_model_update(current_round, client_id)
            assert model_dict is not None, f"Failed to retrieve model for client {client_id}"
            retrieved_models.append(model_dict)
        
        print(f"   ✅ Retrieved all {len(retrieved_models)} models")
        
        # Test 6: Verify model integrity
        print("\n🔐 Step 7: Verifying model integrity...")
        for client_id in client_ids:
            original = test_models[client_id]
            retrieved = retrieved_models[client_id]
            
            # Compare all parameters
            for key in original.keys():
                assert torch.allclose(original[key], retrieved[key]), \
                    f"Model mismatch for client {client_id}, key {key}"
        
        print("   ✅ All models verified - hashes match!")
        
        # Test 7: Finalize round
        print("\n✅ Step 8: Finalizing round...")
        aggregated_model = create_test_model()  # In real FL, this would be averaged
        tx_hash = connector.finalize_round(aggregated_model, current_round)
        
        # Check round info
        round_info = connector.get_round_info(current_round)
        assert round_info["finalized"], "Round not finalized!"
        assert round_info["num_submissions"] == num_clients
        
        print(f"   ✅ Round {current_round} finalized")
        print(f"   Aggregated model hash: {round_info['aggregated_model_hash'][:16]}...")
        
        # Test 8: Gas and cost estimation
        print("\n💰 Step 9: Gas statistics...")
        gas_stats = connector.get_gas_stats()
        print(f"   Current gas price: {gas_stats['gas_price_gwei']:.2f} gwei")
        print(f"   Cost per transaction: ~{gas_stats['estimated_cost_per_tx']:.6f} ETH")
        
        cost_estimate = connector.estimate_experiment_cost(
            num_clients=20,
            num_rounds=50
        )
        print(f"\n   📊 Cost estimate for full experiment:")
        print(f"      Clients: {cost_estimate['num_clients']}")
        print(f"      Rounds: {cost_estimate['num_rounds']}")
        print(f"      Total transactions: {cost_estimate['total_transactions']}")
        print(f"      Total cost: {cost_estimate['total_cost']}")
        
        # Test 9: Storage stats
        print("\n💾 Step 10: Storage statistics...")
        storage_stats = connector.storage.get_storage_stats()
        print(f"   Files stored: {storage_stats['num_files']}")
        print(f"   Total size: {storage_stats['total_size_mb']:.2f} MB")
        print(f"   Storage dir: {storage_stats['storage_dir']}")
        
        # Success!
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n🎉 Blockchain integration working correctly!")
        print("🚀 Ready to run federated learning on blockchain!\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_blockchain_integration()
    sys.exit(0 if success else 1)
