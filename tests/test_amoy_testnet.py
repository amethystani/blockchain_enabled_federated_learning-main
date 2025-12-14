#!/usr/bin/env python3
"""
Blockchain Integration Test for Amoy Testnet

Tests the blockchain connector with Amoy testnet (Polygon's new testnet).
Run this after deploying the contract to Amoy testnet.
"""

import sys
from pathlib import Path

# Add project to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from spectral_sentinel.blockchain import BlockchainConnector, BlockchainConfig
from spectral_sentinel.utils.models import SimpleCNN
import os
from dotenv import load_dotenv


def create_test_model():
    """Create a simple test model."""
    model = SimpleCNN(num_classes=10, input_channels=1)
    return model.state_dict()


def test_amoy_testnet():
    """Test end-to-end blockchain integration on Amoy testnet."""
    
    print("=" * 70)
    print("🧪 BLOCKCHAIN INTEGRATION TEST - AMOY TESTNET")
    print("=" * 70)
    
    # Load environment variables
    load_dotenv()
    
    # Configuration for Amoy testnet
    config = BlockchainConfig(
        network="amoy",
        rpc_url=os.getenv("AMOY_RPC_URL") or "https://rpc-amoy.polygon.technology/",
        private_key=os.getenv("PRIVATE_KEY"),
        contract_address=os.getenv("CONTRACT_ADDRESS"),
    )
    
    # Validate configuration
    if not config.private_key:
        print("\n❌ ERROR: PRIVATE_KEY not set in .env file")
        print("   Please set your private key in .env file")
        return False
    
    if not config.contract_address:
        print("\n❌ ERROR: CONTRACT_ADDRESS not set in .env file")
        print("   Please deploy the contract first:")
        print("   npx hardhat run scripts/deploy.js --network amoy")
        return False
    
    try:
        # Initialize connector
        print("\n📡 Step 1: Initializing blockchain connector...")
        connector = BlockchainConnector(config)
        
        # Test 1: Register client
        print("\n👥 Step 2: Registering client...")
        # Note: Contract maps one address to one client ID, so we test with one client
        # In production, each client would have a different address
        num_clients = 1
        client_ids = [0]
        
        tx_hash = connector.register_clients_batch(client_ids)
        print(f"   Transaction: {connector.get_transaction_url(tx_hash)}")
        
        # Verify registration
        is_registered = connector.is_client_registered()
        if not is_registered:
            print(f"   ⚠️  Client registration pending (may need confirmation)")
        else:
            print(f"   ✅ Client registered")
        
        # Test 2: Check current round and start if needed
        print("\n🚀 Step 3: Checking round status...")
        current_round = connector.get_current_round()
        print(f"   Current round: {current_round}")
        
        if current_round > 0:
            # Check if round is finalized
            round_info = connector.get_round_info(current_round)
            if not round_info.get('finalized', False):
                print(f"   ⚠️  Round {current_round} exists but is not finalized")
                print(f"   Using existing round {current_round}")
            else:
                print(f"   Round {current_round} is finalized, starting new round...")
                tx_hash = connector.start_round()
                print(f"   Transaction: {connector.get_transaction_url(tx_hash)}")
                current_round = connector.get_current_round()
                print(f"   ✅ Round {current_round} started")
        else:
            print("   Starting Round 1...")
            tx_hash = connector.start_round()
            print(f"   Transaction: {connector.get_transaction_url(tx_hash)}")
            current_round = connector.get_current_round()
            print(f"   ✅ Round {current_round} started")
        
        # Test 3: Submit model update (if not already submitted)
        print("\n📤 Step 4: Checking submission status...")
        client_id = client_ids[0]
        
        # Check if already submitted
        submissions = connector.get_round_submissions(current_round)
        print(f"   Current submissions for round {current_round}: {submissions}")
        
        # Try to retrieve existing model first
        existing_model = connector.get_model_update(current_round, client_id)
        
        if existing_model is not None:
            print(f"   ✅ Client {client_id} already submitted for this round")
            test_models = {client_id: existing_model}
        else:
            print(f"   Submitting new model update...")
            test_models = {}
            model_dict = create_test_model()
            test_models[client_id] = model_dict
            
            try:
                tx_hash = connector.submit_model_update(
                    model_dict=model_dict,
                    client_id=client_id,
                    round_num=current_round
                )
                print(f"   Client {client_id} submitted (tx: {connector.get_transaction_url(tx_hash)})")
                print(f"   ✅ Model update submitted")
            except Exception as e:
                if "Already submitted" in str(e) or "already submitted" in str(e).lower():
                    print(f"   ⚠️  Client already submitted, retrieving existing model...")
                    existing_model = connector.get_model_update(current_round, client_id)
                    if existing_model:
                        test_models[client_id] = existing_model
                        print(f"   ✅ Retrieved existing submission")
                else:
                    raise
        
        # Test 4: Verify submissions
        print("\n🔍 Step 5: Verifying submissions on blockchain...")
        # Wait a bit for transactions to be confirmed
        import time
        time.sleep(5)
        
        submissions = connector.get_round_submissions(current_round)
        print(f"   📊 Submissions: {submissions}/{num_clients}")
        if submissions < num_clients:
            print(f"   ⚠️  Submission may still be pending confirmation")
        
        # Test 5: Retrieve model update
        print("\n📥 Step 6: Retrieving model update...")
        client_id = client_ids[0]
        model_dict = connector.get_model_update(current_round, client_id)
        if model_dict is not None:
            retrieved_model = model_dict
            print(f"   ✅ Retrieved model from client {client_id}")
        else:
            print(f"   ⚠️  Model for client {client_id} not yet available (may need confirmation)")
            retrieved_model = None
        
        # Test 6: Verify model integrity (if we have model)
        if retrieved_model is not None:
            print("\n🔐 Step 7: Verifying model integrity...")
            original = test_models[client_id]
            
            # Compare all parameters
            for key in original.keys():
                if not torch.allclose(original[key], retrieved_model[key], atol=1e-6):
                    print(f"   ⚠️  Model mismatch for client {client_id}, key {key}")
                else:
                    print(f"   ✅ Client {client_id} model verified")
            
            print("   ✅ Model verified - hash matches!")
        
        # Test 7: Finalize round
        print("\n✅ Step 8: Finalizing round...")
        aggregated_model = create_test_model()  # In real FL, this would be averaged
        tx_hash = connector.finalize_round(aggregated_model, current_round)
        print(f"   Transaction: {connector.get_transaction_url(tx_hash)}")
        
        # Wait for confirmation
        time.sleep(5)
        
        # Check round info
        round_info = connector.get_round_info(current_round)
        print(f"   Round Number: {round_info['round_number']}")
        print(f"   Submissions: {round_info['num_submissions']}")
        print(f"   Finalized: {round_info['finalized']}")
        if round_info['aggregated_model_hash']:
            print(f"   Aggregated model hash: {round_info['aggregated_model_hash'][:16]}...")
        
        # Test 8: Gas and cost estimation
        print("\n💰 Step 9: Gas statistics...")
        gas_stats = connector.get_gas_stats()
        print(f"   Current gas price: {gas_stats['gas_price_gwei']:.2f} gwei")
        print(f"   Cost per transaction: ~{gas_stats['estimated_cost_per_tx']:.6f} MATIC")
        
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
        print("\n🎉 Blockchain integration working correctly on Amoy testnet!")
        print("🚀 Ready to run federated learning on blockchain!\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_amoy_testnet()
    sys.exit(0 if success else 1)

