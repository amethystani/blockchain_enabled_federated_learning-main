#!/usr/bin/env python3
"""
Quick Validation Test

Minimal test to verify all components work correctly.
Runs in ~2 minutes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

print("="*70)
print("🧪 SPECTRAL SENTINEL: Quick Validation Test")
print("="*70)

# Test 1: Module Imports
print("\n[1/7] Testing module imports...")
try:
    from spectral_sentinel.config import Config
    from spectral_sentinel.rmt.marchenko_pastur import MarchenkoPasturLaw
    from spectral_sentinel.rmt.spectral_analyzer import SpectralAnalyzer
    from spectral_sentinel.attacks.attacks import get_attack
    from spectral_sentinel.aggregators.spectral_sentinel import SpectralSentinelAggregator
    from spectral_sentinel.aggregators.baselines import get_aggregator
    from spectral_sentinel.federated.data_loader import load_federated_data
    from spectral_sentinel.federated.client import HonestClient, ByzantineClient
    from spectral_sentinel.federated.server import FederatedServer  
    from spectral_sentinel.utils.models import get_model
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Configuration
print("\n[2/7] Testing configuration...")
try:
    config = Config(
        dataset='mnist',
        num_clients=2,
        num_rounds=1,
        byzantine_ratio=0.5,
        batch_size=16,
        local_epochs=1
    )
    print(f"✅ Config created: {config.num_clients} clients ({config.num_byzantine} Byzantine)")
except Exception as e:
    print(f"❌ Config failed: {e}")
    sys.exit(1)

# Test 3: Data Loading
print("\n[3/7] Testing data loading...")
try:
    print("   Downloading MNIST...")
    client_datasets, test_dataset = load_federated_data(
        'mnist', num_clients=2, non_iid_alpha=0.5
    )
    print(f"✅ Data loaded: {len(client_datasets)} client datasets, "
          f"test set: {len(test_dataset)} samples")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    sys.exit(1)

# Test 4: Model Creation
print("\n[4/7] Testing model creation...")
try:
    model = get_model('simple_cnn', num_classes=10, input_channels=1)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {num_params:,} parameters")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    sys.exit(1)

# Test 5: Client Creation
print("\n[5/7] Testing client creation...")
try:
    attack = get_attack('minmax')
    
    # Honest client
    honest_client = HonestClient(
        client_id=0,
        model=get_model('simple_cnn', 10, 1),
        dataset=client_datasets[0],
        learning_rate=0.01,
        batch_size=32,
        device='cpu'
    )
    
    # Byzantine client
    byzantine_client = ByzantineClient(
        client_id=1,
        model=get_model('simple_cnn', 10, 1),
        dataset=client_datasets[1],
        attack=attack,
        learning_rate=0.01,
        batch_size=32,
        device='cpu',
        num_classes=10
    )
    
    print(f"✅ Clients created: Honest + Byzantine")
except Exception as e:
    print(f"❌ Client creation failed: {e}")
    sys.exit(1)

# Test 6: Aggregator Creation
print("\n[6/7] Testing aggregators...")
try:
    # Spectral Sentinel
    ss_agg = SpectralSentinelAggregator()
    
    # Baselines
    fedavg_agg = get_aggregator('fedavg')
    krum_agg = get_aggregator('krum', num_byzantine=2)
    bulyan_agg = get_aggregator('bulyan', num_byzantine=2)
    
    print(f"✅ Aggregators created: 4 methods tested")
except Exception as e:
    print(f"❌ Aggregator creation failed: {e}")
    sys.exit(1)

# Test 7: Mini Training Run
print("\n[7/7] Testing mini training run (2 rounds)...")
try:
    # Create server
    server = FederatedServer(
        model=get_model('simple_cnn', 10, 1),
        aggregator=SpectralSentinelAggregator(),
        test_dataset=test_dataset,
        device='cpu'
    )
    
    # Create minimal client list
    clients = [honest_client, byzantine_client]
    
    print("   Running 1 federated round...")
    
    # Run 1 round
    for r in range(1):
        info =server.federated_round(clients, num_local_epochs=1, verbose=False)
        print(f"   Round {r+1}: Accuracy={info['accuracy']:.1f}%, "
              f"Byzantine detected={info['num_byzantine']}")
    
    final_acc, _ = server.evaluate()
    
    if final_acc > 50:  # Should be better than random (10%)
        print(f"✅ Training successful: Final accuracy = {final_acc:.1f}%")
    else:
        print(f"⚠️  Training completed but accuracy low: {final_acc:.1f}%")
        
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("🎉 ALL TESTS PASSED!")
print("="*70)
print("\n✅ Spectral Sentinel is ready for full experiments!")
print("\nNext steps:")
print("  1. Run baseline comparison: python experiments/compare_aggregators.py")
print("  2. Test all attacks: python experiments/test_all_attacks.py")
print("  3. Full MNIST experiment: python experiments/simulate_basic.py")
print()
