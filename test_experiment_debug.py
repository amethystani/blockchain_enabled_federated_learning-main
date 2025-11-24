#!/usr/bin/env python3
"""Debug script to find where experiment hangs"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Step 1: Imports...")
import torch
import os
from dotenv import load_dotenv

print("Step 2: Loading env...")
load_dotenv()

print("Step 3: Importing blockchain...")
from spectral_sentinel.blockchain import BlockchainConnector, BlockchainConfig

print("Step 4: Creating blockchain config...")
config = BlockchainConfig(
    network="amoy",
    rpc_url=os.getenv("AMOY_RPC_URL") or "https://rpc-amoy.polygon.technology/",
    private_key=os.getenv("PRIVATE_KEY"),
    contract_address=os.getenv("CONTRACT_ADDRESS"),
)

print("Step 5: Connecting to blockchain...")
blockchain = BlockchainConnector(config)
print("✅ Blockchain connected!")

print("Step 6: Loading data...")
from spectral_sentinel.federated.data_loader import load_federated_data

print("Step 7: Loading MNIST...")
train_datasets, test_dataset = load_federated_data(
    dataset_name='mnist',
    num_clients=3,
    non_iid_alpha=0.5,
    data_dir='./data'
)
print(f"✅ Data loaded! Train datasets: {len(train_datasets)}, Test: {len(test_dataset)}")

print("Step 8: Creating model...")
from spectral_sentinel.utils.models import SimpleCNN
model = SimpleCNN(num_classes=10, input_channels=1)
print("✅ Model created!")

print("Step 9: Creating aggregator...")
from spectral_sentinel.aggregators.baselines import FedAvgAggregator
aggregator = FedAvgAggregator()
print("✅ Aggregator created!")

print("Step 10: Creating server...")
from spectral_sentinel.federated.server import FederatedServer
server = FederatedServer(
    model=model,
    aggregator=aggregator,
    test_dataset=test_dataset,
    device='cpu',
    batch_size=32
)
print("✅ Server created!")

print("\n✅ All components initialized successfully!")

