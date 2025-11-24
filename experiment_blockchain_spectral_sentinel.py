#!/usr/bin/env python3
"""
On-Chain Spectral Sentinel vs Baseline Comparison Experiment

This experiment runs federated learning on Amoy testnet blockchain and compares:
- Spectral Sentinel (Byzantine-robust via RMT)
- FedAvg (baseline - no defense)
- Krum (baseline - distance-based)
- Geometric Median (baseline - robust to outliers)

All results are logged for research paper documentation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

from spectral_sentinel.blockchain import BlockchainConnector, BlockchainConfig
from spectral_sentinel.federated.server import FederatedServer
from spectral_sentinel.federated.client import HonestClient, ByzantineClient
from spectral_sentinel.federated.data_loader import create_data_loader, load_federated_data
from spectral_sentinel.aggregators.spectral_sentinel import SpectralSentinelAggregator
from spectral_sentinel.aggregators.baselines import (
    FedAvgAggregator, KrumAggregator, GeometricMedianAggregator
)
from spectral_sentinel.utils.models import SimpleCNN


class BlockchainFederatedExperiment:
    """
    Federated learning experiment with blockchain integration.
    
    Runs FL rounds on-chain and compares different aggregators.
    """
    
    def __init__(self,
                 aggregator_name: str,
                 num_clients: int = 5,
                 num_rounds: int = 10,
                 byzantine_ratio: float = 0.2,
                 local_epochs: int = 1,
                 batch_size: int = 32,
                 device: str = "cpu"):
        """
        Initialize experiment.
        
        Args:
            aggregator_name: Name of aggregator ('spectral_sentinel', 'fedavg', 'krum', 'geometric_median')
            num_clients: Number of federated clients
            num_rounds: Number of FL rounds
            byzantine_ratio: Fraction of Byzantine clients
            local_epochs: Local training epochs per round
            batch_size: Batch size for training
            device: Device ('cpu' or 'cuda')
        """
        self.aggregator_name = aggregator_name
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.byzantine_ratio = byzantine_ratio
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = device
        
        # Load blockchain config
        load_dotenv()
        self.blockchain_config = BlockchainConfig(
            network="amoy",
            rpc_url=os.getenv("AMOY_RPC_URL") or "https://rpc-amoy.polygon.technology/",
            private_key=os.getenv("PRIVATE_KEY"),
            contract_address=os.getenv("CONTRACT_ADDRESS"),
        )
        
        # Initialize blockchain connector
        print(f"🔗 Connecting to blockchain...")
        self.blockchain = BlockchainConnector(self.blockchain_config)
        
        # Check balance
        balance_wei = self.blockchain.w3.eth.get_balance(self.blockchain.account.address)
        balance_matic = self.blockchain.w3.from_wei(balance_wei, 'ether')
        print(f"💰 Wallet balance: {balance_matic:.4f} MATIC")
        
        # Estimate required gas
        gas_stats = self.blockchain.get_gas_stats()
        estimated_tx_cost = gas_stats['estimated_cost_per_tx']
        # Estimate: registration + (rounds * (start_round + clients * submit + finalize))
        estimated_total = estimated_tx_cost * (1 + self.num_rounds * (1 + self.num_clients + 1))
        print(f"💰 Estimated total cost: {estimated_total:.6f} MATIC")
        
        if balance_matic < estimated_total * 1.5:  # 1.5x safety margin
            print(f"⚠️  WARNING: Balance may be insufficient!")
            print(f"   Current: {balance_matic:.6f} MATIC")
            print(f"   Estimated: {estimated_total:.6f} MATIC")
            print(f"   Recommended: {estimated_total * 1.5:.6f} MATIC")
            print(f"\n💡 Get more POL from: https://faucet.polygon.technology/")
            response = input("\nContinue anyway? (yes/no): ")
            if response.lower() != 'yes':
                raise ValueError("Insufficient balance. Please add more funds.")
        
        # Initialize model
        self.model = SimpleCNN(num_classes=10, input_channels=1)
        
        # Load dataset
        print(f"📊 Loading MNIST dataset...")
        train_datasets, test_dataset = load_federated_data(
            dataset_name='mnist',
            num_clients=num_clients,
            non_iid_alpha=0.5,  # Non-IID distribution
            data_dir='./data'
        )
        self.test_dataset = test_dataset
        
        # Create aggregator
        self.aggregator = self._create_aggregator()
        
        # Initialize server
        self.server = FederatedServer(
            model=self.model,
            aggregator=self.aggregator,
            test_dataset=test_dataset,
            device=device,
            batch_size=batch_size
        )
        
        # Create clients
        self.clients = self._create_clients(train_datasets)
        
        # Results storage
        self.results = {
            'aggregator': aggregator_name,
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'byzantine_ratio': byzantine_ratio,
            'round_accuracies': [],
            'round_losses': [],
            'round_times': [],
            'byzantine_detected_per_round': [],
            'blockchain_transactions': [],
            'gas_costs': [],
            'start_time': datetime.now().isoformat(),
            'blockchain_info': {
                'network': self.blockchain_config.network,
                'contract_address': self.blockchain_config.contract_address,
                'wallet_address': self.blockchain.account.address
            }
        }
    
    def _create_aggregator(self):
        """Create aggregator based on name."""
        if self.aggregator_name == 'spectral_sentinel':
            return SpectralSentinelAggregator(
                ks_threshold=0.05,
                tail_threshold=0.1,
                use_sketching=False
            )
        elif self.aggregator_name == 'fedavg':
            return FedAvgAggregator()
        elif self.aggregator_name == 'krum':
            num_byzantine = int(self.num_clients * self.byzantine_ratio)
            return KrumAggregator(num_byzantine=num_byzantine)
        elif self.aggregator_name == 'geometric_median':
            return GeometricMedianAggregator(max_iter=10, tol=1e-5)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator_name}")
    
    def _create_clients(self, train_datasets):
        """Create federated clients (mix of honest and Byzantine)."""
        clients = []
        num_byzantine = int(self.num_clients * self.byzantine_ratio)
        
        for i in range(self.num_clients):
            dataset = train_datasets[i]
            is_byzantine = i < num_byzantine
            
            if is_byzantine:
                client = ByzantineClient(
                    client_id=i,
                    dataset=dataset,
                    model=SimpleCNN(num_classes=10, input_channels=1),
                    device=self.device,
                    attack_type='minmax'  # Min-max attack
                )
            else:
                client = HonestClient(
                    client_id=i,
                    dataset=dataset,
                    model=SimpleCNN(num_classes=10, input_channels=1),
                    device=self.device
                )
            
            clients.append(client)
        
        return clients
    
    def run_experiment(self) -> Dict:
        """
        Run complete federated learning experiment on blockchain.
        
        Returns:
            Results dictionary with all metrics
        """
        print(f"\n{'='*70}")
        print(f"🚀 STARTING ON-CHAIN EXPERIMENT: {self.aggregator_name.upper()}")
        print(f"{'='*70}")
        print(f"Aggregator: {self.aggregator_name}")
        print(f"Clients: {self.num_clients} (Byzantine: {int(self.num_clients * self.byzantine_ratio)})")
        print(f"Rounds: {self.num_rounds}")
        print(f"Network: {self.blockchain_config.network}")
        print(f"Contract: {self.blockchain_config.contract_address[:20]}...")
        print(f"{'='*70}\n")
        
        # Register clients on blockchain (skip if already registered)
        print("👥 Checking client registration...")
        client_ids = [c.client_id for c in self.clients]
        
        # Check if clients are already registered
        already_registered = True
        for client_id in client_ids:
            if not self.blockchain.is_client_registered():
                already_registered = False
                break
        
        if not already_registered:
            print(f"   Registering {len(client_ids)} clients on blockchain...")
            try:
                tx_hash = self.blockchain.register_clients_batch(client_ids)
                self.results['blockchain_transactions'].append({
                    'type': 'register_clients',
                    'tx_hash': tx_hash,
                    'url': self.blockchain.get_transaction_url(tx_hash),
                    'timestamp': datetime.now().isoformat()
                })
                print(f"   ✅ Registered {len(client_ids)} clients")
            except Exception as e:
                if "insufficient funds" in str(e).lower():
                    raise
                print(f"   ⚠️  Registration issue (may already be registered): {e}")
        else:
            print(f"   ✅ Clients already registered")
        
        # Run federated learning rounds
        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'─'*70}")
            print(f"📊 ROUND {round_num}/{self.num_rounds}")
            print(f"{'─'*70}")
            
            round_start_time = time.time()
            
            # Start round on blockchain
            try:
                current_round = self.blockchain.get_current_round()
                if current_round == 0 or self._is_round_finalized(current_round):
                    print(f"   🚀 Starting new round on blockchain...")
                    tx_hash = self.blockchain.start_round()
                    current_round = self.blockchain.get_current_round()
                    self.results['blockchain_transactions'].append({
                        'type': 'start_round',
                        'round': current_round,
                        'tx_hash': tx_hash,
                        'url': self.blockchain.get_transaction_url(tx_hash),
                        'timestamp': datetime.now().isoformat()
                    })
                    print(f"   ✅ Round {current_round} started")
                else:
                    print(f"   ℹ️  Using existing round {current_round}")
            except Exception as e:
                print(f"   ⚠️  Round start issue: {e}")
                current_round = round_num
            
            # Run federated round (off-chain computation)
            print(f"   🏋️  Running federated learning round...")
            print(f"   DEBUG: About to call server.federated_round with {len(self.clients)} clients")
            try:
                round_info = self.server.federated_round(
                    clients=self.clients,
                    num_local_epochs=self.local_epochs,
                    verbose=False
                )
                print(f"   DEBUG: Round completed, got round_info")
            except Exception as e:
                print(f"   ❌ Federated round failed: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Submit model updates to blockchain
            print(f"   📤 Submitting model updates to blockchain...")
            submitted_count = 0
            for client in self.clients:
                try:
                    # Get client's model update
                    client_model = client.get_model_parameters()
                    
                    # Submit to blockchain
                    tx_hash = self.blockchain.submit_model_update(
                        model_dict=client_model,
                        client_id=client.client_id,
                        round_num=current_round
                    )
                    
                    self.results['blockchain_transactions'].append({
                        'type': 'submit_update',
                        'round': current_round,
                        'client_id': client.client_id,
                        'tx_hash': tx_hash,
                        'url': self.blockchain.get_transaction_url(tx_hash),
                        'timestamp': datetime.now().isoformat()
                    })
                    submitted_count += 1
                except Exception as e:
                    print(f"   ⚠️  Client {client.client_id} submission failed: {e}")
            
            print(f"   ✅ {submitted_count}/{len(self.clients)} updates submitted")
            
            # Wait for blockchain confirmations
            import time as time_module
            time_module.sleep(3)
            
            # Retrieve models from blockchain and aggregate
            print(f"   📥 Retrieving models from blockchain...")
            retrieved_models = []
            for client in self.clients:
                try:
                    model_dict = self.blockchain.get_model_update(current_round, client.client_id)
                    if model_dict:
                        retrieved_models.append(model_dict)
                except Exception as e:
                    print(f"   ⚠️  Failed to retrieve model for client {client.client_id}: {e}")
            
            # If we have models, aggregate them
            if len(retrieved_models) > 0:
                # Convert to gradients format for aggregator
                global_model = self.server.get_model_parameters()
                gradients = []
                for model_dict in retrieved_models:
                    grad = {}
                    for key in global_model.keys():
                        grad[key] = model_dict[key] - global_model[key]
                    gradients.append(grad)
                
                # Aggregate using the aggregator
                aggregated_grad, agg_info = self.aggregator.aggregate(
                    gradients=gradients,
                    client_ids=[c.client_id for c in self.clients[:len(retrieved_models)]]
                )
                
                # Update global model
                for key in global_model.keys():
                    global_model[key] = global_model[key] + aggregated_grad[key]
                self.server.set_model_parameters(global_model)
                
                # Log aggregation info
                byzantine_detected = agg_info.get('num_byzantine', 0)
                print(f"   🔍 Byzantine detected: {byzantine_detected}/{len(gradients)}")
                self.results['byzantine_detected_per_round'].append(byzantine_detected)
            else:
                # Fallback: use server's aggregated model
                print(f"   ⚠️  No models retrieved, using server aggregation")
                byzantine_detected = round_info.get('byzantine_detected', 0)
                self.results['byzantine_detected_per_round'].append(byzantine_detected)
            
            # Evaluate model
            accuracy, loss = self.server.evaluate()
            print(f"   📈 Accuracy: {accuracy:.2f}% | Loss: {loss:.4f}")
            
            # Finalize round on blockchain
            try:
                aggregated_model = self.server.get_model_parameters()
                tx_hash = self.blockchain.finalize_round(aggregated_model, current_round)
                self.results['blockchain_transactions'].append({
                    'type': 'finalize_round',
                    'round': current_round,
                    'tx_hash': tx_hash,
                    'url': self.blockchain.get_transaction_url(tx_hash),
                    'timestamp': datetime.now().isoformat()
                })
                print(f"   ✅ Round {current_round} finalized on blockchain")
            except Exception as e:
                print(f"   ⚠️  Round finalization issue: {e}")
            
            # Record metrics
            round_time = time.time() - round_start_time
            self.results['round_accuracies'].append(accuracy)
            self.results['round_losses'].append(loss)
            self.results['round_times'].append(round_time)
            
            # Get gas stats
            gas_stats = self.blockchain.get_gas_stats()
            self.results['gas_costs'].append({
                'round': current_round,
                'gas_price_gwei': gas_stats['gas_price_gwei'],
                'estimated_cost': gas_stats['estimated_cost_per_tx']
            })
        
        # Final evaluation
        print(f"\n{'='*70}")
        print(f"📊 FINAL EVALUATION")
        print(f"{'='*70}")
        final_accuracy, final_loss = self.server.evaluate()
        print(f"Final Accuracy: {final_accuracy:.2f}%")
        print(f"Final Loss: {final_loss:.4f}")
        
        # Calculate statistics
        self.results['final_accuracy'] = final_accuracy
        self.results['final_loss'] = final_loss
        self.results['max_accuracy'] = max(self.results['round_accuracies']) if self.results['round_accuracies'] else 0
        self.results['avg_byzantine_detected'] = np.mean(self.results['byzantine_detected_per_round']) if self.results['byzantine_detected_per_round'] else 0
        self.results['total_transactions'] = len(self.results['blockchain_transactions'])
        self.results['total_gas_cost'] = sum(g['estimated_cost'] for g in self.results['gas_costs'])
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_time'] = sum(self.results['round_times'])
        
        # Aggregator statistics
        agg_stats = self.aggregator.get_statistics()
        self.results['aggregator_stats'] = agg_stats
        
        return self.results
    
    def _is_round_finalized(self, round_num: int) -> bool:
        """Check if round is finalized."""
        try:
            round_info = self.blockchain.get_round_info(round_num)
            return round_info.get('finalized', False)
        except:
            return False
    
    def save_results(self, output_dir: str = "./results/blockchain_experiments"):
        """Save results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.aggregator_name}_onchain_{timestamp}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filepath}")
        return filepath


def run_comparison_experiment():
    """Run comparison experiment for all aggregators."""
    
    # Aggregators to compare
    aggregators = [
        'spectral_sentinel',
        'fedavg',
        'krum',
        'geometric_median'
    ]
    
    # Experiment configuration
    config = {
        'num_clients': 5,
        'num_rounds': 5,  # Reduced for testnet (gas costs)
        'byzantine_ratio': 0.2,
        'local_epochs': 1,
        'batch_size': 32,
        'device': 'cpu'
    }
    
    print("="*70)
    print("🔬 ON-CHAIN SPECTRAL SENTINEL vs BASELINE COMPARISON")
    print("="*70)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"\nAggregators to test: {', '.join(aggregators)}")
    print("="*70)
    
    all_results = {}
    
    # Run each aggregator
    for i, agg_name in enumerate(aggregators, 1):
        print(f"\n\n{'#'*70}")
        print(f"# [{i}/{len(aggregators)}] Testing {agg_name.upper()}")
        print(f"{'#'*70}\n")
        
        try:
            # Create and run experiment
            experiment = BlockchainFederatedExperiment(
                aggregator_name=agg_name,
                **config
            )
            
            results = experiment.run_experiment()
            all_results[agg_name] = results
            
            # Save individual results
            experiment.save_results()
            
            print(f"\n✅ {agg_name} completed!")
            print(f"   Final Accuracy: {results['final_accuracy']:.2f}%")
            print(f"   Max Accuracy: {results['max_accuracy']:.2f}%")
            print(f"   Avg Byzantine Detected: {results['avg_byzantine_detected']:.2f}")
            print(f"   Total Transactions: {results['total_transactions']}")
            print(f"   Total Gas Cost: {results['total_gas_cost']:.6f} MATIC")
            
        except Exception as e:
            print(f"\n❌ {agg_name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[agg_name] = {'error': str(e)}
    
    # Save comparison summary
    summary_path = Path("./results/blockchain_experiments/comparison_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'config': config,
        'aggregators': aggregators,
        'results': all_results,
        'comparison': {
            'best_accuracy': max([r.get('final_accuracy', 0) for r in all_results.values() if 'error' not in r]),
            'best_aggregator': max(all_results.items(), 
                                 key=lambda x: x[1].get('final_accuracy', 0) if 'error' not in x[1] else 0)[0]
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n\n{'='*70}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    for agg_name, results in all_results.items():
        if 'error' not in results:
            print(f"{agg_name.upper():20s} | Accuracy: {results['final_accuracy']:6.2f}% | "
                  f"Byzantine Detected: {results['avg_byzantine_detected']:5.2f} | "
                  f"Gas Cost: {results['total_gas_cost']:8.6f} MATIC")
    
    print(f"\n💾 Summary saved to: {summary_path}")
    print(f"{'='*70}\n")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="On-chain Spectral Sentinel experiment")
    parser.add_argument('--aggregator', type=str, default=None,
                       choices=['spectral_sentinel', 'fedavg', 'krum', 'geometric_median'],
                       help='Run single aggregator (default: run all)')
    parser.add_argument('--rounds', type=int, default=5, help='Number of FL rounds')
    parser.add_argument('--clients', type=int, default=5, help='Number of clients')
    
    args = parser.parse_args()
    
    if args.aggregator:
        # Run single aggregator
        experiment = BlockchainFederatedExperiment(
            aggregator_name=args.aggregator,
            num_clients=args.clients,
            num_rounds=args.rounds
        )
        results = experiment.run_experiment()
        experiment.save_results()
    else:
        # Run comparison
        run_comparison_experiment()

