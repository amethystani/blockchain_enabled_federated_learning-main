"""
Federated Learning Server

Coordinates federated training rounds and aggregates client updates.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import copy
import time
from tqdm import tqdm

from spectral_sentinel.federated.client import Client, HonestClient, ByzantineClient
from spectral_sentinel.aggregators.base_aggregator import BaseAggregator
from spectral_sentinel.federated.data_loader import create_data_loader


class FederatedServer:
    """
    Federated learning server coordinator.
    
    Manages:
    - Global model
    - Client selection and coordination
    - Gradient aggregation
    - Model evaluation
    """
    
    def __init__(self,
                 model: nn.Module,
                 aggregator: BaseAggregator,
                 test_dataset: Dataset,
                 device: str = "cpu",
                 batch_size: int = 128):
        """
        Initialize federated server.
        
        Args:
            model: Global model
            aggregator: Aggregation method
            test_dataset: Global test dataset
            device: Device
            batch_size: Batch size for evaluation
        """
        self.model = model.to(device)
        self.aggregator = aggregator
        self.test_dataset = test_dataset
        self.device = device
        self.batch_size = batch_size
        
        # Test data loader
        self.test_loader = create_data_loader(test_dataset, batch_size, shuffle=False)
        
        # Training statistics
        self.stats = {
            'round_accuracies': [],
            'round_losses': [],
            'round_times': [],
            'byzantine_detected_per_round': [],
            'aggregation_info': []
        }
        
        self.current_round = 0
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters."""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Update global model parameters."""
        self.model.load_state_dict(parameters)
    
    def federated_round(self,
                       clients: List[Client],
                       num_local_epochs: int = 5,
                       verbose: bool = True) -> Dict:
        """
        Execute one round of federated learning.
        
        Args:
            clients: List of participating clients
            num_local_epochs: Number of local training epochs
            verbose: Print progress
            
        Returns:
            Round information dict
        """
        round_start = time.time()
        
        # 1. Broadcast current model to all clients
        global_params = self.get_model_parameters()
        for client in clients:
            client.set_model_parameters(global_params)
        
        # 2. Clients perform local training
        gradients = []
        client_ids = []
        
        if verbose:
            print(f"\n🔄 Round {self.current_round + 1}: Local training...")
        
        honest_gradients = []  # For ALIE attack coordination
        
        for client in clients:
            if isinstance(client, HonestClient):
                gradient = client.local_train(num_local_epochs)
                honest_gradients.append(gradient)
            elif isinstance(client, ByzantineClient):
                # Byzantine clients need info about honest gradients (for ALIE)
                gradient = client.local_train(
                    num_local_epochs,
                    aggregated_gradient=None,  # Could provide previous round's aggregate
                    honest_gradients=honest_gradients if len(honest_gradients) > 0 else None
                )
            else:
                gradient = client.local_train(num_local_epochs)
            
            gradients.append(gradient)
            client_ids.append(client.client_id)
        
        # 3. Aggregate gradients using chosen method
        if verbose:
            print(f"🔍 Aggregating {len(gradients)} gradients using {self.aggregator.name}...")
        
        aggregated_gradient, agg_info = self.aggregator.aggregate(
            gradients, client_ids
        )
        
        # 4. Update global model
        current_params = self.get_model_parameters()
        updated_params = {
            k: current_params[k] - aggregated_gradient[k]  # Subtract gradient
            for k in current_params.keys()
        }
        self.set_model_parameters(updated_params)
        
        # 5. Evaluate model
        test_accuracy, test_loss = self.evaluate()
        
        # Record statistics
        round_time = time.time() - round_start
        self.stats['round_accuracies'].append(test_accuracy)
        self.stats['round_losses'].append(test_loss)
        self.stats['round_times'].append(round_time)
        self.stats['byzantine_detected_per_round'].append(
            agg_info.get('num_byzantine', 0)
        )
        self.stats['aggregation_info'].append(agg_info)
        
        if verbose:
            print(f"✅ Round {self.current_round + 1} complete:")
            print(f"   Accuracy: {test_accuracy:.2f}%, Loss: {test_loss:.4f}")
            print(f"   Byzantine detected: {agg_info.get('num_byzantine', 0)}/{len(clients)}")
            print(f"   Time: {round_time:.2f}s")
        
        self.current_round += 1
        
        return {
            'round': self.current_round,
            'accuracy': test_accuracy,
            'loss': test_loss,
            'time': round_time,
            **agg_info
        }
    
    def train(self,
             clients: List[Client],
             num_rounds: int,
             num_local_epochs: int = 5,
             clients_per_round: Optional[int] = None,
             verbose: bool = True) -> Dict:
        """
        Run full federated training.
        
        Args:
            clients: All available clients
            num_rounds: Number of federated rounds
            num_local_epochs: Local epochs per round
            clients_per_round: Number of clients to sample each round (None = all)
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        import random
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 Starting Federated Learning")
            print(f"{'='*60}")
            print(f"Total clients: {len(clients)}")
            print(f"Honest: {sum(isinstance(c, HonestClient) for c in clients)}")
            print(f"Byzantine: {sum(isinstance(c, ByzantineClient) for c in clients)}")
            print(f"Rounds: {num_rounds}")
            print(f"Local epochs: {num_local_epochs}")
            print(f"Aggregator: {self.aggregator.name}")
            print(f"{'='*60}\n")
        
        # Training loop
        for round_num in range(num_rounds):
            # Sample clients if needed
            if clients_per_round is not None and clients_per_round < len(clients):
                selected_clients = random.sample(clients, clients_per_round)
            else:
                selected_clients = clients
            
            # Run round
            round_info = self.federated_round(
                selected_clients,
                num_local_epochs,
                verbose=verbose
            )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Training Complete!")
            print(f"{'='*60}")
            print(f"Final accuracy: {self.stats['round_accuracies'][-1]:.2f}%")
            print(f"Final loss: {self.stats['round_losses'][-1]:.4f}")
            print(f"Total time: {sum(self.stats['round_times']):.2f}s")
            print(f"{'='*60}\n")
        
        return self.get_statistics()
    
    def evaluate(self) -> tuple:
        """
        Evaluate global model on test set.
        
        Returns:
            (accuracy, loss)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        return accuracy, avg_loss
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        import numpy as np
        
        stats = self.stats.copy()
        
        # Add aggregator stats
        stats['aggregator_stats'] = self.aggregator.get_statistics()
        
        # Compute summary statistics
        if len(self.stats['round_accuracies']) > 0:
            stats['max_accuracy'] = max(self.stats['round_accuracies'])
            stats['final_accuracy'] = self.stats['round_accuracies'][-1]
            stats['avg_accuracy'] = np.mean(self.stats['round_accuracies'])
        
        if len(self.stats['round_losses']) > 0:
            stats['min_loss'] = min(self.stats['round_losses'])
            stats['final_loss'] = self.stats['round_losses'][-1]
        
        if len(self.stats['byzantine_detected_per_round']) > 0:
            stats['total_byzantine_detected'] = sum(self.stats['byzantine_detected_per_round'])
            stats['avg_byzantine_per_round'] = np.mean(self.stats['byzantine_detected_per_round'])
        
        return stats
    
    def reset_statistics(self):
        """Reset server and aggregator statistics."""
        self.stats = {
            'round_accuracies': [],
            'round_losses': [],
            'round_times': [],
            'byzantine_detected_per_round': [],
            'aggregation_info': []
        }
        self.aggregator.reset_statistics()
        self.current_round = 0
