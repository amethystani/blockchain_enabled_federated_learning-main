"""
Federated Learning Client Simulation

Simulates honest and Byzantine clients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
import copy

from spectral_sentinel.attacks.attacks import BaseAttack
from spectral_sentinel.federated.data_loader import create_data_loader


class Client:
    """Base client class."""
    
    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 dataset: Dataset,
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 device: str = "cpu"):
        """
        Initialize client.
        
        Args:
            client_id: Unique client ID
            model: Neural network model
            dataset: Client's local dataset
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device to train on
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        
        # Create data loader
        self.data_loader = create_data_loader(dataset, batch_size, shuffle=True)
        
        # Statistics
        self.stats = {
            'total_epochs_trained': 0,
            'total_samples_seen': 0,
            'local_losses': []
        }
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Update local model with global parameters.
        
        Args:
            parameters: Model parameters dict
        """
        self.model.load_state_dict(parameters)
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return copy.deepcopy(self.model.state_dict())
    
    def local_train(self, num_epochs: int) -> Dict[str, torch.Tensor]:
        """
        Perform local training.
        
        Args:
            num_epochs: Number of local epochs
            
        Returns:
            Gradient dict
        """
        raise NotImplementedError("Subclass must implement local_train")
    
    def __repr__(self) -> str:
        return f"Client(id={self.client_id}, data_size={len(self.dataset)})"


class HonestClient(Client):
    """Honest client that trains normally."""
    
    def local_train(self, num_epochs: int) -> Dict[str, torch.Tensor]:
        """
        Train locally and return gradient.
        
        Args:
            num_epochs: Number of local epochs
            
        Returns:
            Gradient dict (change in parameters)
        """
        # Save initial parameters
        initial_params = self.get_model_parameters()
        
        # Set model to training mode
        self.model.train()
        
        # Create optimizer and loss
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                self.stats['total_samples_seen'] += len(data)
        
        self.stats['total_epochs_trained'] += num_epochs
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.stats['local_losses'].append(avg_loss)
        
        # Compute gradient (parameter change)
        final_params = self.get_model_parameters()
        gradient = {
            k: initial_params[k] - final_params[k]
            for k in initial_params.keys()
        }
        
        return gradient


class ByzantineClient(Client):
    """Byzantine client that performs attacks."""
    
    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 dataset: Dataset,
                 attack: BaseAttack,
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 device: str = "cpu",
                 num_classes: int = 10):
        """
        Initialize Byzantine client.
        
        Args:
            client_id: Client ID
            model: Model
            dataset: Local dataset
            attack: Attack strategy
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device
            num_classes: Number of classes (for label flipping)
        """
        super().__init__(client_id, model, dataset, learning_rate, batch_size, device)
        self.attack = attack
        self.num_classes = num_classes
    
    def local_train(self, 
                   num_epochs: int,
                   aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
                   honest_gradients: Optional[list] = None) -> Dict[str, torch.Tensor]:
        """
        Simulate local training and apply attack.
        
        For attacks like label flipping, we actually train with corrupted data.
        For other attacks, we compute honest gradient then modify it.
        
        Args:
            num_epochs: Number of epochs
            aggregated_gradient: Optional aggregated gradient from server
            honest_gradients: Optional list of honest gradients (for ALIE)
            
        Returns:
            Malicious gradient
        """
        # First compute what honest gradient would be
        if self.attack.name == 'LabelFlipAttack':
            # Train with flipped labels
            honest_gradient = self._train_with_label_flip(num_epochs)
        else:
            # Train normally to get honest gradient
            honest_gradient = self._train_normally(num_epochs)
        
        # Apply attack to gradient
        malicious_gradient = self.attack.apply(
            honest_gradient,
            aggregated_gradient=aggregated_gradient,
            honest_gradients_list=honest_gradients
        )
        
        return malicious_gradient
    
    def _train_normally(self, num_epochs: int) -> Dict[str, torch.Tensor]:
        """Train normally (as if honest)."""
        initial_params = self.get_model_parameters()
        
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        final_params = self.get_model_parameters()
        gradient = {
            k: initial_params[k] - final_params[k]
            for k in initial_params.keys()
        }
        
        return gradient
    
    def _train_with_label_flip(self, num_epochs: int) -> Dict[str, torch.Tensor]:
        """Train with flipped labels."""
        initial_params = self.get_model_parameters()
        
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Flip labels
                target = self.attack.flip_labels(target, self.num_classes)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        final_params = self.get_model_parameters()
        gradient = {
            k: initial_params[k] - final_params[k]
            for k in initial_params.keys()
        }
        
        return gradient
