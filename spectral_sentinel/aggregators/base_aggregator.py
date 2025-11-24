"""
Base Aggregator Interface

All aggregation methods inherit from this base class.
"""

import torch
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import time


class BaseAggregator(ABC):
    """
    Base class for all gradient aggregation methods.
    
    Aggregators take gradients from multiple clients and combine them
    into a single update for the global model.
    """
    
    def __init__(self, name: str = "BaseAggregator"):
        """
        Initialize aggregator.
        
        Args:
            name: Name of aggregator
        """
        self.name = name
        
        # Statistics tracking
        self.stats = {
            'total_rounds': 0,
            'total_clients_processed': 0,
            'total_clients_rejected': 0,
            'aggregation_times': [],
            'detection_rate': [],  # For Byzantine detection
        }
    
    @abstractmethod
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Aggregate gradients from multiple clients.
        
        Args:
            gradients: List of gradient dicts, one per client
            client_ids: Optional client IDs
            **kwargs: Aggregator-specific parameters
            
        Returns:
            (aggregated_gradient, info_dict)
            - aggregated_gradient: Combined gradient
            - info_dict: Detection results, rejected clients, etc.
        """
        pass
    
    def _average_gradients(self, 
                          gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Simple averaging of gradients (FedAvg).
        
        Args:
            gradients: List of gradient dicts
            
        Returns:
            Averaged gradient
        """
        if len(gradients) == 0:
            raise ValueError("Cannot average empty gradient list")
        
        # Initialize with zeros
        avg_grad = {
            k: torch.zeros_like(v)
            for k, v in gradients[0].items()
        }
        
        # Sum all gradients
        for grad in gradients:
            for k in avg_grad.keys():
                avg_grad[k] += grad[k]
        
        # Divide by count
        n = len(gradients)
        for k in avg_grad.keys():
            avg_grad[k] /= n
        
        return avg_grad
    
    def _clip_gradients(self,
                       gradients: List[Dict[str, torch.Tensor]],
                       clip_norm: float) -> List[Dict[str, torch.Tensor]]:
        """
        Clip gradients by norm.
        
        Args:
            gradients: List of gradient dicts
            clip_norm: Maximum L2 norm
            
        Returns:
            Clipped gradients
        """
        clipped = []
        
        for grad in gradients:
            # Compute total norm
            total_norm = torch.sqrt(sum(
                torch.sum(v ** 2) for v in grad.values()
            ))
            
            # Clip if necessary
            if total_norm > clip_norm:
                scale = clip_norm / (total_norm + 1e-8)
                clipped_grad = {
                    k: scale * v for k, v in grad.items()
                }
            else:
                clipped_grad = grad
            
            clipped.append(clipped_grad)
        
        return clipped
    
    def update_statistics(self,
                         num_clients: int,
                         num_rejected: int,
                         aggregation_time: float,
                         **kwargs):
        """
        Update aggregator statistics.
        
        Args:
            num_clients: Number of clients processed
            num_rejected: Number of clients rejected
            aggregation_time: Time taken for aggregation
            **kwargs: Additional statistics
        """
        self.stats['total_rounds'] += 1
        self.stats['total_clients_processed'] += num_clients
        self.stats['total_clients_rejected'] += num_rejected
        self.stats['aggregation_times'].append(aggregation_time)
        
        if num_clients > 0:
            detection_rate = num_rejected / num_clients
            self.stats['detection_rate'].append(detection_rate)
    
    def get_statistics(self) -> Dict:
        """Get aggregator statistics."""
        import numpy as np
        
        stats = self.stats.copy()
        
        # Compute averages
        if len(self.stats['aggregation_times']) > 0:
            stats['avg_aggregation_time'] = np.mean(self.stats['aggregation_times'])
            stats['total_aggregation_time'] = np.sum(self.stats['aggregation_times'])
        
        if len(self.stats['detection_rate']) > 0:
            stats['avg_detection_rate'] = np.mean(self.stats['detection_rate'])
        
        return stats
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'total_rounds': 0,
            'total_clients_processed': 0,
            'total_clients_rejected': 0,
            'aggregation_times': [],
            'detection_rate': [],
        }
    
    def __repr__(self) -> str:
        return f"{self.name}Aggregator"


class AggregatorTimer:
    """Context manager for timing aggregation."""
    
    def __init__(self, aggregator: BaseAggregator):
        self.aggregator = aggregator
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        return elapsed
