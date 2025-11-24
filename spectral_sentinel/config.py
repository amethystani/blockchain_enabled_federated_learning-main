"""Configuration management for Spectral Sentinel experiments."""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class Config:
    """Central configuration for Spectral Sentinel experiments."""
    
    # Experiment settings
    exp_name: str = "spectral_sentinel_basic"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Federated Learning settings
    num_clients: int = 20
    num_rounds: int = 50
    clients_per_round: int = 20  # Use all clients each round
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Data settings
    dataset: str = "mnist"  # mnist, cifar10, cifar100
    num_classes: int = 10
    non_iid_alpha: float = 0.5  # Dirichlet concentration (lower = more skew)
    
    # Byzantine settings
    byzantine_ratio: float = 0.4  # Fraction of Byzantine clients
    attack_type: str = "minmax"  # minmax, labelflip, alie, adaptive
    
    # Spectral Sentinel settings
    use_sketching: bool = False  # Enable for large models
    sketch_size: int = 256  # k in O(k²) memory
    detection_frequency: int = 1  # Detect every N rounds (1 = every round)
    mp_threshold: float = 0.05  # KS test p-value threshold
    tail_threshold: float = 0.1  # Tail anomaly detection threshold
    sliding_window_size: int = 50  # For online MP tracking
    layer_wise: bool = False  # Enable layer-wise detection
    
    # Aggregator settings
    aggregator: str = "spectral_sentinel"  # spectral_sentinel, fedavg, krum, median, etc.
    clip_threshold: float = 0.15  # Gradient clipping for adaptive attacks
    
    # Phase transition analysis
    sigma_squared_f_squared: Optional[float] = None  # Auto-computed if None
    
    # Logging
    log_interval: int = 5  # Log every N rounds
    save_model: bool = True
    save_path: str = "./results"
    visualize: bool = True
    
    # Model architecture
    model_type: str = "simple_cnn"  # simple_cnn, resnet18, resnet50
    
    def __post_init__(self):
        """Compute derived parameters."""
        self.num_byzantine = int(self.num_clients * self.byzantine_ratio)
        self.num_honest = self.num_clients - self.num_byzantine
        
        # Update num_classes based on dataset
        if self.dataset in ["mnist", "cifar10"]:
            self.num_classes = 10
        elif self.dataset == "cifar100":
            self.num_classes = 100
        
        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = "cpu"
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
