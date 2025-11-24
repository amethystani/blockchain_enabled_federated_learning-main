"""
Multi-GPU Training Support for Spectral Sentinel

Implements PyTorch DataParallel and DistributedDataParallel for scaling
to multiple GPUs.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
import os


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Initialize distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


class MultiGPUTrainer:
    """
    Multi-GPU training wrapper for Spectral Sentinel.
    
    Supports both DataParallel (single-node multi-GPU) and 
    DistributedDataParallel (multi-node multi-GPU).
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_ddp: bool = False,
        local_rank: Optional[int] = None
    ):
        """
        Initialize multi-GPU trainer.
        
        Args:
            model: Model to train
            use_ddp: Use DistributedDataParallel (True) or DataParallel (False)
            local_rank: Local GPU rank for DDP
        """
        self.model = model
        self.use_ddp = use_ddp
        self.local_rank = local_rank or 0
        
        if use_ddp:
            self._setup_ddp()
        else:
            self._setup_dp()
    
    def _setup_dp(self):
        """Setup DataParallel (single-node)."""
        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def _setup_ddp(self):
        """Setup DistributedDataParallel (multi-node)."""
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        print(f"Using DistributedDataParallel on rank {self.local_rank}")
    
    def get_model(self) -> nn.Module:
        """Get underlying model (unwrapped)."""
        if isinstance(self.model, (nn.DataParallel, DDP)):
            return self.model.module
        return self.model
    
    def save_checkpoint(self, path: str, optimizer=None, epoch=None, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            path: Save path
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            **kwargs: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.get_model().state_dict(),
            'model_config': kwargs.get('model_config', {}),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        # Add any additional kwargs
        for k, v in kwargs.items():
            if k not in checkpoint:
                checkpoint[k] = v
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str, optimizer=None, map_location=None):
        """
        Load model checkpoint.
        
        Args:
            path: Checkpoint path
            optimizer: Optional optimizer to restore
            map_location: Device mapping
            
        Returns:
            Checkpoint dict
        """
        if map_location is None:
            map_location = self.device
        
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load model state
        self.get_model().load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Checkpoint loaded: {path}")
        return checkpoint


class MixedPrecisionTrainer:
    """
    Automatic Mixed Precision (AMP) training wrapper.
    
    Uses torch.cuda.amp for faster training with FP16/FP32 mixed precision.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize AMP trainer.
        
        Args:
            enabled: Enable mixed precision
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)
        
        if self.enabled:
            print("✓ Automatic Mixed Precision enabled")
    
    def scale_loss(self, loss):
        """Scale loss for gradient computation."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """Optimizer step with gradient scaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def autocast(self):
        """Context manager for mixed precision forward pass."""
        return torch.cuda.amp.autocast(enabled=self.enabled)


def get_num_gpus() -> int:
    """Get number of available GPUs."""
    return torch.cuda.device_count()


def print_gpu_info():
    """Print GPU information."""
    if not torch.cuda.is_available():
        print("❌ No GPUs available")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"🖥️  GPU Information")
    print(f"{'='*60}")
    print(f"GPUs available: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    print(f"{'='*60}\n")


# Example usage
if __name__ == '__main__':
    print_gpu_info()
    
    # Example model
    model = nn.Linear(10, 10)
    
    # Multi-GPU setup
    trainer = MultiGPUTrainer(model, use_ddp=False)
    
    # Mixed precision
    amp_trainer = MixedPrecisionTrainer(enabled=True)
    
    print("✓ Multi-GPU and AMP setup complete")
