"""
Model Checkpoint Management System

Handles saving, loading, and managing checkpoints for all experiment scales.
"""

import torch
import os
import json
from typing import Dict, Optional, Any
from pathlib import Path
import shutil


class CheckpointManager:
    """
    Manages model checkpoints across experiments.
    
    Features:
    - Automatic checkpoint saving with metadata
    - Best model tracking
    - Resume training from checkpoints
    - Multi-experiment organization
    """
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Root directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        experiment_name: str,
        is_best: bool = False,
        **kwargs
    ) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch:Current epoch
            metrics: Performance metrics
            experiment_name: Experiment identifier
            is_best: Whether this is the best model so far
            **kwargs: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        exp_dir = self.checkpoint_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'experiment_name': experiment_name,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add additional data
        checkpoint.update(kwargs)
        
        # Save regular checkpoint
        checkpoint_path = exp_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata_path = exp_dir / f'metadata_epoch_{epoch}.json'
        metadata = {
            'epoch': epoch,
            'metrics': metrics,
            'experiment_name': experiment_name
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save as best if applicable
        if is_best:
            best_path = exp_dir / 'best_model.pt'
            shutil.copy(checkpoint_path, best_path)
            print(f"✓ New best model saved (epoch {epoch})")
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints (keep last 5 + best)
        self._cleanup_old_checkpoints(exp_dir, keep_last=5)
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        experiment_name: str,
        epoch: Optional[int] = None,
        load_best: bool = False,
        map_location: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            experiment_name: Experiment identifier
            epoch: Specific epoch to load (None = latest)
            load_best: Load best model instead
            map_location: Device to map checkpoint to
            
        Returns:
            Checkpoint dictionary
        """
        exp_dir = self.checkpoint_dir / experiment_name
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        if load_best:
            checkpoint_path = exp_dir / 'best_model.pt'
            if not checkpoint_path.exists():
                raise ValueError(f"Best model not found for {experiment_name}")
        elif epoch is not None:
            checkpoint_path = exp_dir / f'checkpoint_epoch_{epoch}.pt'
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint for epoch {epoch} not found")
        else:
            # Load latest
            checkpoints = list(exp_dir.glob('checkpoint_epoch_*.pt'))
            if not checkpoints:
                raise ValueError(f"No checkpoints found for {experiment_name}")
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self, exp_dir: Path, keep_last: int = 5):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            exp_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        # Keep last N checkpoints
        for checkpoint in checkpoints[:-keep_last]:
            epoch = int(checkpoint.stem.split('_')[-1])
            checkpoint.unlink()
            metadata_path = exp_dir / f'metadata_epoch_{epoch}.json'
            if metadata_path.exists():
                metadata_path.unlink()
    
    def list_experiments(self) -> list:
        """List all available experiments."""
        experiments = [d.name for d in self.checkpoint_dir.iterdir() if d.is_dir()]
        return experiments
    
    def get_experiment_info(self, experiment_name: str) -> Dict:
        """Get information about an experiment."""
        exp_dir = self.checkpoint_dir / experiment_name
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        checkpoints = list(exp_dir.glob('checkpoint_epoch_*.pt'))
        metadatas = list(exp_dir.glob('metadata_epoch_*.json'))
        
        has_best = (exp_dir / 'best_model.pt').exists()
        
        info = {
            'experiment_name': experiment_name,
            'num_checkpoints': len(checkpoints),
            'has_best_model': has_best,
            'epochs': sorted([int(p.stem.split('_')[-1]) for p in checkpoints])
        }
        
        return info


def save_pretrained_checkpoint(
    model: torch.nn.Module,
    save_path: str,
    model_config: Dict,
    training_config: Dict,
    metrics: Dict
):
    """
    Save a pretrained model checkpoint with full configuration.
    
    For distribution and reproducibility.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_config': training_config,
        'final_metrics': metrics,
        'spectral_sentinel_version': '1.0.0'
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Pretrained checkpoint saved: {save_path}")


def load_pretrained_checkpoint(
    checkpoint_path: str,
    model_class,
    map_location: Optional[torch.device] = None
):
    """
    Load a pretrained model.
    
    Returns:
        (model, model_config, training_config, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Create model from config
    model_config = checkpoint['model_config']
    model = model_class(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    training_config = checkpoint.get('training_config', {})
    metrics = checkpoint.get('final_metrics', {})
    
    print(f"✓ Pretrained model loaded: {checkpoint_path}")
    
    return model, model_config, training_config, metrics


# Example usage
if __name__ == '__main__':
    import torch.nn as nn
    
    # Example model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    manager = CheckpointManager('./checkpoints')
    
    # Save checkpoint
    manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=10,
        metrics={'accuracy': 0.95, 'loss': 0.1},
        experiment_name='test_experiment',
        is_best=True
    )
    
    # Load checkpoint
    checkpoint = manager.load_checkpoint('test_experiment', load_best=True)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # List experiments
    experiments = manager.list_experiments()
    print(f"Available experiments: {experiments}")
