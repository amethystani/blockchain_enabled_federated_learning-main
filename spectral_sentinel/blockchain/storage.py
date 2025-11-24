"""
Model storage for blockchain-enabled federated learning.

Handles compression, hashing, and storage of model updates.
Options: Local storage or IPFS (decentralized).
"""

import json
import pickle
import hashlib
import gzip
from pathlib import Path
from typing import Dict, Tuple, Any
import torch


class ModelStorage:
    """Handle model storage and retrieval."""
    
    def __init__(self, storage_dir: str = "./blockchain_storage", use_ipfs: bool = False):
        """
        Initialize model storage.
        
        Args:
            storage_dir: Directory for local storage
            use_ipfs: Whether to use IPFS (not implemented yet)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.use_ipfs = use_ipfs
        
        if use_ipfs:
            raise NotImplementedError("IPFS storage not yet implemented. Set use_ipfs=False")
    
    def store_model(self, model_dict: Dict[str, torch.Tensor], 
                    client_id: int, round_num: int) -> Tuple[str, str]:
        """
        Store model update and return hash and storage location.
        
        Args:
            model_dict: Model state dict
            client_id: Client identifier
            round_num: Round number
        
        Returns:
            Tuple of (model_hash, storage_path)
        """
        # Convert model to bytes
        model_bytes = self.serialize_model(model_dict)
        
        # Compress
        compressed = self.compress(model_bytes)
        
        # Calculate hash
        model_hash = self.calculate_hash(compressed)
        
        # Save to storage
        filename = f"model_r{round_num}_c{client_id}_{model_hash[:16]}.pkl.gz"
        storage_path = self.storage_dir / filename
        
        with open(storage_path, 'wb') as f:
            f.write(compressed)
        
        return model_hash, str(storage_path)
    
    def retrieve_model(self, model_hash: str, storage_path: str = None) -> Dict[str, torch.Tensor]:
        """
        Retrieve model update from storage.
        
        Args:
            model_hash: Hash of the model
            storage_path: Path to stored model (if None, searches by hash)
        
        Returns:
            Model state dict
        """
        if storage_path is None:
            # Search for file by hash
            storage_path = self._find_by_hash(model_hash)
        
        storage_path = Path(storage_path)
        
        if not storage_path.exists():
            raise FileNotFoundError(f"Model file not found: {storage_path}")
        
        # Read compressed file
        with open(storage_path, 'rb') as f:
            compressed = f.read()
        
        # Verify hash
        calculated_hash = self.calculate_hash(compressed)
        if calculated_hash != model_hash:
            raise ValueError(f"Hash mismatch! Expected {model_hash}, got {calculated_hash}")
        
        # Decompress and deserialize
        model_bytes = self.decompress(compressed)
        model_dict = self.deserialize_model(model_bytes)
        
        return model_dict
    
    def serialize_model(self, model_dict: Dict[str, torch.Tensor]) -> bytes:
        """Serialize model to bytes."""
        # Convert to CPU and detach all tensors
        cpu_dict = {k: v.detach().cpu() for k, v in model_dict.items()}
        return pickle.dumps(cpu_dict)
    
    def deserialize_model(self, model_bytes: bytes) -> Dict[str, torch.Tensor]:
        """Deserialize model from bytes."""
        return pickle.loads(model_bytes)
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data, compresslevel=6)
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress gzipped data."""
        return gzip.decompress(data)
    
    def calculate_hash(self, data: bytes) -> str:
        """Calculate SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    def _find_by_hash(self, model_hash: str) -> Path:
        """Find model file by hash prefix."""
        pattern = f"*_{model_hash[:16]}.pkl.gz"
        matches = list(self.storage_dir.glob(pattern))
        
        if not matches:
            raise FileNotFoundError(f"No model file found with hash {model_hash}")
        
        if len(matches) > 1:
            raise ValueError(f"Multiple files found with hash {model_hash}")
        
        return matches[0]
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        files = list(self.storage_dir.glob("*.pkl.gz"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "num_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_dir": str(self.storage_dir)
        }
    
    def cleanup_round(self, round_num: int):
        """Delete all models from a specific round."""
        pattern = f"model_r{round_num}_*.pkl.gz"
        for f in self.storage_dir.glob(pattern):
            f.unlink()
    
    def cleanup_all(self):
        """Delete all stored models."""
        for f in self.storage_dir.glob("*.pkl.gz"):
            f.unlink()
