"""
Frequent Directions Algorithm for Covariance Sketching

Memory-efficient approximation of covariance matrices for large models.
Reduces memory from O(d²) to O(k²) where k << d.

Reference: Liberty (2013) "Simple and Deterministic Matrix Sketching"
"""

import numpy as np
import torch
from typing import Optional, Tuple


class FrequentDirections:
    """
    Frequent Directions algorithm for matrix sketching.
    
    Maintains a sketch of a streaming matrix that approximates its covariance
    structure with O(k²) memory instead of O(d²).
    
    For a matrix A ∈ R^{n×d}, maintains sketch B ∈ R^{k×d} such that:
        ||A^T A - B^T B||_2 ≤ ||A||_F² / k
    
    This is crucial for models with billions of parameters where storing
    full covariance would require terabytes of memory.
    """
    
    def __init__(self, sketch_size: int, feature_dim: int):
        """
        Initialize Frequent Directions sketch.
        
        Args:
            sketch_size: k, number of rows in sketch (memory budget)
            feature_dim: d, number of features (model parameters)
        """
        self.k = sketch_size
        self.d = feature_dim
        
        # Sketch matrix B ∈ R^{k×d}
        self.sketch = np.zeros((self.k, self.d), dtype=np.float32)
        
        # Number of rows currently filled (initially 0)
        self.filled_rows = 0
        
        # Track number of shrinks (for debugging)
        self.n_shrinks = 0
    
    def update(self, row: np.ndarray):
        """
        Update sketch with a new row (e.g., gradient from one client).
        
        Args:
            row: New row vector of shape (d,)
        """
        row = row.astype(np.float32)
        
        if self.filled_rows < self.k:
            # Sketch not full yet, just append
            self.sketch[self.filled_rows] = row
            self.filled_rows += 1
        else:
            # Sketch full, need to shrink
            self._shrink_and_insert(row)
    
    def batch_update(self, rows: np.ndarray):
        """
        Update sketch with multiple rows at once.
        
        Args:
            rows: Matrix of shape (n, d)
        """
        for row in rows:
            self.update(row)
    
    def _shrink_and_insert(self, row: np.ndarray):
        """
        Shrink sketch and insert new row.
        
        Core of the Frequent Directions algorithm:
        1. Insert new row into sketch
        2. Compute SVD of sketch
        3. Zero out smallest singular value
        4. Shrink all singular values by δ = σ²_{k/2}
        5. Keep top k/2 singular vectors
        """
        # Insert new row at bottom
        self.sketch = np.vstack([self.sketch, row])
        
        # SVD: B = UΣV^T
        U, sigma, Vt = np.linalg.svd(self.sketch, full_matrices=False)
        
        # Shrink: subtract δ = σ²_{k/2} from squared singular values
        delta = sigma[self.k // 2] ** 2
        
        # Compute new singular values: σ'_i = √(σ²_i - δ)
        sigma_sq = sigma ** 2 - delta
        sigma_sq = np.maximum(sigma_sq, 0)  # Clip to non-negative
        new_sigma = np.sqrt(sigma_sq)
        
        # Keep only top k/2 rows
        # Reconstruction: B' = Σ' V^T
        self.sketch = new_sigma[:self.k // 2, None] * Vt[:self.k // 2, :]
        
        # Zero-pad to size k
        self.sketch = np.vstack([
            self.sketch,
            np.zeros((self.k - self.k // 2, self.d), dtype=np.float32)
        ])
        
        self.filled_rows = self.k // 2
        self.n_shrinks += 1
    
    def get_covariance_approximation(self) -> np.ndarray:
        """
        Get approximation of A^T A where A is the streamed matrix.
        
        Returns:
            Approximation B^T B ∈ R^{d×d}
        """
        # Only use filled rows
        active_sketch = self.sketch[:self.filled_rows]
        return active_sketch.T @ active_sketch
    
    def get_eigenvalues(self) -> np.ndarray:
        """
        Get approximate eigenvalues of A^T A / n.
        
        Returns:
            Eigenvalues in descending order
        """
        cov_approx = self.get_covariance_approximation()
        
        # For memory efficiency with large d, use only sketch eigenvalues
        # eigenvalues of B^T B = eigenvalues of B B^T (smaller matrix)
        active_sketch = self.sketch[:self.filled_rows]
        
        if self.filled_rows > 0:
            small_cov = active_sketch @ active_sketch.T  # k × k matrix
            eigenvalues = np.linalg.eigvalsh(small_cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Normalize by number of samples (for covariance)
            # This will be adjusted by the caller based on total samples
            return eigenvalues
        else:
            return np.array([])
    
    def get_sketch(self) -> np.ndarray:
        """Get current sketch matrix."""
        return self.sketch[:self.filled_rows].copy()
    
    def reset(self):
        """Reset sketch to empty state."""
        self.sketch = np.zeros((self.k, self.d), dtype=np.float32)
        self.filled_rows = 0
        self.n_shrinks = 0
    
    def get_memory_usage(self) -> int:
        """
        Get memory usage in bytes.
        
        Returns:
            Memory in bytes
        """
        # k × d matrix of float32 (4 bytes each)
        return self.k * self.d * 4
    
    def get_approximation_error_bound(self, frobenius_norm_sq: float) -> float:
        """
        Theoretical upper bound on approximation error.
        
        ||A^T A - B^T B||_2 ≤ ||A||_F² / (k/2)
        
        Args:
            frobenius_norm_sq: ||A||_F², Frobenius norm squared of input matrix
            
        Returns:
            Error bound
        """
        return frobenius_norm_sq / (self.k / 2)
    
    @staticmethod
    def from_gradients(gradients: list, sketch_size: int):
        """
        Create sketch from list of gradient tensors.
        
        Args:
            gradients: List of gradient tensors
            sketch_size: k, sketch size
            
        Returns:
            FrequentDirections sketch
        """
        # Flatten first gradient to get dimension
        if isinstance(gradients[0], dict):
            first_grad = torch.cat([v.flatten() for v in gradients[0].values()])
        else:
            first_grad = gradients[0].flatten()
        
        feature_dim = first_grad.numel()
        
        # Create sketch
        fd = FrequentDirections(sketch_size, feature_dim)
        
        # Add all gradients
        for grad in gradients:
            if isinstance(grad, dict):
                flat = torch.cat([v.flatten() for v in grad.values()])
            else:
                flat = grad.flatten()
            
            fd.update(flat.detach().cpu().numpy())
        
        return fd
    
    def __repr__(self) -> str:
        return (f"FrequentDirections(k={self.k}, d={self.d}, "
                f"filled={self.filled_rows}, shrinks={self.n_shrinks})")


class AdaptiveSketchSize:
    """
    Automatically determine sketch size based on model architecture.
    
    Rules of thumb:
    - CNNs (ResNet, etc.): k = 256 sufficient (low rank)
    - Transformers: k = 512 required (higher rank)
    - Foundation models: k = 1024 for 1B+ parameters
    """
    
    @staticmethod
    def recommend_sketch_size(model_params: int, 
                             model_type: str = "unknown",
                             available_memory_gb: float = 4.0) -> int:
        """
        Recommend sketch size given model size and available memory.
        
        Args:
            model_params: Number of model parameters
            model_type: 'cnn', 'transformer', or 'unknown'
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            Recommended sketch size k
        """
        # Memory constraint: k² * d * 4 bytes ≤ available_memory
        # k ≤ √(available_memory_gb * 1e9 / (4 * d))
        max_k_memory = int(np.sqrt(available_memory_gb * 1e9 / (4 * model_params)))
        
        # Architecture-based recommendation
        if model_type == "cnn":
            # CNNs have low-rank gradients
            recommended_k = min(256, model_params // 1000)
        elif model_type == "transformer":
            # Transformers have higher rank
            recommended_k = min(512, model_params // 500)
        else:
            # Conservative default
            recommended_k = min(512, model_params // 1000)
        
        # Take minimum of memory and architecture constraints
        final_k = min(recommended_k, max_k_memory)
        
        # Ensure minimum size
        final_k = max(64, final_k)
        
        return int(final_k)
    
    @staticmethod
    def estimate_memory_usage(sketch_size: int, feature_dim: int) -> Tuple[float, str]:
        """
        Estimate memory usage of sketch.
        
        Args:
            sketch_size: k
            feature_dim: d
            
        Returns:
            (memory_value, memory_unit) e.g., (2.5, "GB")
        """
        bytes_used = sketch_size * feature_dim * 4
        
        if bytes_used < 1024:
            return bytes_used, "B"
        elif bytes_used < 1024 ** 2:
            return bytes_used / 1024, "KB"
        elif bytes_used < 1024 ** 3:
            return bytes_used / (1024 ** 2), "MB"
        else:
            return bytes_used / (1024 ** 3), "GB"
