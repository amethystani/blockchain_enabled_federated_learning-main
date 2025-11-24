"""
Layer-wise Sketching for Transformer/LLM Architectures

Decompose model by layer and apply sketching to each layer separately.
Enables detection of targeted attacks on specific layers (e.g., attention blocks).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from spectral_sentinel.sketching.frequent_directions import FrequentDirections


class LayerWiseSketch:
    """
    Apply Frequent Directions sketching to each layer of a model separately.
    
    For transformers/LLMs with L layers and d_layer parameters per layer:
    - Memory: O(L * k² * d_layer) vs O(d²) for full model
    - Detection: Can identify which layers are under attack
    """
    
    def __init__(self, 
                 layer_dims: Dict[str, int],
                 sketch_size: int = 256):
        """
        Initialize layer-wise sketcher.
        
        Args:
            layer_dims: Dict mapping layer names to parameter counts
            sketch_size: k for each layer's sketch
        """
        self.layer_dims = layer_dims
        self.sketch_size = sketch_size
        
        # Create one sketch per layer
        self.layer_sketches: Dict[str, FrequentDirections] = {}
        for layer_name, dim in layer_dims.items():
            self.layer_sketches[layer_name] = FrequentDirections(sketch_size, dim)
    
    def update(self, gradient_dict: Dict[str, torch.Tensor]):
        """
        Update sketches with gradients organized by layer.
        
        Args:
            gradient_dict: Dict mapping layer names to gradient tensors
        """
        for layer_name, grad in gradient_dict.items():
            if layer_name in self.layer_sketches:
                # Flatten gradient for this layer
                flat_grad = grad.flatten().detach().cpu().numpy()
                self.layer_sketches[layer_name].update(flat_grad)
    
    def batch_update(self, gradient_dicts: List[Dict[str, torch.Tensor]]):
        """
        Update with gradients from multiple clients.
        
        Args:
            gradient_dicts: List of gradient dicts, one per client
        """
        for grad_dict in gradient_dicts:
            self.update(grad_dict)
    
    def get_layer_eigenvalues(self, layer_name: str) -> np.ndarray:
        """
        Get eigenvalues for a specific layer.
        
        Args:
            layer_name: Name of layer
            
        Returns:
            Eigenvalues for that layer
        """
        if layer_name not in self.layer_sketches:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        return self.layer_sketches[layer_name].get_eigenvalues()
    
    def get_all_eigenvalues(self) -> Dict[str, np.ndarray]:
        """
        Get eigenvalues for all layers.
        
        Returns:
            Dict mapping layer names to eigenvalues
        """
        return {
            layer_name: sketch.get_eigenvalues()
            for layer_name, sketch in self.layer_sketches.items()
        }
    
    def detect_layer_attacks(self, 
                            mp_laws: Dict[str, 'MarchenkoPasturLaw'],
                            tail_threshold: float = 0.1) -> Dict[str, bool]:
        """
        Detect if each layer is under attack.
        
        Args:
            mp_laws: Dict of MP laws for each layer
            tail_threshold: Threshold for tail anomaly
            
        Returns:
            Dict mapping layer names to attack detected (bool)
        """
        from spectral_sentinel.rmt.marchenko_pastur import detect_tail_anomalies
        
        attack_detected = {}
        
        for layer_name, sketch in self.layer_sketches.items():
            eigenvalues = sketch.get_eigenvalues()
            
            if len(eigenvalues) == 0:
                attack_detected[layer_name] = False
                continue
            
            if layer_name not in mp_laws:
                # No baseline, skip
                attack_detected[layer_name] = False
                continue
            
            mp_law = mp_laws[layer_name]
            
            # Detect tail anomalies
            _, n_anomalies = detect_tail_anomalies(eigenvalues, mp_law, tail_threshold)
            
            attack_detected[layer_name] = n_anomalies > 0
        
        return attack_detected
    
    def get_total_memory_usage(self) -> int:
        """
        Get total memory usage across all layers.
        
        Returns:
            Memory in bytes
        """
        return sum(sketch.get_memory_usage() 
                  for sketch in self.layer_sketches.values())
    
    def reset(self):
        """Reset all layer sketches."""
        for sketch in self.layer_sketches.values():
            sketch.reset()
    
    @staticmethod
    def from_model(model: nn.Module, 
                   sketch_size: int = 256,
                   layer_names: Optional[List[str]] = None) -> 'LayerWiseSketch':
        """
        Create layer-wise sketch from a PyTorch model.
        
        Args:
            model: PyTorch model
            sketch_size: k for each layer
            layer_names: Optional list of layer names to track (default: all)
            
        Returns:
            LayerWiseSketch instance
        """
        # Extract layer dimensions
        layer_dims = OrderedDict()
        
        for name, param in model.named_parameters():
            if layer_names is None or name in layer_names:
                layer_dims[name] = param.numel()
        
        return LayerWiseSketch(layer_dims, sketch_size)
    
    @staticmethod
    def organize_gradients_by_layer(gradients: List[Dict[str, torch.Tensor]],
                                    model: nn.Module) -> List[Dict[str, torch.Tensor]]:
        """
        Helper to organize flat gradients into per-layer dicts.
        
        Args:
            gradients: List of gradient dicts from clients
            model: Model to get layer structure
            
        Returns:
            List of layer-organized gradient dicts
        """
        # This is already in the right format if gradients are dicts
        # Just validate structure
        param_names = set(name for name, _ in model.named_parameters())
        
        organized = []
        for grad_dict in gradients:
            # Filter to only include parameters in model
            filtered = {
                k: v for k, v in grad_dict.items()
                if k in param_names
            }
            organized.append(filtered)
        
        return organized
    
    def __repr__(self) -> str:
        total_mem_bytes = self.get_total_memory_usage()
        mem_mb = total_mem_bytes / (1024 ** 2)
        return (f"LayerWiseSketch(layers={len(self.layer_sketches)}, "
                f"k={self.sketch_size}, memory={mem_mb:.2f}MB)")


def analyze_layer_vulnerability(layer_eigenvalues: Dict[str, np.ndarray],
                                layer_mp_laws: Dict[str, 'MarchenkoPasturLaw']) -> Dict[str, float]:
    """
    Compute vulnerability score for each layer.
    
    Higher score = more vulnerable to targeted attacks
    
    Args:
        layer_eigenvalues: Eigenvalues per layer
        layer_mp_laws: MP laws per layer
        
    Returns:
        Vulnerability scores (0-1) per layer
    """
    vulnerability = {}
    
    for layer_name in layer_eigenvalues.keys():
        eigenvals = layer_eigenvalues[layer_name]
        
        if len(eigenvals) == 0 or layer_name not in layer_mp_laws:
            vulnerability[layer_name] = 0.0
            continue
        
        mp_law = layer_mp_laws[layer_name]
        
        # Vulnerability = fraction of eigenvalues beyond MP support
        lambda_plus = mp_law.lambda_plus
        beyond_support = np.sum(eigenvals > lambda_plus * 1.05)
        vulnerability[layer_name] = beyond_support / len(eigenvals)
    
    return vulnerability
