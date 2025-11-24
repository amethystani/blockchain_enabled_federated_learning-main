"""
Spectral Sentinel Aggregator

Main Byzantine-robust aggregator using Random Matrix Theory.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import time

from spectral_sentinel.aggregators.base_aggregator import BaseAggregator
from spectral_sentinel.rmt.spectral_analyzer import SpectralAnalyzer
from spectral_sentinel.sketching.frequent_directions import FrequentDirections


class SpectralSentinelAggregator(BaseAggregator):
    """
    Spectral Sentinel: Byzantine-robust aggregation via RMT.
    
    Pipeline:
    1. Collect gradients from clients
    2. Optionally apply sketching for large models
    3. Analyze spectral properties using Marchenko-Pastur law
    4. Detect and filter Byzantine gradients
    5. Aggregate honest gradients
    """
    
    def __init__(self,
                 ks_threshold: float = 0.05,
                 tail_threshold: float = 0.1,
                 use_sketching: bool = False,
                 sketch_size: int = 256,
                 window_size: int = 50,
                 clip_threshold: Optional[float] = None):
        """
        Initialize Spectral Sentinel aggregator.
        
        Args:
            ks_threshold: KS test p-value threshold
            tail_threshold: Tail anomaly threshold
            use_sketching: Enable Frequent Directions sketching
            sketch_size: Sketch size k (if sketching enabled)
            window_size: Sliding window for online MP tracking
            clip_threshold: Optional gradient clipping threshold
        """
        super().__init__(name="SpectralSentinel")
        
        self.ks_threshold = ks_threshold
        self.tail_threshold = tail_threshold
        self.use_sketching = use_sketching
        self.sketch_size = sketch_size
        self.clip_threshold = clip_threshold
        
        # Initialize spectral analyzer
        self.analyzer = SpectralAnalyzer(
            ks_threshold=ks_threshold,
            tail_threshold=tail_threshold,
            window_size=window_size,
            adaptive=True
        )
        
        # Sketching (initialized on first use)
        self.sketcher: Optional[FrequentDirections] = None
        
        # Additional statistics
        self.stats.update({
            'byzantine_detected_per_round': [],
            'false_positives': [],
            'ks_statistics': [],
            'tail_anomalies': []
        })
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Aggregate gradients with Byzantine detection.
        
        Args:
            gradients: List of gradient dicts from clients
            client_ids: Optional client IDs
            **kwargs: Additional parameters
            
        Returns:
            (aggregated_gradient, detection_info)
        """
        start_time = time.time()
        
        if len(gradients) == 0:
            raise ValueError("Cannot aggregate empty gradient list")
        
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        # Optional: Clip gradients
        if self.clip_threshold is not None:
            gradients = self._clip_gradients(gradients, self.clip_threshold)
        
        # Analyze gradients for Byzantine detection
        detection_results = self.analyzer.analyze_gradients(gradients, client_ids)
        
        # Filter out Byzantine clients
        honest_client_ids = detection_results['honest_clients']
        byzantine_client_ids = detection_results['byzantine_detected']
        
        # Get honest gradients
        honest_gradients = [
            gradients[i] for i, cid in enumerate(client_ids)
            if cid in honest_client_ids
        ]
        
        # If all clients detected as Byzantine, fall back to simple average
        if len(honest_gradients) == 0:
            print("⚠️  Warning: All clients flagged as Byzantine, using all gradients")
            honest_gradients = gradients
            byzantine_client_ids = []
        
        # Aggregate honest gradients (simple averaging)
        aggregated = self._average_gradients(honest_gradients)
        
        # Update statistics
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=len(byzantine_client_ids),
            aggregation_time=elapsed
        )
        
        self.stats['byzantine_detected_per_round'].append(len(byzantine_client_ids))
        self.stats['ks_statistics'].append(detection_results['ks_statistic'])
        self.stats['tail_anomalies'].append(detection_results['tail_anomalies'])
        
        # Prepare info dict
        info = {
            'honest_clients': honest_client_ids,
            'byzantine_clients': byzantine_client_ids,
            'num_honest': len(honest_client_ids),
            'num_byzantine': len(byzantine_client_ids),
            'ks_statistic': detection_results['ks_statistic'],
            'ks_pvalue': detection_results['ks_pvalue'],
            'tail_anomalies': detection_results['tail_anomalies'],
            'detection_method': detection_results['detection_method'],
            'aggregation_time': elapsed
        }
        
        return aggregated, info
    
    def visualize_detection(self, save_path: Optional[str] = None):
        """
        Visualize spectral detection results.
        
        Args:
            save_path: Path to save figure
        """
        self.analyzer.visualize_spectrum(save_path)
    
    def get_extended_statistics(self) -> Dict:
        """Get extended statistics including RMT analysis."""
        stats = self.get_statistics()
        
        # Add analyzer statistics
        analyzer_stats = self.analyzer.get_statistics()
        stats.update({
            'analyzer': analyzer_stats
        })
        
        return stats
    
    def __repr__(self) -> str:
        return (f"SpectralSentinel(ks_threshold={self.ks_threshold}, "
                f"sketching={self.use_sketching}, "
                f"sketch_size={self.sketch_size if self.use_sketching else 'N/A'})")
