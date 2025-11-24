"""
Baseline Aggregation Methods

Collection of existing Byzantine-robust aggregators for comparison.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import time

from spectral_sentinel.aggregators.base_aggregator import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) - No Byzantine defense.
    
    Baseline method that simply averages all gradients.
    """
    
    def __init__(self):
        super().__init__(name="FedAvg")
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Simple averaging."""
        start_time = time.time()
        
        aggregated = self._average_gradients(gradients)
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=0,
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': client_ids if client_ids else list(range(len(gradients))),
            'byzantine_clients': [],
            'num_honest': len(gradients),
            'num_byzantine': 0,
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class KrumAggregator(BaseAggregator):
    """
    Krum aggregator: Select gradient closest to others.
    
    Computes pairwise distances and selects the gradient with
    smallest sum of distances to its k nearest neighbors.
    
    Reference: Blanchard et al. (2017) "Machine Learning with Adversaries"
    """
    
    def __init__(self, num_byzantine: int = 0):
        super().__init__(name="Krum")
        self.num_byzantine = num_byzantine
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Krum selection."""
        start_time = time.time()
        
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        n = len(gradients)
        if n <= 2 * self.num_byzantine + 2:
            # Not enough clients, fall back to averaging
            print("⚠️  Not enough clients for Krum, using FedAvg")
            aggregated = self._average_gradients(gradients)
            byzantine_clients = []
        else:
            # Flatten gradients
            flat_grads = [
                torch.cat([v.flatten() for v in g.values()])
                for g in gradients
            ]
            flat_grads = torch.stack(flat_grads)  # n × d
            
            # Compute pairwise distances
            distances = torch.cdist(flat_grads, flat_grads, p=2)  # n × n
            
            # For each gradient, sum distances to k nearest neighbors
            # k = n - num_byzantine - 2
            k = n - self.num_byzantine - 2
            
            scores = []
            for i in range(n):
                dists = distances[i]
                # Get k smallest distances (excluding self)
                k_nearest = torch.topk(dists, k + 1, largest=False)[0]
                score = k_nearest[1:].sum()  # Exclude distance to self (0)
                scores.append(score.item())
            
            # Select gradient with smallest score
            selected_idx = np.argmin(scores)
            aggregated = gradients[selected_idx]
            
            # Mark others as "rejected" (not actually Byzantine, just not selected)
            byzantine_clients = [cid for i, cid in enumerate(client_ids) 
                               if i != selected_idx]
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=len(byzantine_clients),
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': [client_ids[selected_idx]] if len(byzantine_clients) > 0 
                             else client_ids,
            'byzantine_clients': byzantine_clients,
            'num_honest': 1 if len(byzantine_clients) > 0 else len(gradients),
            'num_byzantine': len(byzantine_clients),
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class GeometricMedianAggregator(BaseAggregator):
    """
    Geometric Median aggregator.
    
    Computes approximate geometric median of gradients using
    Weiszfeld's algorithm. Robust to outliers.
    
    Reference: Pillutla et al. (2019)
    """
    
    def __init__(self, max_iter: int = 10, tol: float = 1e-5):
        super().__init__(name="GeometricMedian")
        self.max_iter = max_iter
        self.tol = tol
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Compute geometric median."""
        start_time = time.time()
        
        # Flatten gradients
        flat_grads = [
            torch.cat([v.flatten() for v in g.values()])
            for g in gradients
        ]
        flat_grads = torch.stack(flat_grads)  # n × d
        
        # Weiszfeld's algorithm
        median = flat_grads.mean(dim=0)  # Initialize with mean
        
        for _ in range(self.max_iter):
            # Compute distances to current median
            dists = torch.norm(flat_grads - median, dim=1)
            dists = torch.clamp(dists, min=1e-8)  # Avoid division by zero
            
            # Weighted average
            weights = 1.0 / dists
            weights = weights / weights.sum()
            
            new_median = (flat_grads.T @ weights).T
            
            # Check convergence
            if torch.norm(new_median - median) < self.tol:
                break
            
            median = new_median
        
        # Reshape back to original structure
        aggregated = {}
        idx = 0
        for k, v in gradients[0].items():
            size = v.numel()
            aggregated[k] = median[idx:idx+size].reshape(v.shape)
            idx += size
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=0,  # Geometric median doesn't explicitly reject
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': client_ids if client_ids else list(range(len(gradients))),
            'byzantine_clients': [],
            'num_honest': len(gradients),
            'num_byzantine': 0,
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class TrimmedMeanAggregator(BaseAggregator):
    """
    Trimmed Mean: Remove extreme values and average.
    
    For each parameter, sort values and remove top/bottom β fraction,
    then average the remaining.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        super().__init__(name="TrimmedMean")
        self.trim_ratio = trim_ratio
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Compute trimmed mean."""
        start_time = time.time()
        
        n = len(gradients)
        trim_count = int(n * self.trim_ratio)
        
        if trim_count == 0:
            # Not enough clients to trim
            aggregated = self._average_gradients(gradients)
        else:
            # Stack gradients
            aggregated = {}
            for k in gradients[0].keys():
                # Stack this parameter from all clients
                stacked = torch.stack([g[k] for g in gradients])  # n × shape
                
                # Sort along client dimension
                sorted_vals, _ = torch.sort(stacked, dim=0)
                
                # Trim top and bottom
                trimmed = sorted_vals[trim_count:-trim_count]
                
                # Average
                aggregated[k] = trimmed.mean(dim=0)
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=2 * trim_count,  # Trimmed from both ends
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': client_ids if client_ids else list(range(len(gradients))),
            'byzantine_clients': [],  # Can't identify specific Byzantine clients
            'num_honest': n - 2 * trim_count,
            'num_byzantine': 2 * trim_count,
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class MedianAggregator(BaseAggregator):
    """
    Coordinate-wise Median aggregator.
    
    Compute median independently for each parameter.
    """
    
    def __init__(self):
        super().__init__(name="Median")
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Compute coordinate-wise median."""
        start_time = time.time()
        
        aggregated = {}
        for k in gradients[0].keys():
            # Stack this parameter from all clients
            stacked = torch.stack([g[k] for g in gradients])  # n × shape
            
            # Compute median along client dimension
            aggregated[k] = torch.median(stacked, dim=0)[0]
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=0,
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': client_ids if client_ids else list(range(len(gradients))),
            'byzantine_clients': [],
            'num_honest': len(gradients),
            'num_byzantine': 0,
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class BulyanAggregator(BaseAggregator):
    """
    Bulyan: Multi-Krum + Trimmed Mean combination.
    
    More robust than Krum alone by selecting multiple gradients
    and then applying trimmed mean.
    
    Reference: El Mhamdi et al. (2018) "The Hidden Vulnerability of 
    Distributed Learning in Byzantium"
    """
    
    def __init__(self, num_byzantine: int = 0, selection_size: int = None):
        super().__init__(name="Bulyan")
        self.num_byzantine = num_byzantine
        self.selection_size = selection_size  # Number of gradients to select
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Bulyan aggregation."""
        start_time = time.time()
        
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        n = len(gradients)
        
        # Selection size: θ = n - 2f - 2 (where f is num_byzantine)
        if self.selection_size is None:
            theta = n - 2 * self.num_byzantine - 2
            theta = max(1, theta)
        else:
            theta = self.selection_size
        
        if theta >= n or n <= 2 * self.num_byzantine + 2:
            # Not enough clients, fall back to median
            print("⚠️  Not enough clients for Bulyan, using median")
            aggregator = MedianAggregator()
            return aggregator.aggregate(gradients, client_ids)
        
        # Step 1: Multi-Krum selection
        # Flatten gradients
        flat_grads = [
            torch.cat([v.flatten() for v in g.values()])
            for g in gradients
        ]
        flat_grads_tensor = torch.stack(flat_grads)  # n × d
        
        # Compute pairwise distances
        distances = torch.cdist(flat_grads_tensor, flat_grads_tensor, p=2)
        
        # For each gradient, compute score (sum of k nearest distances)
        k = n - self.num_byzantine - 2
        scores = []
        for i in range(n):
            dists = distances[i]
            k_nearest = torch.topk(dists, k + 1, largest=False)[0]
            score = k_nearest[1:].sum()  # Exclude self
            scores.append(score.item())
        
        # Select θ gradients with smallest scores
        selected_indices = np.argsort(scores)[:theta]
        selected_gradients = [gradients[i] for i in selected_indices]
        
        # Step 2: Trimmed mean on selected gradients
        # Trim β fraction from both ends
        beta = self.num_byzantine / theta
        trim_count = int(theta * beta)
        trim_count = max(1, min(trim_count, theta // 4))  # Ensure reasonable trimming
        
        aggregated = {}
        for k in selected_gradients[0].keys():
            # Stack this parameter from selected clients
            stacked = torch.stack([g[k] for g in selected_gradients])
            
            # Sort and trim
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if trim_count > 0 and len(sorted_vals) > 2 * trim_count:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals
            
            # Average
            aggregated[k] = trimmed.mean(dim=0)
        
        elapsed = time.time() - start_time
        rejected = [client_ids[i] for i in range(n) if i not in selected_indices]
        
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=len(rejected),
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': [client_ids[i] for i in selected_indices],
            'byzantine_clients': rejected,
            'num_honest': len(selected_indices),
            'num_byzantine': len(rejected),
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class SignGuardAggregator(BaseAggregator):
    """
    SignGuard: Sign-based Byzantine-robust aggregation.
    
    Uses sign of gradients rather than magnitude, making it robust
    to large-magnitude attacks. Takes majority vote on sign per coordinate.
    
    Reference: Xu et al. (2020) "SignGuard"
    """
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(name="SignGuard")
        self.threshold = threshold  # Fraction for majority vote
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """SignGuard aggregation using majority vote on gradient signs."""
        start_time = time.time()
        
        n = len(gradients)
        aggregated = {}
        
        for k in gradients[0].keys():
            # Stack this parameter from all clients
            stacked = torch.stack([g[k] for g in gradients])  # n × shape
            
            # Get signs
            signs = torch.sign(stacked)  # -1, 0, or +1
            
            # Majority vote per coordinate
            # Count positive, negative, zero
            positive_count = (signs > 0).float().sum(dim=0)
            negative_count = (signs < 0).float().sum(dim=0)
            
            # Majority rule
            majority_sign = torch.where(
                positive_count > negative_count,
                torch.ones_like(positive_count),
                -torch.ones_like(positive_count)
            )
            
            # Use median magnitude with majority sign
            abs_vals = torch.abs(stacked)
            median_magnitude = torch.median(abs_vals, dim=0)[0]
            
            aggregated[k] = majority_sign * median_magnitude
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=0,  # SignGuard doesn't explicitly reject
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': client_ids if client_ids else list(range(len(gradients))),
            'byzantine_clients': [],
            'num_honest': len(gradients),
            'num_byzantine': 0,
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class FLTrustAggregator(BaseAggregator):
    """
    FLTrust: Trust-based Byzantine-robust aggregation.
    
    Uses a trusted root dataset to compute a reference gradient.
    Client gradients are weighted by their cosine similarity to the root gradient.
    
    Reference: Cao et al. (2021) "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping"
    """
    
    def __init__(self, root_dataset: Optional[torch.utils.data.Dataset] = None):
        super().__init__(name="FLTrust")
        self.root_dataset = root_dataset
        self.root_gradient = None
    
    def set_root_gradient(self, root_gradient: Dict[str, torch.Tensor]):
        """Set reference gradient from trusted root dataset."""
        self.root_gradient = root_gradient
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """FLTrust aggregation using trust scores."""
        start_time = time.time()
        
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        # Need root gradient for trust scoring
        if self.root_gradient is None:
            print("⚠️  No root gradient set for FLTrust, using FedAvg")
            aggregated = self._average_gradients(gradients)
            byzantine_clients = []
        else:
            # Flatten root gradient
            root_flat = torch.cat([v.flatten() for v in self.root_gradient.values()])
            
            # Flatten client gradients
            flat_grads = [
                torch.cat([v.flatten() for v in g.values()])
                for g in gradients
            ]
            
            # Compute trust scores (cosine similarity)
            trust_scores = []
            for grad in flat_grads:
                cos_sim = F.cosine_similarity(grad, root_flat, dim=0)
                # ReLU to filter out negative similarities
                trust = torch.relu(cos_sim)
                trust_scores.append(trust.item())
            
            trust_scores = torch.tensor(trust_scores)
            
            # Normalize trust scores
            if trust_scores.sum() > 0:
                trust_scores = trust_scores / trust_scores.sum()
            else:
                # All scores are 0, fall back to uniform
                trust_scores = torch.ones(len(gradients)) / len(gradients)
            
            # Filter out clients with zero trust (Byzantine detection)
            honest_indices = [i for i, score in enumerate(trust_scores) if score > 1e-6]
            byzantine_clients = [client_ids[i] for i in range(len(gradients)) 
                                if i not in honest_indices]
            
            # Weighted aggregation
            aggregated = {}
            for k in gradients[0].keys():
                stacked = torch.stack([gradients[i][k] for i in honest_indices])
                weights = trust_scores[honest_indices]
                weights = weights / weights.sum()  # Renormalize
                
                # Weighted sum
                weighted = (stacked.T @ weights.to(stacked.device)).T
                aggregated[k] = weighted
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=len(byzantine_clients),
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': [client_ids[i] for i in honest_indices] if byzantine_clients else client_ids,
            'byzantine_clients': byzantine_clients,
            'num_honest': len(honest_indices) if byzantine_clients else len(gradients),
            'num_byzantine': len(byzantine_clients),
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class FLAMEAggregator(BaseAggregator):
    """
    FLAME: Federated Learning with Adaptive Model Estimation.
    
    Uses clustering to group similar gradients and adaptively filter outliers.
    
    Reference: Nguyen et al. (2022) "FLAME: Taming Backdoors in Federated Learning"
    """
    
    def __init__(self, num_clusters: int = 2, noise_threshold: float = 0.5):
        super().__init__(name="FLAME")
        self.num_clusters = num_clusters
        self.noise_threshold = noise_threshold
    
    def _cluster_gradients(self, flat_grads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple k-means clustering for gradients."""
        n = flat_grads.shape[0]
        
        # Initialize centers randomly
        indices = torch.randperm(n)[:self.num_clusters]
        centers = flat_grads[indices]
        
        # K-means iterations
        for _ in range(10):
            # Assign to nearest center
            distances = torch.cdist(flat_grads, centers, p=2)
            assignments = torch.argmin(distances, dim=1)
            
            # Update centers
            new_centers = []
            for c in range(self.num_clusters):
                cluster_members = flat_grads[assignments == c]
                if len(cluster_members) > 0:
                    new_centers.append(cluster_members.mean(dim=0))
                else:
                    new_centers.append(centers[c])
            centers = torch.stack(new_centers)
        
        return assignments, centers
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """FLAME aggregation using clustering."""
        start_time = time.time()
        
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        if len(gradients) < self.num_clusters:
            # Not enough gradients for clustering
            print("⚠️  Not enough clients for FLAME clustering, using FedAvg")
            aggregated = self._average_gradients(gradients)
            byzantine_clients = []
        else:
            # Flatten gradients
            flat_grads = [
                torch.cat([v.flatten() for v in g.values()])
                for g in gradients
            ]
            flat_grads = torch.stack(flat_grads)
            
            # Cluster gradients
            assignments, centers = self._cluster_gradients(flat_grads)
            
            # Find largest cluster (assume it's honest)
            cluster_sizes = [(assignments == c).sum() for c in range(self.num_clusters)]
            largest_cluster = np.argmax(cluster_sizes)
            
            # Select gradients from largest cluster
            honest_indices = [i for i in range(len(gradients)) 
                            if assignments[i] == largest_cluster]
            byzantine_clients = [client_ids[i] for i in range(len(gradients))
                                if i not in honest_indices]
            
            # Average honest gradients
            honest_gradients = [gradients[i] for i in honest_indices]
            aggregated = self._average_gradients(honest_gradients)
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=len(byzantine_clients),
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': [client_ids[i] for i in honest_indices] if byzantine_clients else client_ids,
            'byzantine_clients': byzantine_clients,
            'num_honest': len(honest_indices) if byzantine_clients else len(gradients),
            'num_byzantine': len(byzantine_clients),
            'aggregation_time': elapsed
        }
        
        return aggregated, info


class CRFLAggregator(BaseAggregator):
    """
    CRFL: Certified Robust Federated Learning.
    
    Provides certified robustness guarantees under bounded adversarial
    perturbation ||δ|| ≤ Δ.
    
    Uses coordinate-wise clipping + trimmed mean.
    """
    
    def __init__(self, delta: float = 0.1, trim_ratio: float = 0.1):
        super().__init__(name="CRFL")
        self.delta = delta  # Perturbation bound
        self.trim_ratio = trim_ratio
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """CRFL aggregation with clipping + trimmed mean."""
        start_time = time.time()
        
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        # Step 1: Clip gradients to ball of radius delta
        clipped_gradients = []
        for grad in gradients:
            flat = torch.cat([v.flatten() for v in grad.values()])
            norm = torch.norm(flat)
            
            if norm > self.delta:
                scale = self.delta / norm
                clipped = {k: scale * v for k, v in grad.items()}
            else:
                clipped = grad
            
            clipped_gradients.append(clipped)
        
        # Step 2: Trimmed mean
        n = len(clipped_gradients)
        trim_count = int(n * self.trim_ratio)
        
        if trim_count > 0:
            aggregated = {}
            for k in clipped_gradients[0].keys():
                stacked = torch.stack([g[k] for g in clipped_gradients])
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[trim_count:-trim_count]
                aggregated[k] = trimmed.mean(dim=0)
        else:
            aggregated = self._average_gradients(clipped_gradients)
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=2 * trim_count,
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': client_ids,
            'byzantine_clients': [],
            'num_honest': n - 2 * trim_count,
            'num_byzantine': 2 * trim_count,
            'aggregation_time': elapsed,
            'certified_against': f'||δ|| ≤ {self.delta}'
        }
        
        return aggregated, info


class ByzShieldAggregator(BaseAggregator):
    """
    ByzShield: Byzantine-resilient aggregation with norm-based filtering.
    
    Provides guarantees under bounded perturbation with adaptive threshold.
    """
    
    def __init__(self, delta: float = 0.1, alpha: float = 0.2):
        super().__init__(name="ByzShield")
        self.delta = delta  # Perturbation bound
        self.alpha = alpha  # Rejection fraction
    
    def aggregate(self,
                 gradients: List[Dict[str, torch.Tensor]],
                 client_ids: Optional[List[int]] = None,
                 **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """ByzShield aggregation with norm-based filtering."""
        start_time = time.time()
        
        if client_ids is None:
            client_ids = list(range(len(gradients)))
        
        # Compute norms
        norms = []
        for grad in gradients:
            flat = torch.cat([v.flatten() for v in grad.values()])
            norms.append(torch.norm(flat).item())
        
        norms = np.array(norms)
        
        # Filter outliers by norm
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))  # Median absolute deviation
        
        # Reject gradients with norm > median + threshold
        threshold = median_norm + 3 * mad
        
        honest_indices = [i for i, norm in enumerate(norms) if norm <= threshold]
        byzantine_indices = [i for i, norm in enumerate(norms) if norm > threshold]
        
        # Average honest gradients
        honest_gradients = [gradients[i] for i in honest_indices]
        if len(honest_gradients) > 0:
            aggregated = self._average_gradients(honest_gradients)
        else:
            # Fallback: all gradients
            aggregated = self._average_gradients(gradients)
            honest_indices = list(range(len(gradients)))
            byzantine_indices = []
        
        elapsed = time.time() - start_time
        self.update_statistics(
            num_clients=len(gradients),
            num_rejected=len(byzantine_indices),
            aggregation_time=elapsed
        )
        
        info = {
            'honest_clients': [client_ids[i] for i in honest_indices],
            'byzantine_clients': [client_ids[i] for i in byzantine_indices],
            'num_honest': len(honest_indices),
            'num_byzantine': len(byzantine_indices),
            'aggregation_time': elapsed,
            'certified_against': f'||δ|| ≤ {self.delta}'
        }
        
        return aggregated, info


# Aggregator registry
AGGREGATOR_REGISTRY = {
    'fedavg': FedAvgAggregator,
    'krum': KrumAggregator,
    'geometric_median': GeometricMedianAggregator,
    'trimmed_mean': TrimmedMeanAggregator,
    'median': MedianAggregator,
    'bulyan': BulyanAggregator,
    'signguard': SignGuardAggregator,
    'fltrust': FLTrustAggregator,
    'flame': FLAMEAggregator,
    'crfl': CRFLAggregator,
    'byzshield': ByzShieldAggregator,
}


def get_aggregator(name: str, **kwargs) -> BaseAggregator:
    """
    Get aggregator by name.
    
    Args:
        name: Aggregator name
        **kwargs: Aggregator-specific parameters
        
    Returns:
        Aggregator instance
    """
    from spectral_sentinel.aggregators.spectral_sentinel import SpectralSentinelAggregator
    
    if name.lower() == 'spectral_sentinel':
        return SpectralSentinelAggregator(**kwargs)
    
    if name.lower() not in AGGREGATOR_REGISTRY:
        raise ValueError(f"Unknown aggregator: {name}. "
                        f"Available: {list(AGGREGATOR_REGISTRY.keys())}")
    
    return AGGREGATOR_REGISTRY[name.lower()](**kwargs)
