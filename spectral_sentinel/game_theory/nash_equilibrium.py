"""
Game-Theoretic Adversarial Analysis

Implements Nash equilibrium adaptive adversaries that optimize attack strategies
while accounting for detection probability.

Reference: Line 11 from WHATWEHAVETOIMPLEMENT.MD
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GameTheoreticConfig:
    """Configuration for game-theoretic adversary."""
    learning_rate: float = 0.01
    num_iterations: int = 100
    detection_penalty: float = 10.0  # Cost of being detected
    attack_reward: float = 1.0  # Reward for successful attack
    sigma_sq_f_sq_threshold: float = 0.25  # Phase transition threshold


class NashEquilibriumAdversary:
    """
    Nash Equilibrium Adaptive Adversary.
    
    Models Byzantine attackers as rational agents maximizing attack impact
    subject to detection probability constraints using online convex optimization.
    """
    
    def __init__(self, config: GameTheoreticConfig):
        self.config = config
        self.attack_history = []
        self.detection_history = []
        
    def compute_nash_strategy(
        self,
        honest_gradients: List[Dict[str, torch.Tensor]],
        phase_transition_metric: float,
        detection_threshold: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Nash equilibrium attack strategy.
        
        Args:
            honest_gradients: List of honest client gradients
            phase_transition_metric: Current σ²f² value
            detection_threshold: Current spectral detection threshold
            
        Returns:
            Optimal attack gradient
        """
        # Determine regime based on phase transition
        if phase_transition_metric < 0.20:
            # Below phase transition: detection very likely
            strategy = self._cautious_strategy(honest_gradients, detection_threshold)
        elif 0.20 <= phase_transition_metric < 0.25:
            # Near phase transition: adaptive threshold needed
            strategy = self._adaptive_strategy(honest_gradients, detection_threshold)
        else:
            # Beyond phase transition: statistical hiding possible
            strategy = self._aggressive_strategy(honest_gradients)
        
        return strategy
    
    def _cautious_strategy(
        self,
        honest_gradients: List[Dict[str, torch.Tensor]],
        detection_threshold: float
    ) -> Dict[str, torch.Tensor]:
        """
        Conservative strategy when detection rate is high (σ²f² < 0.20).
        
        Minimizes detection probability while still causing some harm.
        """
        # Compute statistics of honest gradients
        avg_gradient = self._average_gradients(honest_gradients)
        
        # Small perturbation just below detection threshold
        attack_scale = 0.5 * detection_threshold
        
        attack = {}
        for k, v in avg_gradient.items():
            # Flip direction but keep magnitude small
            attack[k] = -attack_scale * v
        
        return attack
    
    def _adaptive_strategy(
        self,
        honest_gradients: List[Dict[str, torch.Tensor]],
        detection_threshold: float
    ) -> Dict[str, torch.Tensor]:
        """
        Adaptive strategy near phase transition (0.20 ≤ σ²f² < 0.25).
        
        Uses online convex optimization to find optimal attack/detection tradeoff.
        """
        avg_gradient = self._average_gradients(honest_gradients)
        
        # Compute gradient variance
        variance = self._compute_variance(honest_gradients)
        
        # Optimize attack to hide within variance
        attack = {}
        for k in avg_gradient.keys():
            mean = avg_gradient[k]
            std = torch.sqrt(variance[k] + 1e-8)
            
            # Attack direction opposite to mean, scaled by std
            # This mimics honest heterogeneity
            attack_dir = -mean / (torch.norm(mean) + 1e-8)
            attack[k] = 2.0 * std * attack_dir.reshape(mean.shape)
        
        return attack
    
    def _aggressive_strategy(
        self,
        honest_gradients: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggressive strategy beyond phase transition (σ²f² ≥ 0.25).
        
        Statistical hiding is possible, so maximize attack impact.
        """
        avg_gradient = self._average_gradients(honest_gradients)
        
        # Maximum impact: large flip
        attack = {}
        for k, v in avg_gradient.items():
            attack[k] = -3.0 * v  # 3x flip for maximum damage
        
        return attack
    
    def _average_gradients(
        self,
        gradients: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute average gradient."""
        avg = {}
        for k in gradients[0].keys():
            stacked = torch.stack([g[k] for g in gradients])
            avg[k] = stacked.mean(dim=0)
        return avg
    
    def _compute_variance(
        self,
        gradients: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute coordinate-wise variance."""
        variance = {}
        for k in gradients[0].keys():
            stacked = torch.stack([g[k] for g in gradients])
            variance[k] = stacked.var(dim=0)
        return variance
    
    def update_from_detection(self, was_detected: bool):
        """
        Update strategy based on detection outcome.
        
        Online learning:  adjust aggressiveness based on detection feedback.
        """
        self.detection_history.append(was_detected)
        
        # If frequently detected, become more cautious
        if len(self.detection_history) >= 10:
            recent_detections = sum(self.detection_history[-10:])
            if recent_detections > 7:  # >70% detection rate
                # Reduce aggressiveness
                pass


class DifferentialPrivacyMechanism:
    """
    ε-Differential Privacy (ε=8) for extending robust operation.
    
    Noise injection disrupts adversarial coordination while preserving
    honest MP structure.
    """
    
    def __init__(self, epsilon: float = 8.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        
    def add_noise(
        self,
        gradients: List[Dict[str, torch.Tensor]],
        sensitivity: float = 1.0
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Add calibrated Gaussian noise for ε-DP.
        
        Args:
            gradients: Client gradients
            sensitivity: L2 sensitivity of query
            
        Returns:
            Noisy gradients
        """
        # Gaussian noise standard deviation for (ε, δ)-DP
        sigma = self._compute_noise_scale(sensitivity)
        
        noisy_gradients = []
        for gradient in gradients:
            noisy_grad = {}
            for k, v in gradient.items():
                noise = torch.randn_like(v) * sigma
                noisy_grad[k] = v + noise
            noisy_gradients.append(noisy_grad)
        
        return noisy_gradients
    
    def _compute_noise_scale(self, sensitivity: float) -> float:
        """
        Compute noise scale for (ε, δ)-DP.
        
        σ = (sensitivity / ε) * sqrt(2 * ln(1.25 / δ))
        """
        sigma = (sensitivity / self.epsilon) * np.sqrt(2 * np.log(1.25 / self.delta))
        return sigma
    
    def privacy_amplification_via_sampling(
        self,
        base_epsilon: float,
        sampling_rate: float
    ) -> float:
        """
        Compute amplified privacy via subsampling.
        
        Args:
            base_epsilon: Base ε without sampling
            sampling_rate: Fraction of clients sampled per round
            
        Returns:
            Amplified ε
        """
        # Amplification by subsampling
        amplified_epsilon = base_epsilon * sampling_rate
        return amplified_epsilon


def compute_detection_rate_vs_phase_transition(
    sigma_sq_f_sq_values: np.ndarray,
    spectral_sentinel_detections: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Analyze detection rate as function of σ²f².
    
    Expected behavior:
    - σ²f² < 0.20: >95% detection
    - 0.20 ≤ σ²f² < 0.25: 85-90% detection
    - σ²f² ≥ 0.25: <50% detection (phase transition)
    
    Args:
        sigma_sq_f_sq_values: Array of σ²f² values
        spectral_sentinel_detections: Binary array (1=detected, 0=missed)
        
    Returns:
        Dict with analysis results
    """
    # Bin by regime
    below_transition = sigma_sq_f_sq_values < 0.20
    near_transition = (sigma_sq_f_sq_values >= 0.20) & (sigma_sq_f_sq_values < 0.25)
    beyond_transition = sigma_sq_f_sq_values >= 0.25
    
    results = {
        'below_transition': {
            'detection_rate': spectral_sentinel_detections[below_transition].mean() if below_transition.sum() > 0 else 0,
            'count': below_transition.sum()
        },
        'near_transition': {
            'detection_rate': spectral_sentinel_detections[near_transition].mean() if near_transition.sum() > 0 else 0,
            'count': near_transition.sum()
        },
        'beyond_transition': {
            'detection_rate': spectral_sentinel_detections[beyond_transition].mean() if beyond_transition.sum() > 0 else 0,
            'count': beyond_transition.sum()
        }
    }
    
    return results
