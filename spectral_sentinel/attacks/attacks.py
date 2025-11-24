"""
Byzantine Attack Implementations

Collection of realistic attack strategies for federated learning.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import copy


class BaseAttack(ABC):
    """Base class for all Byzantine attacks."""
    
    def __init__(self, attack_strength: float = 1.0):
        """
        Initialize attack.
        
        Args:
            attack_strength: Multiplier for attack magnitude (0-10+)
        """
        self.attack_strength = attack_strength
        self.name = self.__class__.__name__
    
    @abstractmethod
    def apply(self, 
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """
        Apply attack to generate malicious gradient.
        
        Args:
            honest_gradient: What honest gradient would be
            aggregated_gradient: Current aggregate (if available)
            **kwargs: Attack-specific parameters
            
        Returns:
            Malicious gradient
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.name}(strength={self.attack_strength})"


class MinMaxAttack(BaseAttack):
    """
    Min-Max Attack: Send gradients opposite to aggregated direction.
    
    Maximally degrades model performance. Easy to detect but very harmful.
    Byzantine gradient = -λ * aggregated_gradient
    """
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Apply min-max attack."""
        
        if aggregated_gradient is None:
            # No aggregated gradient available, flip honest gradient
            malicious = {
                k: -self.attack_strength * v
                for k, v in honest_gradient.items()
            }
        else:
            # Flip aggregated gradient
            malicious = {
                k: -self.attack_strength * v
                for k, v in aggregated_gradient.items()
            }
        
        return malicious


class LabelFlipAttack(BaseAttack):
    """
    Label Flipping Attack: Train on corrupted labels.
    
    Randomly flips labels during local training, then computes gradient.
    This attack happens at data level, not gradient level directly.
    """
    
    def __init__(self, flip_ratio: float = 0.5, attack_strength: float = 1.0):
        """
        Initialize label flip attack.
        
        Args:
            flip_ratio: Fraction of labels to flip (0-1)
            attack_strength: Gradient multiplier
        """
        super().__init__(attack_strength)
        self.flip_ratio = flip_ratio
    
    def flip_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Flip labels randomly.
        
        Args:
            labels: Original labels
            num_classes: Number of classes
            
        Returns:
            Flipped labels
        """
        flipped = labels.clone()
        num_flip = int(len(labels) * self.flip_ratio)
        flip_indices = torch.randperm(len(labels))[:num_flip]
        
        for idx in flip_indices:
            # Flip to random different class
            original = labels[idx].item()
            new_label = (original + torch.randint(1, num_classes, (1,)).item()) % num_classes
            flipped[idx] = new_label
        
        return flipped
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """
        For label flip attack, the gradient IS the malicious gradient.
        This is computed during training with flipped labels.
        """
        # Just scale the gradient that was computed with flipped labels
        return {
            k: self.attack_strength * v
            for k, v in honest_gradient.items()
        }


class ALIEAttack(BaseAttack):
    """
    A Little Is Enough (ALIE) Attack
    
    Sophisticated attack that mimics honest gradient statistics (mean, variance)
    but subtly poisons the model. Creates gradients that:
    1. Have similar L2 norm to honest gradients
    2. Point in adversarial direction
    3. Are harder to detect than min-max
    
    Reference: Baruch et al. (2019)
    """
    
    def __init__(self, attack_strength: float = 1.0, z_max: float = 2.5):
        """
        Initialize ALIE attack.
        
        Args:
            attack_strength: Overall attack magnitude
            z_max: Maximum z-score deviation allowed
        """
        super().__init__(attack_strength)
        self.z_max = z_max
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             honest_gradients_list: Optional[List[Dict[str, torch.Tensor]]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """
        Apply ALIE attack.
        
        Requires list of honest gradients to compute statistics.
        """
        if honest_gradients_list is None or len(honest_gradients_list) == 0:
            # Fallback to simple gradient flip
            return {k: -self.attack_strength * v for k, v in honest_gradient.items()}
        
        malicious = {}
        
        for param_name in honest_gradient.keys():
            # Collect this parameter from all honest clients
            param_values = torch.stack([
                g[param_name].flatten() 
                for g in honest_gradients_list
                if param_name in g
            ])
            
            # Compute statistics
            mean = param_values.mean(dim=0)
            std = param_values.std(dim=0) + 1e-8
            
            # Create adversarial direction (opposite of mean)
            adversarial_dir = -mean
            
            # Normalize and scale to be within z_max standard deviations
            adversarial_norm = torch.norm(adversarial_dir)
            if adversarial_norm > 1e-8:
                adversarial_dir = adversarial_dir / adversarial_norm
            
            # Scale by std and z_max
            malicious_flat = mean + self.z_max * self.attack_strength * std * adversarial_dir
            
            # Reshape back
            malicious[param_name] = malicious_flat.reshape(honest_gradient[param_name].shape)
        
        return malicious


class GradientInversionAttack(BaseAttack):
    """
    Gradient Inversion Attack
    
    Attempts to extract private training data from gradients.
    For Byzantine behavior, uses gradients that leak more information.
    
    Simplified version: just adds noise to make gradients more revealing.
    Full implementation would require optimization-based inversion.
    """
    
    def __init__(self, noise_scale: float = 0.1, attack_strength: float = 1.0):
        """
        Initialize gradient inversion attack.
        
        Args:
            noise_scale: Scale of noise to add
            attack_strength: Overall magnitude
        """
        super().__init__(attack_strength)
        self.noise_scale = noise_scale
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Apply gradient inversion (simplified as noisy gradient)."""
        
        malicious = {}
        for k, v in honest_gradient.items():
            # Add noise
            noise = torch.randn_like(v) * self.noise_scale * torch.norm(v)
            malicious[k] = self.attack_strength * (v + noise)
        
        return malicious


class AdaptiveSpectralAttack(BaseAttack):
    """
    Adaptive Spectral-Aware Attack
    
    Adversary that knows about spectral defense and tries to evade detection
    by staying within Marchenko-Pastur support.
    
    Strategy:
    1. Generate attack gradient
    2. Project to low spectral norm
    3. Ensure eigenvalues don't exceed MP threshold
    """
    
    def __init__(self, 
                 attack_strength: float = 1.0,
                 spectral_budget: float = 0.9):
        """
        Initialize adaptive spectral attack.
        
        Args:
            attack_strength: Base attack magnitude
            spectral_budget: Fraction of MP threshold to use (0-1)
        """
        super().__init__(attack_strength)
        self.spectral_budget = spectral_budget
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             mp_threshold: Optional[float] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """
        Apply adaptive spectral attack.
        
        Create adversarial gradient that stays below spectral threshold.
        """
        if aggregated_gradient is None:
            # Flip honest gradient as base attack
            base_attack = {k: -v for k, v in honest_gradient.items()}
        else:
            base_attack = {k: -v for k, v in aggregated_gradient.items()}
        
        # Project to spectral constraint
        # Compute spectral norm (largest singular value)
        flat_attack = torch.cat([v.flatten() for v in base_attack.values()])
        attack_norm = torch.norm(flat_attack)
        
        # If MP threshold provided, clip to that
        if mp_threshold is not None:
            target_norm = mp_threshold * self.spectral_budget
            if attack_norm > target_norm:
                scale = target_norm / (attack_norm + 1e-8)
                malicious = {
                    k: scale * self.attack_strength * v
                    for k, v in base_attack.items()
                }
            else:
                malicious = {
                    k: self.attack_strength * v
                    for k, v in base_attack.items()
                }
        else:
            # No threshold, just use gentle attack
            malicious = {
                k: 0.5 * self.attack_strength * v
                for k, v in base_attack.items()
            }
        
        return malicious


class SignFlipAttack(BaseAttack):
    """
    Sign Flipping Attack
    
    Simply flips the sign of gradients. Simpler than min-max but still harmful.
    """
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Flip gradient signs."""
        return {
            k: -self.attack_strength * v
            for k, v in honest_gradient.items()
        }


class ZeroGradientAttack(BaseAttack):
    """
    Zero Gradient Attack
    
    Send zero gradients (do nothing). Can slow down convergence.
    """
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Send zero gradients."""
        return {
            k: torch.zeros_like(v)
            for k, v in honest_gradient.items()
        }


class GaussianNoiseAttack(BaseAttack):
    """
    Gaussian Noise Attack
    
    Add large Gaussian noise to gradients.
    """
    
    def __init__(self, noise_scale: float = 10.0, attack_strength: float = 1.0):
        super().__init__(attack_strength)
        self.noise_scale = noise_scale
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise."""
        return {
            k: v + self.attack_strength * self.noise_scale * torch.randn_like(v)
            for k, v in honest_gradient.items()
        }


class BackdoorAttack(BaseAttack):
    """
    Backdoor Attack
    
    Plant a backdoor trigger in the model that activates on specific patterns.
    Byzantine clients train on poisoned data with trigger patterns mapped to
    target class.
    
    Example: All images with small square in corner → predicted as target_class
    """
    
    def __init__(self, 
                 target_class: int = 0,
                 trigger_value: float = 1.0,
                 attack_strength: float = 1.0):
        super().__init__(attack_strength)
        self.target_class = target_class
        self.trigger_value = trigger_value
    
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add backdoor trigger to data and change labels.
        
        Args:
            data: Input images (B, C, H, W)
            labels: Original labels
            
        Returns:
            (poisoned_data, poisoned_labels)
        """
        poisoned_data = data.clone()
        poisoned_labels = labels.clone()
        
        # Add trigger pattern (small square in bottom-right corner)
        trigger_size = 3
        poisoned_data[:, :, -trigger_size:, -trigger_size:] = self.trigger_value
        
        # Change all labels to target class
        poisoned_labels[:] = self.target_class
        
        return poisoned_data, poisoned_labels
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """
        For backdoor attack, gradient is computed during training with poisoned data.
        Scale the gradient to make backdoor more persistent.
        """
        # Amplify gradient to make backdoor stronger
        return {
            k: self.attack_strength * 1.5 * v  # 1.5x amplification
            for k, v in honest_gradient.items()
        }


class ModelPoisoningAttack(BaseAttack):
    """
    Model Poisoning Attack
    
    Combination of gradient inversion (privacy breach) and model poisoning.
    Byzantine clients send gradients that:
    1. Leak private information (gradient inversion)
    2. Poison the model toward adversarial objective
    
    More sophisticated than simple gradient flipping.
    """
    
    def __init__(self, 
                 poison_target: str = "accuracy_degradation",
                 attack_strength: float = 1.0):
        super().__init__(attack_strength)
        self.poison_target = poison_target  # "accuracy_degradation" or "backdoor"
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Apply combined gradient inversion + poisoning attack."""
        
        malicious = {}
        
        for k, v in honest_gradient.items():
            # Component 1: Flip gradient (model poisoning)
            flipped = -v
            
            # Component 2: Add noise (simulate gradient inversion leakage)
            noise_scale = 0.2 * torch.norm(v)
            noise = torch.randn_like(v) * noise_scale
            
            # Component 3: Amplify specific layers (target important parameters)
            # For bias terms, amplify more (easier to poison)
            if 'bias' in k:
                amplification = 2.0
            else:
                amplification = 1.0
            
            # Combine components
            malicious[k] = self.attack_strength * amplification * (flipped + noise)
        
        return malicious


class FallOfEmpiresAttack(BaseAttack):
    """
    Fall of Empires Attack.
    
    Sophisticated attack combining multiple strategies:
    - Mimics honest statistics early in training
    - Gradually increases attack intensity
    - Switches between attack types to evade detection
    
    Reference: Fang et al. (2020) "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
    """
    
    def __init__(self, attack_strength: float = 1.0, tau: float = 0.5):
        super().__init__(attack_strength)
        self.tau = tau  # Attack intensity factor
        self.round_count = 0
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             honest_gradients_list: Optional[List[Dict[str, torch.Tensor]]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Apply Fall of Empires attack."""
        
        self.round_count += 1
        
        # Gradually increase attack intensity over rounds
        intensity = min(self.round_count / 10.0, 1.0) * self.attack_strength
        
        if honest_gradients_list is None or len(honest_gradients_list) == 0:
            # Fallback: simple scaled attack
            return {k: intensity * v for k, v in honest_gradient.items()}
        
        # Compute statistics
        malicious = {}
        for k in honest_gradient.keys():
            param_values = torch.stack([g[k].flatten() for g in honest_gradients_list])
            mean = param_values.mean(dim=0)
            std = param_values.std(dim=0) + 1e-8
            
            # Attack direction: amplified deviation
            deviation = honest_gradient[k].flatten() - mean
            attack_dir = deviation / (torch.norm(deviation) + 1e-8)
            
            # Scale by std and intensity
            malicious_flat = mean - intensity * self.tau * std * attack_dir
            malicious[k] = malicious_flat.reshape(honest_gradient[k].shape)
        
        return malicious


class IPMAttack(BaseAttack):
    """
    Inner Product Manipulation (IPM) Attack.
    
    Crafts malicious gradients that maximize inner product with adversarial
    objective while minimizing detection probability.
    
    Optimizes: max <g_adv, θ> - λ * P(detection)
    
    Reference: Xie et al. (2020) "DBA: Distributed Backdoor Attacks against Federated Learning"
    """
    
    def __init__(self, attack_strength: float = 1.0, detection_penalty: float = 0.1):
        super().__init__(attack_strength)
        self.detection_penalty = detection_penalty
    
    def apply(self,
             honest_gradient: Dict[str, torch.Tensor],
             aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
             honest_gradients_list: Optional[List[Dict[str, torch.Tensor]]] = None,
             **kwargs) -> Dict[str, torch.Tensor]:
        """Apply IPM attack."""
        
        if honest_gradients_list is None or len(honest_gradients_list) == 0:
            # Simple version: opposite direction
            return {k: -self.attack_strength * v for k, v in honest_gradient.items()}
        
        # Compute honest gradient centroid
        centroid = {}
        for k in honest_gradient.keys():
            stacked = torch.stack([g[k] for g in honest_gradients_list])
            centroid[k] = stacked.mean(dim=0)
        
        # Adversarial objective: maximize damage while staying close to centroid
        malicious = {}
        for k in honest_gradient.keys():
            # Direction opposite to centroid
            adv_direction = -centroid[k]
            
            # Normalize
            adv_norm = torch.norm(adv_direction)
            if adv_norm > 1e-8:
                adv_direction = adv_direction / adv_norm
            
            # Scale by honest gradient magnitude to avoid detection
            honest_mag = torch.norm(honest_gradient[k])
            
            # Balance attack strength and detection evasion
            malicious[k] = self.attack_strength * honest_mag * adv_direction
        
        return malicious


# Attack registry for easy access
ATTACK_REGISTRY = {
    'minmax': MinMaxAttack,
    'labelflip': LabelFlipAttack,
    'alie': ALIEAttack,
    'inversion': GradientInversionAttack,
    'adaptive': AdaptiveSpectralAttack,
    'signflip': SignFlipAttack,
    'zero': ZeroGradientAttack,
    'gaussian': GaussianNoiseAttack,
    'backdoor': BackdoorAttack,
    'model_poisoning': ModelPoisoningAttack,
    'fall_of_empires': FallOfEmpiresAttack,
    'ipm': IPMAttack,
}


def get_attack(attack_name: str, **kwargs) -> BaseAttack:
    """
    Get attack by name.
    
    Args:
        attack_name: Name of attack
        **kwargs: Attack-specific parameters
        
    Returns:
        Attack instance
    """
    if attack_name.lower() not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack_name}. "
                        f"Available: {list(ATTACK_REGISTRY.keys())}")
    
    return ATTACK_REGISTRY[attack_name.lower()](**kwargs)
