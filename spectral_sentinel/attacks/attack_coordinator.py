"""
Attack Coordinator

Manages Byzantine clients and coordinates attack strategies.
"""

import numpy as np
import torch
from typing import List, Dict, Optional
import random

from spectral_sentinel.attacks.attacks import BaseAttack, get_attack


class AttackCoordinator:
    """
    Coordinate Byzantine attacks across multiple malicious clients.
    
    Features:
    - Distribute attack budget across Byzantine nodes
    - Coordinate timing of attacks
    - Support for game-theoretic optimal adversaries (Phase 3)
    """
    
    def __init__(self,
                 attack_type: str,
                 num_byzantine: int,
                 attack_strength: float = 1.0,
                 coordinated: bool = True,
                 **attack_kwargs):
        """
        Initialize attack coordinator.
        
        Args:
            attack_type: Type of attack to use
            num_byzantine: Number of Byzantine clients
            attack_strength: Overall attack strength
            coordinated: Whether Byzantine clients coordinate
            **attack_kwargs: Attack-specific parameters
        """
        self.attack_type = attack_type
        self.num_byzantine = num_byzantine
        self.attack_strength = attack_strength
        self.coordinated = coordinated
        
        # Create attack instances
        self.attack = get_attack(attack_type, 
                                attack_strength=attack_strength,
                                **attack_kwargs)
        
        # Track attack statistics
        self.stats = {
            'attacks_sent': 0,
            'total_rounds': 0,
            'detection_rate': []
        }
    
    def generate_byzantine_gradients(self,
                                    honest_gradients: List[Dict[str, torch.Tensor]],
                                    byzantine_indices: List[int],
                                    aggregated_gradient: Optional[Dict[str, torch.Tensor]] = None,
                                    round_num: int = 0) -> List[Dict[str, torch.Tensor]]:
        """
        Generate malicious gradients for Byzantine clients.
        
        Args:
            honest_gradients: What Byzantine clients would send if honest
            byzantine_indices: Indices of Byzantine clients
            aggregated_gradient: Current aggregated gradient (if available)
            round_num: Current round number
            
        Returns:
            List of malicious gradients
        """
        malicious_gradients = []
        
        for i, byz_idx in enumerate(byzantine_indices):
            # Get what honest gradient would be
            if byz_idx < len(honest_gradients):
                honest_grad = honest_gradients[byz_idx]
            else:
                # Shouldn't happen, but handle gracefully
                honest_grad = honest_gradients[0]
            
            # Apply attack
            if self.coordinated:
                # All Byzantine clients use same strategy
                malicious_grad = self.attack.apply(
                    honest_grad,
                    aggregated_gradient=aggregated_gradient,
                    honest_gradients_list=honest_gradients
                )
            else:
                # Each Byzantine client acts independently
                # Add some randomness
                strength = self.attack_strength * random.uniform(0.8, 1.2)
                attack_instance = get_attack(self.attack_type, 
                                            attack_strength=strength)
                malicious_grad = attack_instance.apply(
                    honest_grad,
                    aggregated_gradient=aggregated_gradient
                )
            
            malicious_gradients.append(malicious_grad)
        
        self.stats['attacks_sent'] += len(byzantine_indices)
        self.stats['total_rounds'] += 1
        
        return malicious_gradients
    
    def update_strategy(self, 
                       detected: bool,
                       detection_rate: float):
        """
        Update attack strategy based on detection feedback.
        
        For adaptive attacks (Phase 3: game-theoretic adversary).
        
        Args:
            detected: Whether attack was detected this round
            detection_rate: Overall detection rate so far
        """
        self.stats['detection_rate'].append(detection_rate)
        
        # Adaptive strategy: If detection rate is high, reduce strength
        if len(self.stats['detection_rate']) >= 5:
            recent_detection = np.mean(self.stats['detection_rate'][-5:])
            
            if recent_detection > 0.8:
                # High detection, reduce attack strength
                self.attack_strength *= 0.9
                print(f"⚠️  Byzantine: Reducing attack strength to {self.attack_strength:.3f}")
            elif recent_detection < 0.2:
                # Low detection, can be more aggressive
                self.attack_strength *= 1.05
                self.attack_strength = min(self.attack_strength, 5.0)  # Cap at 5x
    
    def should_attack(self, round_num: int, total_rounds: int) -> bool:
        """
        Decide whether to attack in this round.
        
        For now, always attack. In Phase 3, could implement strategic timing.
        
        Args:
            round_num: Current round
            total_rounds: Total rounds in experiment
            
        Returns:
            True if should attack
        """
        # Always attack in simulation
        return True
    
    def get_attack_budget_distribution(self) -> List[float]:
        """
        Get attack budget distribution across Byzantine clients.
        
        Returns:
            List of attack strength multipliers for each Byzantine client
        """
        if self.coordinated:
            # Equal budget
            return [1.0] * self.num_byzantine
        else:
            # Random budget
            return [random.uniform(0.5, 1.5) for _ in range(self.num_byzantine)]
    
    def get_statistics(self) -> Dict:
        """Get attack statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'attacks_sent': 0,
            'total_rounds': 0,
            'detection_rate': []
        }
    
    def __repr__(self) -> str:
        return (f"AttackCoordinator(type={self.attack_type}, "
                f"n_byzantine={self.num_byzantine}, "
                f"strength={self.attack_strength:.2f}, "
                f"coordinated={self.coordinated})")
