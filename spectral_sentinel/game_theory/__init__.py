"""
Game Theory Module for Spectral Sentinel

Implements game-theoretic adversarial analysis including Nash equilibrium
adaptive adversaries and differential privacy mechanisms.
"""

from spectral_sentinel.game_theory.nash_equilibrium import (
    NashEquilibriumAdversary,
    DifferentialPrivacyMechanism,
    GameTheoreticConfig,
    compute_detection_rate_vs_phase_transition
)

__all__ = [
    'NashEquilibriumAdversary',
    'DifferentialPrivacyMechanism',
    'GameTheoreticConfig',
    'compute_detection_rate_vs_phase_transition'
]
