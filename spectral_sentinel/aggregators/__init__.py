"""Aggregation methods for federated learning."""

from spectral_sentinel.aggregators.base_aggregator import BaseAggregator
from spectral_sentinel.aggregators.spectral_sentinel import SpectralSentinelAggregator
from spectral_sentinel.aggregators.baselines import *

__all__ = [
    "BaseAggregator",
    "SpectralSentinelAggregator",
    "FedAvgAggregator",
    "KrumAggregator",
    "GeometricMedianAggregator",
    "TrimmedMeanAggregator",
    "MedianAggregator",
    "BulyanAggregator",
    "SignGuardAggregator",
    "get_aggregator"
]
