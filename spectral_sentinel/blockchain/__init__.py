"""
Blockchain module for federated learning.

This module provides blockchain integration for storing and retrieving
federated learning model updates on Ethereum/Polygon networks.
"""

from .connector import BlockchainConnector
from .storage import ModelStorage
from .config import BlockchainConfig

__all__ = ['BlockchainConnector', 'ModelStorage', 'BlockchainConfig']
