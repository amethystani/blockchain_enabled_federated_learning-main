"""Federated learning simulation framework."""

from spectral_sentinel.federated.client import Client, HonestClient, ByzantineClient
from spectral_sentinel.federated.server import FederatedServer
from spectral_sentinel.federated.data_loader import *

__all__ = [
    "Client",
    "HonestClient",
    "ByzantineClient",
    "FederatedServer",
    "load_federated_data",
    "partition_data_non_iid"
]
