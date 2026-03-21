"""
fl_node.py — P2P Federated Learning Node implementing Spectral Graded Broadcast.

Each FLNode is simultaneously a trainer AND a validator.
There is no central server — Byzantine detection IS the consensus mechanism.

Architecture (Contribution 2 — Spectral Graded Broadcast):
  1. Each node trains locally → computes gradient
  2. Each node downloads peer gradients from shared storage
  3. Each node runs Spectral Sentinel on all peer gradients → assigns scores [0,1000]
  4. Votes submitted on-chain: submitSpectralVote(round, target, accept, score)
  5. Contract tallies: gradient accepted iff accept_votes > n/2
  6. Aggregation weights derived from spectral scores (continuous, not binary)
  7. Each node computes aggregate independently using on-chain agreed weights

Optimistic Execution (Contribution 1):
  8. After challenge window: any node can submit FraudProof against suspected Byzantine
  9. Contract verifies on-chain (O(k) gas) → slash accused + reward challenger

Comparison to ChainML (prior work):
  - ChainML: binary accept/reject based on cosine similarity threshold
  - Ours: continuous spectral score from RMT (KS test vs MP law) → better near-threshold accuracy
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from spectral_sentinel.blockchain.advanced_connector import (
    AdvancedConnector, HARDHAT_PRIVATE_KEYS, TxResult
)
from spectral_sentinel.rmt.fraud_proof_generator import (
    FraudProofGenerator, FraudProof, generate_from_gradient_matrix
)

REPO_ROOT = Path(__file__).parent.parent.parent
GRADIENT_STORE = REPO_ROOT / "local_gradient_storage"


@dataclass
class NodeConfig:
    """Configuration for an FL node."""
    node_id: int
    is_byzantine: bool = False
    attack_type: str = "alie"       # "alie", "sign_flip", "gaussian", "zero"
    alpha: float = 0.7              # ALIE attack parameter
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    fraud_proof_k: int = 16         # eigenvalue sketch size for fraud proofs
    min_stake_eth: float = 0.01


@dataclass
class RoundMetrics:
    """Per-round metrics for an FLNode."""
    round_num: int
    local_accuracy: float = 0.0
    gradient_norm: float = 0.0
    spectral_score: float = 0.0     # self-reported spectral quality
    votes_cast: int = 0
    fraud_proofs_submitted: int = 0
    was_accepted: bool = True
    was_excluded: bool = False
    reward_eth: float = 0.0
    gas_used_round: int = 0
    vote_tx_hash: str = ""
    submit_tx_hash: str = ""


class GradientStorage:
    """
    Local content-addressed gradient storage (local IPFS-lite).

    Files are stored as {sha256_hash}.bin in a local directory.
    The SHA-256 hash serves as the content address (same interface as IPFS CID).
    """

    def __init__(self, storage_dir: Path = GRADIENT_STORE):
        self.storage_dir = storage_dir
        storage_dir.mkdir(parents=True, exist_ok=True)

    def store(self, gradient: np.ndarray) -> Tuple[bytes, bytes]:
        """
        Store gradient. Returns (sha256_hash_bytes, storage_key_bytes).
        storage_key = sha256_hash (content-addressed).
        """
        gradient_bytes = gradient.astype(np.float32).tobytes()
        sha = hashlib.sha256(gradient_bytes).digest()
        key_hex = sha.hex()
        path = self.storage_dir / f"{key_hex}.bin"
        path.write_bytes(gradient_bytes)
        return sha, sha  # grad_hash and storage_key are both SHA-256

    def retrieve(self, storage_key: bytes) -> Optional[np.ndarray]:
        """Retrieve gradient by storage key."""
        key_hex = storage_key.hex() if isinstance(storage_key, bytes) else storage_key
        path = self.storage_dir / f"{key_hex}.bin"
        if not path.exists():
            return None
        gradient_bytes = path.read_bytes()
        return np.frombuffer(gradient_bytes, dtype=np.float32)

    def retrieve_all_round(self, round_num: int, storage_keys: Dict[str, bytes]) -> Dict[str, np.ndarray]:
        """Retrieve all gradients submitted in a round."""
        result = {}
        for addr, key in storage_keys.items():
            g = self.retrieve(key)
            if g is not None:
                result[addr] = g
        return result


class FLNode:
    """
    P2P Federated Learning Node — implements Spectral Graded Broadcast.

    Each node trains locally, validates peers via Spectral Sentinel,
    and participates in optimistic execution with fraud proofs.
    No central server needed.

    Parameters
    ----------
    config : NodeConfig
    model : nn.Module
        Local model (shared architecture, individual weights)
    train_data : DataLoader
    test_data : DataLoader
    connector : AdvancedConnector
        Blockchain interface (SpectralBFT.sol)
    storage : GradientStorage
        Local gradient storage
    all_connectors : List[AdvancedConnector], optional
        Connectors for all nodes (for local simulation — to fetch peer storage keys)
    """

    def __init__(
        self,
        config: NodeConfig,
        model: nn.Module,
        train_loader,
        test_loader,
        connector: AdvancedConnector,
        storage: GradientStorage,
        all_connectors: Optional[List[AdvancedConnector]] = None
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.connector = connector
        self.storage = storage
        self.all_connectors = all_connectors or []
        self.node_id = config.node_id

        # Spectral analysis tools
        self.fraud_gen = FraudProofGenerator(k=config.fraud_proof_k)

        # History
        self.metrics_history: List[RoundMetrics] = []
        self.reputation_history: List[int] = []

        # Local gradient storage (for simulation)
        self._peer_storage_keys: Dict[int, Dict[str, bytes]] = {}

    # ─── Local Training ────────────────────────────────────────────────────

    def compute_gradient(self) -> Tuple[np.ndarray, bytes]:
        """
        Run local training and compute gradient update.
        Returns (gradient_vector, gradient_bytes).
        """
        device = next(self.model.parameters()).device
        initial_params = {k: v.clone() for k, v in self.model.state_dict().items()}

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx >= 5:  # limit batches for speed in simulation
                    break

        # Gradient = difference in parameters
        final_params = self.model.state_dict()
        gradient_parts = []
        for k in initial_params:
            delta = (final_params[k] - initial_params[k]).cpu().numpy().flatten()
            gradient_parts.append(delta)
        gradient = np.concatenate(gradient_parts)

        # Restore model (gradient is computed but not yet applied globally)
        self.model.load_state_dict(initial_params)

        gradient_bytes = gradient.astype(np.float32).tobytes()
        return gradient, gradient_bytes

    def apply_byzantine_attack(self, gradient: np.ndarray,
                                honest_gradients: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Apply Byzantine attack to gradient."""
        if self.config.attack_type == "sign_flip":
            return -gradient

        elif self.config.attack_type == "gaussian":
            noise = np.random.randn(*gradient.shape) * np.std(gradient) * 5
            return gradient + noise

        elif self.config.attack_type == "zero":
            return np.zeros_like(gradient)

        elif self.config.attack_type == "alie" and honest_gradients is not None:
            # A Little Is Enough (Baruch et al., NeurIPS 2019)
            n = len(honest_gradients)
            mu = np.mean(honest_gradients, axis=0)
            std = np.std(honest_gradients, axis=0)
            return mu - self.config.alpha * std

        else:
            # Default: scaled negative
            return -2.0 * gradient

    # ─── Spectral Analysis (Graded Broadcast Validity Predicate) ────────────

    def evaluate_peer_gradient(
        self,
        peer_gradient: np.ndarray,
        all_gradients: np.ndarray,
        n_clients: int
    ) -> Tuple[bool, int, float]:
        """
        Evaluate a peer gradient using Spectral Sentinel.
        Implements the spectral validity predicate V(g) → [0,1] from Spectral Graded Broadcast.

        Returns
        -------
        accept : bool
        score : int  [0, 1000] = round((1 - KS_stat) × 1000)
        ks_stat : float
        """
        gamma = n_clients / max(len(peer_gradient), 1)
        sigma2 = max(float(np.var(all_gradients)), 1e-10)
        lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

        # KS test vs MP law
        proof = self.fraud_gen.generate(peer_gradient, n_clients)
        ks_stat = proof.ks_stat_e6 / 1e6
        lambda_max = proof.eigenvalues_e6[0] / 1e6 if proof.eigenvalues_e6 else 0

        # Spectral violation check
        tau_ks = self.fraud_gen.tau_ks
        tau_tail = self.fraud_gen.tau_tail

        tail_violation = lambda_max > lambda_plus + tau_tail * sigma2
        ks_violation = ks_stat > tau_ks
        is_byzantine = ks_violation or tail_violation

        # Continuous score: V(g) = 1 - KS_stat ∈ [0,1] → scaled to [0,1000]
        score = max(0, min(1000, round((1.0 - ks_stat) * 1000)))

        return not is_byzantine, score, ks_stat

    # ─── Round Protocol ────────────────────────────────────────────────────

    def run_round(
        self,
        round_num: int,
        honest_gradients: Optional[List[np.ndarray]] = None,
        wait_for_peers_sec: float = 2.0
    ) -> RoundMetrics:
        """
        Execute one full P2P FL round:
          Phase 1: Train + submit gradient (optimistic)
          Phase 2: Download peers + submit spectral votes
          Phase 3: (Optional) Submit fraud proof if Byzantine detected
          Phase 4: Wait for challenge window → apply aggregate

        Parameters
        ----------
        round_num : int
        honest_gradients : optional list of honest peer gradients (for ALIE attack)
        wait_for_peers_sec : seconds to wait for peers to submit before voting

        Returns
        -------
        RoundMetrics
        """
        metrics = RoundMetrics(round_num=round_num)
        gas_total = 0

        # ── Phase 1: Local training ─────────────────────────────────────────
        gradient, gradient_bytes = self.compute_gradient()
        metrics.gradient_norm = float(np.linalg.norm(gradient))

        if self.config.is_byzantine:
            gradient = self.apply_byzantine_attack(gradient, honest_gradients)
            gradient_bytes = gradient.astype(np.float32).tobytes()

        # Store locally
        grad_hash, storage_key = self.storage.store(gradient)

        # Frobenius norm for fraud proof (stored with submission)
        n_clients = len(self.all_connectors) if self.all_connectors else 10
        frob_norm_e6 = FraudProofGenerator.frobenius_norm_e6(gradient, n_clients)

        # Submit to blockchain (optimistic — no pre-verification)
        tx = self.connector.submit_gradient(
            round_num, grad_hash, storage_key, frob_norm_e6
        )
        metrics.submit_tx_hash = tx.tx_hash
        gas_total += tx.gas_used

        # ── Phase 2: Spectral voting ────────────────────────────────────────
        time.sleep(wait_for_peers_sec)  # wait for peers to submit

        # Fetch all peer gradients from local storage
        peer_gradients_dict = self._fetch_peer_gradients(round_num)

        if len(peer_gradients_dict) > 0:
            all_grads = np.stack(list(peer_gradients_dict.values()))
            fraud_candidates = []

            for peer_addr, peer_grad in peer_gradients_dict.items():
                if peer_addr == self.connector.account.address:
                    continue  # don't vote on own gradient

                accept, score, ks_stat = self.evaluate_peer_gradient(
                    peer_grad, all_grads, n_clients
                )

                tx = self.connector.submit_spectral_vote(round_num, peer_addr, accept, score)
                gas_total += tx.gas_used
                metrics.votes_cast += 1

                # Track fraud candidates (non-accepting votes)
                if not accept:
                    fraud_candidates.append((peer_addr, peer_grad, ks_stat))

            metrics.spectral_score = float(
                np.mean([self.evaluate_peer_gradient(g, all_grads, n_clients)[2]
                         for g in all_grads])
            )

            # ── Phase 3: Fraud proof submission (optimistic challenge) ──────
            for accused_addr, suspected_grad, ks_stat in fraud_candidates:
                proof = self.fraud_gen.generate(suspected_grad, n_clients)
                tx = self.connector.submit_fraud_proof(round_num, accused_addr, proof)
                if tx.success:
                    metrics.fraud_proofs_submitted += 1
                    gas_total += tx.gas_used

        metrics.gas_used_round = gas_total
        self.metrics_history.append(metrics)
        return metrics

    def _fetch_peer_gradients(self, round_num: int) -> Dict[str, np.ndarray]:
        """
        Fetch all peer gradients from local storage for a round.
        In local simulation, reads directly from shared filesystem.
        """
        result = {}
        submitters = self.connector.get_round_submitters(round_num)

        for addr in submitters:
            rec = self.connector.get_submission(round_num, addr)
            if not rec["submitted"] or rec["excluded"]:
                continue
            storage_key = bytes.fromhex(rec["gradient_hash"])
            grad = self.storage.retrieve(storage_key)
            if grad is not None:
                result[addr] = grad

        return result

    def update_model(self, aggregate_gradient: np.ndarray):
        """Apply aggregated gradient to local model."""
        params = list(self.model.parameters())
        idx = 0
        with torch.no_grad():
            for param in params:
                n = param.numel()
                delta = torch.tensor(
                    aggregate_gradient[idx:idx+n].reshape(param.shape),
                    dtype=param.dtype
                )
                param.data.add_(delta)
                idx += n

    def evaluate(self) -> float:
        """Evaluate model on test data."""
        device = next(self.model.parameters()).device
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += len(target)
        return correct / max(total, 1)

    def get_gradient_vector(self) -> np.ndarray:
        """Get current model parameters as a flattened vector."""
        parts = []
        for p in self.model.parameters():
            parts.append(p.data.cpu().numpy().flatten())
        return np.concatenate(parts)
