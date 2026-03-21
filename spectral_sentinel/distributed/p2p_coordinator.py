"""
p2p_coordinator.py — Coordinates multiple FLNodes for local simulation.

Runs the full SpectralBFT protocol across N nodes:
  - Orchestrates rounds (start → submit → vote → tally → challenge → finalize)
  - Advances Hardhat blockchain blocks for challenge window simulation
  - Collects metrics from all nodes
  - Computes and applies aggregated gradients using on-chain weights
  - Reports accuracy, detection rate, gas costs, reputation evolution
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from spectral_sentinel.blockchain.advanced_connector import AdvancedConnector
from spectral_sentinel.distributed.fl_node import FLNode, GradientStorage, NodeConfig


@dataclass
class ExperimentResult:
    """Results from a full SpectralBFT local experiment."""
    n_rounds: int
    n_clients: int
    n_byzantine: int
    accuracies: List[float] = field(default_factory=list)
    detection_rates: List[float] = field(default_factory=list)
    fraud_proofs_per_round: List[int] = field(default_factory=list)
    gas_per_round: List[int] = field(default_factory=list)
    reputation_history: Dict[int, List[int]] = field(default_factory=dict)
    slash_pool_history: List[float] = field(default_factory=list)
    finality_times_blocks: List[int] = field(default_factory=list)
    total_gas_used: int = 0

    def summary(self) -> str:
        avg_acc = np.mean(self.accuracies[-5:]) if self.accuracies else 0
        avg_detect = np.mean(self.detection_rates) if self.detection_rates else 0
        total_fraud = sum(self.fraud_proofs_per_round)
        return (
            f"SpectralBFT Results:\n"
            f"  Rounds: {self.n_rounds}  Clients: {self.n_clients}  Byzantine: {self.n_byzantine}\n"
            f"  Final Accuracy: {avg_acc:.1%}\n"
            f"  Avg Detection Rate: {avg_detect:.1%}\n"
            f"  Total Fraud Proofs: {total_fraud}\n"
            f"  Total Gas Used: {self.total_gas_used:,}\n"
            f"  Avg Gas/Round: {np.mean(self.gas_per_round):.0f}"
        )


class P2PCoordinator:
    """
    Coordinator for multi-node SpectralBFT local simulation.

    This is NOT a central server — it's a simulation harness that:
    1. Starts rounds (owner/deployer account calls startRound())
    2. Waits for all nodes to complete their round
    3. Advances Hardhat blocks past the challenge window
    4. Triggers vote tallying and round finalization
    5. Distributes the agreed aggregate to all nodes

    In a production deployment, each node runs independently and the
    blockchain enforces the protocol — no coordinator is needed.

    Parameters
    ----------
    nodes : List[FLNode]
    owner_connector : AdvancedConnector
        Deployer/owner account connector (for startRound, registerClients)
    storage : GradientStorage
    """

    def __init__(
        self,
        nodes: List[FLNode],
        owner_connector: AdvancedConnector,
        storage: GradientStorage
    ):
        self.nodes = nodes
        self.owner = owner_connector
        self.storage = storage
        self.n_clients = len(nodes)
        self.n_byzantine = sum(1 for n in nodes if n.config.is_byzantine)
        self.honest_indices = [i for i, n in enumerate(nodes) if not n.config.is_byzantine]
        self.byzantine_indices = [i for i, n in enumerate(nodes) if n.config.is_byzantine]

    def setup(self, stake_eth: float = 0.05):
        """
        Register all clients and stake ETH.
        Minimal stake satisfies Nash equilibrium condition (paper Theorem 3).
        """
        print(f"\nRegistering {self.n_clients} clients...")

        addresses = [n.connector.account.address for n in self.nodes]
        ids = list(range(self.n_clients))

        # Batch registration (gas efficient)
        tx = self.owner.register_clients_batch(addresses, ids)
        print(f"  Batch registration: {tx.tx_hash[:20]}... gas={tx.gas_used:,}")

        # Stake ETH for each client
        print(f"  Staking {stake_eth} ETH per client...")
        for i, node in enumerate(self.nodes):
            tx = node.connector.stake_eth(stake_eth)
            if not tx.success:
                print(f"  WARNING: Client {i} stake failed: {tx.error}")

        print(f"  Setup complete. {self.n_clients} clients registered and staked.")

    def run_experiment(
        self,
        n_rounds: int = 20,
        test_loader=None,
        verbose: bool = True
    ) -> ExperimentResult:
        """
        Run full SpectralBFT experiment for n_rounds.

        Returns ExperimentResult with accuracy, detection, gas, reputation metrics.
        """
        result = ExperimentResult(
            n_rounds=n_rounds,
            n_clients=self.n_clients,
            n_byzantine=self.n_byzantine
        )
        # Initialize reputation tracking
        for i in range(self.n_clients):
            result.reputation_history[i] = []

        for round_num in range(1, n_rounds + 1):
            if verbose:
                print(f"\n{'─'*50}")
                print(f"  Round {round_num}/{n_rounds}")

            round_metrics = self._run_round(round_num, verbose)

            # Collect metrics
            round_gas = sum(m.gas_used_round for m in round_metrics)
            round_fraud = sum(m.fraud_proofs_submitted for m in round_metrics)
            result.gas_per_round.append(round_gas)
            result.fraud_proofs_per_round.append(round_fraud)
            result.total_gas_used += round_gas

            # Detection rate: fraction of Byzantine nodes that were excluded
            excluded_count = self._count_excluded(round_num)
            detection_rate = excluded_count / max(self.n_byzantine, 1)
            result.detection_rates.append(detection_rate)

            # Slash pool
            round_result = self.owner.get_round_result(round_num)
            result.slash_pool_history.append(round_result["slash_pool_eth"])

            # Reputation history
            for i, node in enumerate(self.nodes):
                rep = node.connector.get_reputation(node.connector.account.address)
                result.reputation_history[i].append(rep)

            # Accuracy (from honest nodes)
            if test_loader is not None:
                acc = self._evaluate_honest_nodes()
                result.accuracies.append(acc)
            else:
                # Use local evaluation from nodes
                accs = [n.evaluate() for n in self.nodes[:3]]  # sample 3 nodes
                result.accuracies.append(float(np.mean(accs)))

            if verbose:
                print(f"    Gas used: {round_gas:,}  "
                      f"Fraud proofs: {round_fraud}  "
                      f"Detection: {detection_rate:.0%}  "
                      f"Accuracy: {result.accuracies[-1]:.1%}")

        if verbose:
            print(f"\n{'═'*50}")
            print(result.summary())
            self._print_gas_breakdown()

        return result

    def _run_round(self, round_num: int, verbose: bool) -> list:
        """Execute one full round of the SpectralBFT protocol."""

        # ── Owner starts round ──────────────────────────────────────────────
        tx = self.owner.start_round()
        start_block = tx.block_number

        # Collect honest gradients (for ALIE attack simulation)
        honest_gradients = []
        if any(n.config.attack_type == "alie" for n in self.nodes if n.config.is_byzantine):
            for i in self.honest_indices[:3]:  # sample from honest nodes
                g, _ = self.nodes[i].compute_gradient()
                honest_gradients.append(g)

        # ── All nodes submit + vote (Phase 1 + 2) ───────────────────────────
        # In local simulation, run nodes sequentially
        # In production, nodes run independently and in parallel
        all_metrics = []
        for i, node in enumerate(self.nodes):
            m = node.run_round(
                round_num=round_num,
                honest_gradients=honest_gradients if node.config.is_byzantine else None,
                wait_for_peers_sec=0  # no wait needed in sequential simulation
            )
            all_metrics.append(m)

        # ── Tally votes ─────────────────────────────────────────────────────
        # Any node can call tallyVotes after vote window
        # Advance blocks past vote window first
        vote_window = self.owner.contract.functions.voteWindowBlocks().call()
        self.owner.mine_blocks(vote_window + 1)

        try:
            tx = self.owner.tally_votes(round_num)
            if verbose:
                print(f"    Vote tally: gas={tx.gas_used:,}")
        except Exception as e:
            if verbose:
                print(f"    Tally skipped: {e}")

        # ── Advance past challenge window (optimistic finality) ─────────────
        challenge_window = self.owner.contract.functions.challengeWindowBlocks().call()
        self.owner.mine_blocks(challenge_window + 1)

        finality_block = self.owner.w3.eth.block_number
        result.finality_times_blocks.append(finality_block - start_block) if hasattr(self, '_result_ref') else None

        # ── Finalize round ──────────────────────────────────────────────────
        # Compute aggregated model hash (from accepted gradients)
        weights = self.owner.get_aggregation_weights(round_num)

        agg_gradient = self._compute_aggregate(round_num, weights)
        if agg_gradient is not None:
            agg_bytes = agg_gradient.astype(np.float32).tobytes()
            import hashlib
            agg_hash = hashlib.sha256(agg_bytes).digest()

            # Apply aggregate to all honest nodes
            for node in self.nodes:
                if not node.config.is_byzantine:
                    node.update_model(agg_gradient)
        else:
            agg_hash = b'\x00' * 32

        tx = self.owner.finalize_round(round_num, agg_hash)
        if verbose:
            print(f"    Round finalised: gas={tx.gas_used:,}")

        return all_metrics

    def _compute_aggregate(
        self,
        round_num: int,
        weights: Dict[str, float]
    ) -> Optional[np.ndarray]:
        """
        Compute weighted aggregate of accepted gradients using on-chain weights.
        Any node can perform this computation independently given on-chain data.
        This is the key property of Spectral Graded Broadcast: fully verifiable aggregation.
        """
        if not weights:
            return None

        weighted_sum = None
        total_weight = 0.0

        for addr, weight in weights.items():
            rec = self.owner.get_submission(round_num, addr)
            if not rec["submitted"] or rec["excluded"]:
                continue

            storage_key = bytes.fromhex(rec["gradient_hash"])
            grad = self.storage.retrieve(storage_key)
            if grad is None:
                continue

            if weighted_sum is None:
                weighted_sum = np.zeros_like(grad.astype(np.float64))
            weighted_sum += weight * grad.astype(np.float64)
            total_weight += weight

        if weighted_sum is None or total_weight == 0:
            return None

        return (weighted_sum / total_weight).astype(np.float32)

    def _count_excluded(self, round_num: int) -> int:
        """Count how many Byzantine clients were excluded in a round."""
        excluded = 0
        for i in self.byzantine_indices:
            addr = self.nodes[i].connector.account.address
            rec = self.owner.get_submission(round_num, addr)
            if rec.get("excluded"):
                excluded += 1
        return excluded

    def _evaluate_honest_nodes(self) -> float:
        """Average accuracy across all honest nodes."""
        accs = [n.evaluate() for n in self.nodes if not n.config.is_byzantine]
        return float(np.mean(accs)) if accs else 0.0

    def _print_gas_breakdown(self):
        """Print detailed gas cost breakdown."""
        print("\nGas Cost Breakdown:")
        print("  Operation               | Approx Gas | Description")
        print("  ─────────────────────────────────────────────────────────")
        print("  submitGradient          |     45,000 | Gradient hash + Frobenius norm")
        print("  submitSpectralVote      |     30,000 | Per-vote (n×(n-1) total)")
        print("  tallyVotes              |     60,000 | Compute weights (one-time)")
        print("  submitFraudProof (k=16) |     55,000 | O(k) verification")
        print("  finalizeRound           |     35,000 | Publish aggregate hash")
        print("  claimReward             |     25,000 | Per honest client")
        print(f"\n  Total gas used this experiment: {sum(self.nodes[0].connector.get_gas_stats()['total_gas'] for _ in self.nodes):,}")
