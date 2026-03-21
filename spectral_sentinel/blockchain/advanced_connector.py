"""
advanced_connector.py — Python interface for SpectralBFT.sol.

Wraps the SpectralBFT smart contract with typed Python methods for:
  - Optimistic gradient submission (Contribution 1)
  - Spectral graded broadcast voting (Contribution 2)
  - Fraud proof submission and verification
  - Staking, slashing, reward claims
  - Reputation queries

Designed for local Hardhat stack (no testnet required) but also works
on any EVM-compatible network.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from spectral_sentinel.rmt.fraud_proof_generator import FraudProof

REPO_ROOT = Path(__file__).parent.parent.parent
DEPLOYMENT_FILE = REPO_ROOT / "local_deployment.json"
ABI_FILE = REPO_ROOT / "contracts" / "SpectralBFT.abi.json"

# Hardhat pre-funded private keys (for local experiments)
HARDHAT_PRIVATE_KEYS = [
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
    "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
    "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba",
    "0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564",
    "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356",
    "0xdbda1821b80551c9d65939329250132c444b91a8b1c09d3cf046965ad0eca3e9",
    "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6d409c6",
]


@dataclass
class TxResult:
    """Result of an on-chain transaction."""
    tx_hash: str
    block_number: int
    gas_used: int
    success: bool
    error: str = ""


class AdvancedConnector:
    """
    Python interface for SpectralBFT.sol smart contract.

    Parameters
    ----------
    rpc_url : str
        Web3 RPC endpoint (default: http://127.0.0.1:8545 for local Hardhat)
    private_key : str
        Account private key for signing transactions
    contract_address : str
        Deployed SpectralBFT contract address
    abi_path : Path, optional
        Path to SpectralBFT.abi.json (auto-detected from local_deployment.json)
    gas_limit : int
        Gas limit per transaction
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:8545",
        private_key: str = HARDHAT_PRIVATE_KEYS[0],
        contract_address: Optional[str] = None,
        abi_path: Optional[Path] = None,
        gas_limit: int = 1_000_000
    ):
        self.rpc_url = rpc_url
        self.gas_limit = gas_limit
        self.gas_stats = {"total_gas": 0, "tx_count": 0}

        # Load from local_deployment.json if available
        deployment = self._load_deployment()
        if contract_address is None:
            contract_address = deployment.get("contractAddress")
        if contract_address is None:
            raise ValueError("No contract address — run scripts/start_local_stack.py first")

        # Web3 setup
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        # PoA middleware for Polygon networks
        if not (rpc_url.startswith("http://127") or "localhost" in rpc_url):
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to {rpc_url}")

        self.account = self.w3.eth.account.from_key(private_key)
        self.private_key = private_key

        # Load ABI
        if abi_path is None:
            abi_path = ABI_FILE
        with open(abi_path) as f:
            abi = json.load(f)

        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )
        self.contract_address = contract_address

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _load_deployment(self) -> dict:
        if DEPLOYMENT_FILE.exists():
            with open(DEPLOYMENT_FILE) as f:
                return json.load(f)
        return {}

    def _send_tx(self, fn, value_wei: int = 0, max_retries: int = 3) -> TxResult:
        """Build, sign, and send a contract transaction with retry logic."""
        for attempt in range(max_retries):
            try:
                nonce = self.w3.eth.get_transaction_count(self.account.address)
                gas_price = self.w3.eth.gas_price

                tx = fn.build_transaction({
                    "from": self.account.address,
                    "nonce": nonce,
                    "gas": self.gas_limit,
                    "gasPrice": gas_price,
                    "value": value_wei,
                    "chainId": self.w3.eth.chain_id
                })

                signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

                gas_used = receipt.gasUsed
                self.gas_stats["total_gas"] += gas_used
                self.gas_stats["tx_count"] += 1

                return TxResult(
                    tx_hash=tx_hash.hex(),
                    block_number=receipt.blockNumber,
                    gas_used=gas_used,
                    success=receipt.status == 1
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    return TxResult(tx_hash="", block_number=0, gas_used=0,
                                   success=False, error=str(e))

    @staticmethod
    def _to_bytes32(data: bytes) -> bytes:
        """Pad bytes to 32 bytes for Solidity bytes32."""
        return data[:32].ljust(32, b'\x00')

    # ─── Client Registration ─────────────────────────────────────────────────

    def register_client(self, client_address: str, client_id: int) -> TxResult:
        fn = self.contract.functions.registerClient(
            Web3.to_checksum_address(client_address), client_id
        )
        return self._send_tx(fn)

    def register_clients_batch(self, addresses: List[str], ids: List[int]) -> TxResult:
        fn = self.contract.functions.registerClientsBatch(
            [Web3.to_checksum_address(a) for a in addresses], ids
        )
        return self._send_tx(fn)

    # ─── Round Management ────────────────────────────────────────────────────

    def start_round(self) -> TxResult:
        fn = self.contract.functions.startRound()
        return self._send_tx(fn)

    def get_current_round(self) -> int:
        return self.contract.functions.currentRound().call()

    # ─── Contribution 1: Optimistic Gradient Submission ──────────────────────

    def submit_gradient(
        self,
        round_num: int,
        grad_hash: bytes,
        storage_key: bytes,
        frob_norm_e6: int
    ) -> TxResult:
        """
        Submit gradient hash + Frobenius norm for optimistic acceptance.
        Round proceeds without pre-verification.

        Parameters
        ----------
        round_num    : Current FL round
        grad_hash    : keccak256(gradient_bytes) — 32 bytes
        storage_key  : Content-addressed storage key — 32 bytes
        frob_norm_e6 : ||gradient||²/n × 10⁶ — used in fraud proof Frobenius check
        """
        fn = self.contract.functions.submitGradient(
            round_num,
            self._to_bytes32(grad_hash),
            self._to_bytes32(storage_key),
            frob_norm_e6
        )
        return self._send_tx(fn)

    # ─── Contribution 2: Spectral Graded Broadcast Voting ────────────────────

    def submit_spectral_vote(
        self,
        round_num: int,
        target_address: str,
        accept: bool,
        score: int  # 0–1000
    ) -> TxResult:
        """
        Submit spectral vote for a peer's gradient quality.
        score = round((1 - KS_stat) × 1000) ∈ [0, 1000].

        Implements Spectral Graded Broadcast: the first instantiation of
        graded broadcast with a continuous spectral validity predicate V(g) → [0,1].
        """
        fn = self.contract.functions.submitSpectralVote(
            round_num,
            Web3.to_checksum_address(target_address),
            accept,
            score
        )
        return self._send_tx(fn)

    def tally_votes(self, round_num: int) -> TxResult:
        """Tally votes and compute on-chain aggregation weights."""
        fn = self.contract.functions.tallyVotes(round_num)
        return self._send_tx(fn)

    # ─── Contribution 1: Fraud Proof Submission ───────────────────────────────

    def submit_fraud_proof(
        self,
        round_num: int,
        accused_address: str,
        proof: FraudProof
    ) -> TxResult:
        """
        Submit on-chain spectral fraud proof against an accused client.

        Theorem 1 (On-Chain Gas): O(k) gas where k = len(proof.eigenvalues_e6).
        Total gas ≈ 21000 + 200k.

        Parameters
        ----------
        round_num        : Round in which accused submitted
        accused_address  : Accused client's Ethereum address
        proof            : FraudProof from fraud_proof_generator.py
        """
        fraud_proof_tuple = (
            self._to_bytes32(proof.gradient_hash),
            proof.eigenvalues_e6,
            proof.ks_stat_e6,
            proof.sigma2_e6,
            proof.lambda_plus_e6,
        )
        fn = self.contract.functions.submitFraudProof(
            round_num,
            Web3.to_checksum_address(accused_address),
            fraud_proof_tuple
        )
        return self._send_tx(fn)

    # ─── Finalization ────────────────────────────────────────────────────────

    def finalize_round(self, round_num: int, agg_model_hash: bytes) -> TxResult:
        """Finalise round after challenge window. Publishes aggregated model hash on-chain."""
        fn = self.contract.functions.finalizeRound(
            round_num, self._to_bytes32(agg_model_hash)
        )
        return self._send_tx(fn)

    def claim_reward(self, round_num: int) -> TxResult:
        """Claim honest client's share of slash pool for a finalised round."""
        fn = self.contract.functions.claimReward(round_num)
        return self._send_tx(fn)

    # ─── Staking ─────────────────────────────────────────────────────────────

    def stake_eth(self, amount_eth: float) -> TxResult:
        """Stake ETH to participate as FL client."""
        amount_wei = self.w3.to_wei(amount_eth, "ether")
        fn = self.contract.functions.stakeETH()
        return self._send_tx(fn, value_wei=amount_wei)

    def unstake(self, amount_eth: float) -> TxResult:
        """Unstake ETH."""
        amount_wei = self.w3.to_wei(amount_eth, "ether")
        fn = self.contract.functions.unstake(amount_wei)
        return self._send_tx(fn)

    def get_stake(self, address: str) -> float:
        """Get staked ETH for an address."""
        wei = self.contract.functions.stake(Web3.to_checksum_address(address)).call()
        return float(self.w3.from_wei(wei, "ether"))

    # ─── Views ───────────────────────────────────────────────────────────────

    def get_aggregation_weights(self, round_num: int) -> Dict[str, float]:
        """
        Get on-chain agreed aggregation weights (Contribution 2 output).
        Returns dict of {client_address: weight ∈ [0,1]}.
        Weights are derived from spectral scores: w_i = Σ_j score_j(i) / Σ_i Σ_j score_j(i).
        """
        addresses, weights_e6 = self.contract.functions.getAggregationWeights(round_num).call()
        total = sum(weights_e6) or 1
        return {addr: w / total for addr, w in zip(addresses, weights_e6)}

    def get_reputation(self, address: str) -> int:
        """Get client reputation score [0, 1000]."""
        return self.contract.functions.reputation(Web3.to_checksum_address(address)).call()

    def get_submission(self, round_num: int, address: str) -> dict:
        """Get gradient submission record for a client in a round."""
        grad_hash, submitted, excluded, frob_norm_e6 = \
            self.contract.functions.getSubmission(
                round_num, Web3.to_checksum_address(address)
            ).call()
        return {
            "gradient_hash": grad_hash.hex(),
            "submitted": submitted,
            "excluded": excluded,
            "frob_norm_e6": frob_norm_e6
        }

    def get_round_result(self, round_num: int) -> dict:
        """Get round finalization result."""
        finalised, agg_hash, accepted, slash_pool = \
            self.contract.functions.getRoundResult(round_num).call()
        return {
            "finalised": finalised,
            "aggregated_model_hash": agg_hash.hex(),
            "accepted_count": accepted,
            "slash_pool_eth": float(self.w3.from_wei(slash_pool, "ether"))
        }

    def get_round_submitters(self, round_num: int) -> List[str]:
        """Get list of client addresses that submitted in a round."""
        return self.contract.functions.getRoundSubmitters(round_num).call()

    def get_all_clients(self) -> List[str]:
        """Get all registered client addresses."""
        return self.contract.functions.getClients().call()

    def get_balance(self, address: Optional[str] = None) -> float:
        """Get ETH balance."""
        addr = address or self.account.address
        wei = self.w3.eth.get_balance(Web3.to_checksum_address(addr))
        return float(self.w3.from_wei(wei, "ether"))

    def get_gas_stats(self) -> dict:
        """Get cumulative gas usage statistics."""
        return {
            "total_gas": self.gas_stats["total_gas"],
            "tx_count": self.gas_stats["tx_count"],
            "avg_gas_per_tx": (self.gas_stats["total_gas"] / max(self.gas_stats["tx_count"], 1))
        }

    # ─── Utility ─────────────────────────────────────────────────────────────

    @staticmethod
    def gradient_to_hash(gradient_bytes: bytes) -> bytes:
        """SHA-256 hash of gradient bytes for submission."""
        return hashlib.sha256(gradient_bytes).digest()

    @staticmethod
    def gradient_to_storage_key(gradient_bytes: bytes) -> bytes:
        """Content-addressed storage key (SHA-256 of gradient)."""
        return hashlib.sha256(gradient_bytes).digest()

    def mine_blocks(self, n: int = 1):
        """
        Mine n blocks on local Hardhat (advances block number for challenge window).
        Only works on local Hardhat node.
        """
        for _ in range(n):
            self.w3.provider.make_request("evm_mine", [])

    def advance_past_challenge_window(self):
        """Mine enough blocks to pass the challenge window (for testing)."""
        challenge_window = self.contract.functions.challengeWindowBlocks().call()
        vote_window = self.contract.functions.voteWindowBlocks().call()
        total = challenge_window + vote_window + 2
        self.mine_blocks(total)


def create_local_connector(account_index: int = 0) -> AdvancedConnector:
    """
    Create an AdvancedConnector using local Hardhat node and pre-funded accounts.
    Reads contract address from local_deployment.json.
    """
    return AdvancedConnector(
        rpc_url="http://127.0.0.1:8545",
        private_key=HARDHAT_PRIVATE_KEYS[account_index],
        abi_path=ABI_FILE
    )


def create_multi_connectors(n_clients: int = 10) -> List[AdvancedConnector]:
    """
    Create one connector per FL client using Hardhat accounts 0..n_clients-1.
    Account 0 = deployer/owner. Accounts 1..n_clients = FL clients.
    """
    return [create_local_connector(i) for i in range(n_clients)]
