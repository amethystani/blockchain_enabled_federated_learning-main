"""
Blockchain connector for federated learning.

Handles Web3 connection, contract interaction, and transaction management.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from .config import BlockchainConfig
from .storage import ModelStorage


class BlockchainConnector:
    """Connector for blockchain operations in federated learning."""
    
    def __init__(self, config: BlockchainConfig):
        """
        Initialize blockchain connector.
        
        Args:
            config: BlockchainConfig instance
        """
        self.config = config
        self.config.validate()
        
        # Initialize Web3
        self.w3 = self._init_web3()
        
        # Initialize account
        self.account = self._init_account()
        
        # Load contract
        self.contract = self._load_contract()
        
        # Initialize storage
        self.storage = ModelStorage(
            storage_dir=config.storage_dir,
            use_ipfs=config.use_ipfs
        )
        
        print(f"✅ Connected to {config.network} (Chain ID: {self.w3.eth.chain_id})")
        print(f"👤 Account: {self.account.address}")
        print(f"💰 Balance: {self._get_balance()} {self._get_currency()}")
        if self.contract:
            print(f"📜 Contract: {self.config.contract_address}")
    
    def _init_web3(self) -> Web3:
        """Initialize Web3 connection."""
        w3 = Web3(Web3.HTTPProvider(self.config.rpc_url))
        
        # Add PoA middleware for networks like Polygon
        if self.config.network in ["mumbai", "polygon"]:
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        if not w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.config.rpc_url}")
        
        return w3
    
    def _init_account(self) -> Account:
        """Initialize account from private key."""
        if not self.config.private_key:
            raise ValueError("Private key not configured")
        
        # Remove 0x prefix if present
        private_key = self.config.private_key
        if private_key.startswith("0x"):
            private_key = private_key[2:]
        
        account = Account.from_key(private_key)
        self.config.account_address = account.address
        
        return account
    
    def _load_contract(self):
        """Load contract from ABI and address."""
        if not self.config.contract_address:
            print("⚠️  Contract address not set. Deploy contract first.")
            return None
        
        abi_path = Path(self.config.abi_path)
        if not abi_path.exists():
            print(f"⚠️  ABI file not found: {abi_path}")
            print("   Run: npx hardhat compile")
            return None
        
        with open(abi_path, 'r') as f:
            abi = json.load(f)
        
        contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.config.contract_address),
            abi=abi
        )
        
        return contract
    
    # ==================== Client Management ====================
    
    def register_client(self, client_id: int, client_address: Optional[str] = None) -> str:
        """
        Register a client on the blockchain.
        
        Args:
            client_id: Client identifier
            client_address: Client address (uses own address if None)
        
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        if client_address is None:
            client_address = self.account.address
        
        tx_hash = self._send_transaction(
            self.contract.functions.registerClient(
                Web3.to_checksum_address(client_address),
                client_id
            )
        )
        
        print(f"✅ Registered client {client_id} (tx: {tx_hash[:10]}...)")
        return tx_hash
    
    def register_clients_batch(self, client_ids: List[int], 
                              client_addresses: Optional[List[str]] = None) -> str:
        """
        Register multiple clients in a single transaction.
        
        Args:
            client_ids: List of client identifiers
            client_addresses: List of client addresses (uses own if None)
        
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        if client_addresses is None:
            client_addresses = [self.account.address] * len(client_ids)
        
        if len(client_ids) != len(client_addresses):
            raise ValueError("client_ids and client_addresses must have same length")
        
        # Convert to checksum addresses
        client_addresses = [Web3.to_checksum_address(addr) for addr in client_addresses]
        
        tx_hash = self._send_transaction(
            self.contract.functions.registerClientsBatch(
                client_addresses,
                client_ids
            )
        )
        
        print(f"✅ Registered {len(client_ids)} clients (tx: {tx_hash[:10]}...)")
        return tx_hash
    
    def is_client_registered(self, client_address: Optional[str] = None) -> bool:
        """Check if client is registered."""
        if not self.contract:
            return False
        
        if client_address is None:
            client_address = self.account.address
        
        return self.contract.functions.isRegisteredClient(
            Web3.to_checksum_address(client_address)
        ).call()
    
    # ==================== Round Management ====================
    
    def start_round(self) -> str:
        """
        Start a new federated learning round.
        
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        tx_hash = self._send_transaction(
            self.contract.functions.startRound()
        )
        
        round_num = self.get_current_round()
        print(f"🚀 Started round {round_num} (tx: {tx_hash[:10]}...)")
        
        return tx_hash
    
    def get_current_round(self) -> int:
        """Get current round number."""
        if not self.contract:
            return 0
        return self.contract.functions.currentRound().call()
    
    def submit_model_update(self, model_dict: Dict, client_id: int, 
                           round_num: Optional[int] = None) -> str:
        """
        Submit model update to blockchain.
        
        Args:
            model_dict: Model state dictionary
            client_id: Client identifier
            round_num: Round number (uses current if None)
        
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        if round_num is None:
            round_num = self.get_current_round()
        
        # Store model and get hash
        model_hash, storage_path = self.storage.store_model(model_dict, client_id, round_num)
        
        # Convert hash to bytes32
        model_hash_bytes = Web3.to_bytes(hexstr=model_hash)
        
        # Submit to blockchain
        tx_hash = self._send_transaction(
            self.contract.functions.submitModelUpdate(model_hash_bytes)
        )
        
        print(f"📤 Client {client_id} submitted update for round {round_num} (tx: {tx_hash[:10]}...)")
        
        return tx_hash
    
    def get_model_update(self, round_num: int, client_id: int) -> Optional[Dict]:
        """
        Retrieve model update from blockchain and storage.
        
        Args:
            round_num: Round number
            client_id: Client identifier
        
        Returns:
            Model state dictionary or None if not found
        """
        if not self.contract:
            return None
        
        # Get update info from blockchain
        update = self.contract.functions.getModelUpdate(round_num, client_id).call()
        
        if update[2] == b'\x00' * 32:  # modelHash is zero
            return None
        
        # Extract hash
        model_hash = update[2].hex()
        
        # Retrieve from storage
        try:
            model_dict = self.storage.retrieve_model(model_hash)
            return model_dict
        except FileNotFoundError:
            print(f"⚠️  Model file not found for hash {model_hash[:16]}...")
            return None
    
    def finalize_round(self, aggregated_model: Dict, round_num: Optional[int] = None) -> str:
        """
        Finalize round with aggregated model.
        
        Args:
            aggregated_model: Aggregated model state dictionary
            round_num: Round number (uses current if None)
        
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise ValueError("Contract not loaded")
        
        if round_num is None:
            round_num = self.get_current_round()
        
        # Store aggregated model
        model_hash, storage_path = self.storage.store_model(
            aggregated_model, 
            client_id=9999,  # Special ID for aggregated model
            round_num=round_num
        )
        
        # Convert hash to bytes32
        model_hash_bytes = Web3.to_bytes(hexstr=model_hash)
        
        # Finalize on blockchain
        tx_hash = self._send_transaction(
            self.contract.functions.finalizeRound(model_hash_bytes)
        )
        
        print(f"✅ Finalized round {round_num} (tx: {tx_hash[:10]}...)")
        
        return tx_hash
    
    def get_round_info(self, round_num: int) -> Dict[str, Any]:
        """Get information about a round."""
        if not self.contract:
            return {}
        
        info = self.contract.functions.getRoundInfo(round_num).call()
        
        return {
            "round_number": info[0],
            "start_time": info[1],
            "end_time": info[2],
            "aggregated_model_hash": info[3].hex() if info[3] != b'\x00' * 32 else None,
            "num_submissions": info[4],
            "finalized": info[5]
        }
    
    def get_round_submissions(self, round_num: int) -> int:
        """Get number of submissions for a round."""
        if not self.contract:
            return 0
        return self.contract.functions.getRoundSubmissions(round_num).call()
    
    def wait_for_submissions(self, round_num: int, expected_count: int, 
                            timeout: int = 300, poll_interval: int = 5) -> bool:
        """
        Wait for expected number of submissions.
        
        Args:
            round_num: Round number
            expected_count: Expected number of submissions
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
        
        Returns:
            True if expected submissions received, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current_count = self.get_round_submissions(round_num)
            
            if current_count >= expected_count:
                print(f"✅ Received {current_count}/{expected_count} submissions")
                return True
            
            print(f"⏳ Waiting for submissions: {current_count}/{expected_count}")
            time.sleep(poll_interval)
        
        print(f"⏱️  Timeout: Only {current_count}/{expected_count} submissions received")
        return False
    
    # ==================== Transaction Helpers ====================
    
    def _send_transaction(self, function, value: int = 0) -> str:
        """
        Send a transaction with retries.
        
        Args:
            function: Contract function to call
            value: ETH/MATIC value to send
        
        Returns:
            Transaction hash
        """
        for attempt in range(self.config.max_retries):
            try:
                # Build transaction
                tx = function.build_transaction({
                    'from': self.account.address,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                    'gas': self.config.gas_limit,
                    'gasPrice': self.config.gas_price or self.w3.eth.gas_price,
                    'value': value,
                    'chainId': self.config.chain_id
                })
                
                # Sign transaction
                signed_tx = self.w3.eth.account.sign_transaction(tx, self.config.private_key)
                
                # Send transaction
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for confirmation
                receipt = self.w3.eth.wait_for_transaction_receipt(
                    tx_hash, 
                    timeout=self.config.transaction_timeout
                )
                
                if receipt['status'] != 1:
                    raise Exception(f"Transaction failed: {tx_hash.hex()}")
                
                return tx_hash.hex()
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    print(f"⚠️  Transaction failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    raise Exception(f"Transaction failed after {self.config.max_retries} attempts: {e}")
    
    def _get_balance(self) -> str:
        """Get account balance."""
        balance_wei = self.w3.eth.get_balance(self.account.address)
        return f"{self.w3.from_wei(balance_wei, 'ether'):.4f}"
    
    def _get_currency(self) -> str:
        """Get network currency name."""
        if self.config.network in ["mumbai", "polygon"]:
            return "MATIC"
        return "ETH"
    
    # ==================== Utility Functions ====================
    
    def get_transaction_url(self, tx_hash: str) -> str:
        """Get block explorer URL for transaction."""
        explorers = {
            "mumbai": f"https://mumbai.polygonscan.com/tx/{tx_hash}",
            "polygon": f"https://polygonscan.com/tx/{tx_hash}",
            "sepolia": f"https://sepolia.etherscan.io/tx/{tx_hash}",
            "mainnet": f"https://etherscan.io/tx/{tx_hash}"
        }
        return explorers.get(self.config.network, f"Transaction: {tx_hash}")
    
    def get_gas_stats(self) -> Dict[str, Any]:
        """Get current gas statistics."""
        gas_price_wei = self.w3.eth.gas_price
        gas_price_gwei = self.w3.from_wei(gas_price_wei, 'gwei')
        
        return {
            "gas_price_wei": gas_price_wei,
            "gas_price_gwei": float(gas_price_gwei),
            "estimated_cost_per_tx": float(gas_price_gwei) * self.config.gas_limit / 1e9
        }
    
    def estimate_experiment_cost(self, num_clients: int, num_rounds: int) -> Dict[str, Any]:
        """Estimate total cost for an experiment."""
        gas_stats = self.get_gas_stats()
        
        # Transactions per experiment:
        # - num_rounds transactions to start rounds
        # - num_clients * num_rounds transactions for client updates
        # - num_rounds transactions to finalize rounds
        total_txs = num_rounds + (num_clients * num_rounds) + num_rounds
        
        cost_per_tx_eth = gas_stats["estimated_cost_per_tx"]
        total_cost_eth = cost_per_tx_eth * total_txs
        
        return {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "total_transactions": total_txs,
            "cost_per_transaction": f"{cost_per_tx_eth:.6f} {self._get_currency()}",
            "total_cost": f"{total_cost_eth:.4f} {self._get_currency()}",
            "current_gas_price_gwei": gas_stats["gas_price_gwei"]
        }
