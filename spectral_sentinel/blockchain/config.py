"""
Blockchain configuration for federated learning.
"""

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path


@dataclass
class BlockchainConfig:
    """Configuration for blockchain connection and operations."""
    
    # Network configuration
    network: str = "mumbai"  # mumbai, polygon, sepolia, mainnet, local
    rpc_url: Optional[str] = None
    chain_id: Optional[int] = None
    
    # Account configuration
    private_key: Optional[str] = None
    account_address: Optional[str] = None
    
    # Contract configuration
    contract_address: Optional[str] = None
    abi_path: Optional[str] = None
    
    # Storage configuration  
    use_ipfs: bool = False
    ipfs_api: str = "/ip4/127.0.0.1/tcp/5001"
    storage_dir: str = "./blockchain_storage"
    
    # Transaction configuration
    gas_limit: int = 500000
    gas_price: Optional[int] = None  # Auto if None
    confirmation_blocks: int = 1
    transaction_timeout: int = 300  # seconds
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    
    def __post_init__(self):
        """Load configuration from environment variables if not provided."""
        
        # Load from .env file if exists
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            self._load_from_env()
        
        # Set default RPC URLs based on network
        if self.rpc_url is None:
            self.rpc_url = self._get_default_rpc()
        
        # Set default chain ID based on network
        if self.chain_id is None:
            self.chain_id = self._get_chain_id()
        
        # Set default ABI path
        if self.abi_path is None:
            self.abi_path = str(Path(__file__).parent.parent.parent / "contracts" / "FederatedLearning.abi.json")
        
        # Create storage directory
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        from dotenv import load_dotenv
        load_dotenv()
        
        # Override with env vars if present
        if not self.network and os.getenv("BLOCKCHAIN_NETWORK"):
            self.network = os.getenv("BLOCKCHAIN_NETWORK")
        
        if not self.rpc_url:
            env_key = f"{self.network.upper()}_RPC_URL"
            self.rpc_url = os.getenv(env_key) or os.getenv("RPC_URL")
        
        if not self.private_key:
            self.private_key = os.getenv("PRIVATE_KEY")
        
        if not self.contract_address:
            self.contract_address = os.getenv("CONTRACT_ADDRESS")
        
        if os.getenv("USE_IPFS"):
            self.use_ipfs = os.getenv("USE_IPFS").lower() == "true"
        
        if os.getenv("IPFS_API"):
            self.ipfs_api = os.getenv("IPFS_API")
        
        if os.getenv("GAS_LIMIT"):
            self.gas_limit = int(os.getenv("GAS_LIMIT"))
        
        if os.getenv("CONFIRMATION_BLOCKS"):
            self.confirmation_blocks = int(os.getenv("CONFIRMATION_BLOCKS"))
    
    def _get_default_rpc(self) -> str:
        """Get default RPC URL for network."""
        rpc_urls = {
            "mumbai": "https://rpc-mumbai.maticvigil.com",
            "polygon": "https://polygon-rpc.com",
            "sepolia": "https://rpc.sepolia.org",
            "mainnet": "https://eth.llamarpc.com",
            "local": "http://127.0.0.1:8545"
        }
        return rpc_urls.get(self.network, "http://127.0.0.1:8545")
    
    def _get_chain_id(self) -> int:
        """Get chain ID for network."""
        chain_ids = {
            "mumbai": 80001,
            "polygon": 137,
            "sepolia": 11155111,
            "mainnet": 1,
            "local": 31337
        }
        return chain_ids.get(self.network, 31337)
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.rpc_url:
            raise ValueError("RPC URL not configured")
        
        if not self.private_key:
            raise ValueError("Private key not configured. Set PRIVATE_KEY in .env file")
        
        if not self.contract_address and self.network != "local":
            raise ValueError("Contract address not configured. Deploy contract first or set CONTRACT_ADDRESS in .env")
        
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding private key)."""
        return {
            "network": self.network,
            "rpc_url": self.rpc_url,
            "chain_id": self.chain_id,
            "contract_address": self.contract_address,
            "use_ipfs": self.use_ipfs,
            "gas_limit": self.gas_limit,
            "confirmation_blocks": self.confirmation_blocks
        }
    
    def __repr__(self) -> str:
        """String representation (hide private key)."""
        return f"BlockchainConfig(network={self.network}, chain_id={self.chain_id}, contract={self.contract_address[:10] if self.contract_address else 'None'}...)"
