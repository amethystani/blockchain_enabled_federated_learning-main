#!/usr/bin/env python3
"""
Check wallet balance on Amoy testnet.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web3 import Web3
from web3.middleware import geth_poa_middleware
import os
from dotenv import load_dotenv

def check_balance():
    """Check wallet balance on Amoy."""
    
    print("\n💰 Checking Amoy Testnet Balance...\n")
    
    # Load .env
    load_dotenv()
    
    wallet_address = os.getenv("WALLET_ADDRESS")
    rpc_url = os.getenv("AMOY_RPC_URL") or "https://rpc-amoy.polygon.technology/"
    
    # Connect to Amoy
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    
    if not w3.is_connected():
        print(f"❌ Failed to connect to {rpc_url}")
        return False
    
    print(f"✅ Connected to Amoy testnet")
    print(f"📡 Chain ID: {w3.eth.chain_id}")
    print(f"\n👤 Wallet: {wallet_address}")
    
    # Get balance
    balance_wei = w3.eth.get_balance(wallet_address)
    balance_matic = w3.from_wei(balance_wei, 'ether')
    
    print(f"💰 Balance: {balance_matic:.4f} POL/MATIC")
    
    # Check if sufficient
    min_balance = 0.01
    if float(balance_matic) < min_balance:
        print(f"\n⚠️  Balance too low (need at least {min_balance} POL)")
        print(f"\n🚰 Get more from faucet:")
        print(f"   https://faucet.polygon.technology/")
        print(f"   (Select Amoy network)")
        print(f"\n   Use address: {wallet_address}")
        return False
    else:
        print(f"\n✅ Sufficient balance for deployment!")
        return True


if __name__ == "__main__":
    success = check_balance()
    sys.exit(0 if success else 1)
