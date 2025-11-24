#!/usr/bin/env python3
"""
Setup script for testnet deployment.

Helps with wallet setup, testnet token acquisition, and contract deployment.
"""

import sys
import os
import json
from pathlib import Path
from web3 import Web3
from web3.middleware import geth_poa_middleware

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectral_sentinel.blockchain import BlockchainConfig


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║          🔗 BLOCKCHAIN TESTNET SETUP ASSISTANT               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def check_environment():
    """Check if .env file exists and is configured."""
    print("\n📋 Step 1: Checking environment configuration...\n")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  .env file not found. Creating from .env.example...")
            import shutil
            shutil.copy(env_example, env_file)
            print("✅ Created .env file")
            print("\n⚠️  IMPORTANT: Edit .env file and set your PRIVATE_KEY")
            print("   You can generate a new wallet at: https://metamask.io/\n")
            return False
        else:
            print("❌ Neither .env nor .env.example found!")
            return False
    
    # Load and check
    from dotenv import load_dotenv
    load_dotenv()
    
    private_key = os.getenv("PRIVATE_KEY")
    if not private_key or private_key == "your_private_key_here_without_0x_prefix":
        print("❌ PRIVATE_KEY not set in .env file")
        print("\n📝 To set up your wallet:")
        print("1. Install MetaMask: https://metamask.io/")
        print("2. Create a new account (or use existing)")
        print("3. Export private key from MetaMask")
        print("4. Add to .env file: PRIVATE_KEY=your_key_here")
        print("5. NEVER commit .env to git!\n")
        return False
    
    print("✅ .env file configured")
    return True


def check_wallet_balance(config: BlockchainConfig):
    """Check wallet balance on testnet."""
    print(f"\n💰 Step 2: Checking wallet balance on {config.network}...\n")
    
    try:
        # Initialize Web3
        w3 = Web3(Web3.HTTPProvider(config.rpc_url))
        
        # Add PoA middleware for Polygon
        if config.network in ["mumbai", "polygon"]:
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        if not w3.is_connected():
            print(f"❌ Failed to connect to {config.rpc_url}")
            return False
        
        # Get account from private key
        from eth_account import Account
        private_key = config.private_key
        if private_key.startswith("0x"):
            private_key = private_key[2:]
        account = Account.from_key(private_key)
        
        # Check balance
        balance_wei = w3.eth.get_balance(account.address)
        balance_eth = w3.from_wei(balance_wei, 'ether')
        
        currency = "MATIC" if config.network in ["mumbai", "polygon"] else "ETH"
        
        print(f"👤 Address: {account.address}")
        print(f"💰 Balance: {balance_eth:.4f} {currency}")
        
        # Check if sufficient
        min_balance = 0.1  # Minimum recommended
        if float(balance_eth) < min_balance:
            print(f"\n⚠️  Balance too low! Recommended minimum: {min_balance} {currency}")
            
            # Provide faucet links
            if config.network == "mumbai":
                print("\n🚰 Get free MATIC from faucets:")
                print("   1. https://faucet.polygon.technology/")
                print("   2. https://mumbaifaucet.com/")
                print("\n   Use your address:", account.address)
            elif config.network == "sepolia":
                print("\n🚰 Get free ETH from faucets:")
                print("   1. https://sepoliafaucet.com/")
                print("   2. https://faucet.sepolia.dev/")
                print("\n   Use your address:", account.address)
            
            return False
        
        print(f"✅ Sufficient balance for deployment")
        return True
        
    except Exception as e:
        print(f"❌ Error checking balance: {e}")
        return False


def load_deployment_info():
    """Load deployment info if exists."""
    deployment_file = Path("deployment.json")
    if deployment_file.exists():
        with open(deployment_file) as f:
            return json.load(f)
    return None


def update_env_with_contract(contract_address: str):
    """Update .env file with deployed contract address."""
    env_file = Path(".env")
    
    if not env_file.exists():
        return
    
    # Read current .env
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update or add CONTRACT_ADDRESS
    found = False
    for i, line in enumerate(lines):
        if line.startswith("CONTRACT_ADDRESS="):
            lines[i] = f"CONTRACT_ADDRESS={contract_address}\n"
            found = True
            break
    
    if not found:
        lines.append(f"\nCONTRACT_ADDRESS={contract_address}\n")
    
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Updated .env with CONTRACT_ADDRESS={contract_address}")


def main():
    print_banner()
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Setup incomplete. Please configure .env file first.\n")
        return False
    
    # Load config
    config = BlockchainConfig()
    
    print(f"📡 Network: {config.network}")
    print(f"🔗 RPC: {config.rpc_url}")
    
    # Step 2: Check balance
    if not check_wallet_balance(config):
        print("\n❌ Insufficient balance. Get testnet tokens first.\n")
        return False
    
    # Step 3: Check if already deployed
    print("\n📜 Step 3: Checking deployment status...\n")
    
    deployment = load_deployment_info()
    if deployment and deployment.get("contractAddress"):
        print(f"✅ Contract already deployed!")
        print(f"   Address: {deployment['contractAddress']}")
        print(f"   Network: {deployment['network']}")
        print(f"   Explorer: {deployment['explorerUrl']}")
        
        if deployment['network'] == config.network:
            update_env_with_contract(deployment['contractAddress'])
            print("\n✅ Ready to run tests!")
            print("\n🚀 Next step: python test_blockchain.py\n")
            return True
        else:
            print(f"\n⚠️  Deployed on {deployment['network']}, but config is {config.network}")
            print("   You may need to deploy to a different network.\n")
    
    # Step 4: Deploy contract
    print("\n🚀 Step 4: Deploying smart contract...\n")
    print("   This will:")
    print(f"   - Compile Solidity contract")
    print(f"   - Deploy to {config.network}")
    print(f"   - Save deployment info")
    print()
    
    response = input("   Continue with deployment? (yes/no): ")
    if response.lower() != 'yes':
        print("\n❌ Deployment cancelled.\n")
        return False
    
    # Check if Node.js/npm is installed
    import subprocess
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ npm not found. Please install Node.js first.")
            print("   Download from: https://nodejs.org/\n")
            return False
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js first.")
        print("   Download from: https://nodejs.org/\n")
        return False
    
    # Install dependencies
    print("📦 Installing Hardhat dependencies...")
    result = subprocess.run(['npm', 'install'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ npm install failed: {result.stderr}")
        return False
    print("✅ Dependencies installed")
    
    # Compile contract
    print("\n🔨 Compiling contract...")
    result = subprocess.run(['npx', 'hardhat', 'compile'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Compilation failed: {result.stderr}")
        return False
    print("✅ Contract compiled")
    
    # Deploy
    print(f"\n🚀 Deploying to {config.network}...")
    result = subprocess.run(
        ['npx', 'hardhat', 'run', 'scripts/deploy.js', '--network', config.network],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Deployment failed: {result.stderr}")
        return False
    
    # Load deployment info
    deployment = load_deployment_info()
    if deployment:
        update_env_with_contract(deployment['contractAddress'])
        
        print("\n" + "=" * 70)
        print("✅ DEPLOYMENT SUCCESSFUL!")
        print("=" * 70)
        print(f"\n📜 Contract: {deployment['contractAddress']}")
        print(f"🔗 Explorer: {deployment['explorerUrl']}")
        print(f"\n🚀 Next step: python test_blockchain.py\n")
        return True
    else:
        print("⚠️  Deployment completed but couldn't load deployment info")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
