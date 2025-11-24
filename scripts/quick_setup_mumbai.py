#!/usr/bin/env python3
"""
Quick Mumbai Testnet Setup

Creates .env file with your wallet configuration.
"""

import os
from pathlib import Path

def create_env_file():
    """Create .env file for Mumbai testnet."""
    
    print("""
╔════════════════════════════════════════════════════════╗
║                                                        ║
║        🔗 MUMBAI TESTNET QUICK SETUP                  ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    """)
    
    wallet_address = "0x0f8bbF3E6c8AA11C79Af8590B49144f09ed5d7d2"
    
    print(f"\n✅ Wallet Address: {wallet_address}")
    print(f"✅ Testnet MATIC claimed (arriving in 2-3 minutes)")
    print(f"\n🔑 Now we need your private key from MetaMask:\n")
    
    print("📝 HOW TO EXPORT PRIVATE KEY FROM METAMASK:")
    print("   1. Open MetaMask browser extension")
    print("   2. Click the 3 dots (⋮) next to your account name")
    print("   3. Select 'Account Details'")
    print("   4. Click 'Show Private Key'")
    print("   5. Enter your MetaMask password")
    print("   6. Click to reveal and copy the private key")
    print("   7. Paste it below\n")
    
    print("⚠️  SECURITY WARNING:")
    print("   - This key will be saved to .env (which is in .gitignore)")
    print("   - NEVER share this key with anyone")
    print("   - NEVER commit .env to git")
    print("   - This is TESTNET only - don't use mainnet keys!\n")
    
    private_key = input("Enter your private key (without 0x prefix): ").strip()
    
    # Remove 0x if present
    if private_key.startswith("0x"):
        private_key = private_key[2:]
    
    # Validate length
    if len(private_key) != 64:
        print(f"\n❌ Invalid private key length: {len(private_key)} (should be 64)")
        print("   Make sure to copy the entire key without 0x prefix")
        return False
    
    # Create .env content
    env_content = f"""# ====================================
# Blockchain Configuration - Mumbai Testnet
# ====================================

# Network Selection
BLOCKCHAIN_NETWORK=mumbai

# RPC URLs
MUMBAI_RPC_URL=https://rpc-mumbai.maticvigil.com

# Your Wallet
WALLET_ADDRESS={wallet_address}
PRIVATE_KEY={private_key}

# Deployed Contract Address (will be set after deployment)
CONTRACT_ADDRESS=

# ====================================
# Gas Configuration
# ====================================
GAS_LIMIT=500000
CONFIRMATION_BLOCKS=1

# ====================================
# Storage
# ====================================
USE_IPFS=false
"""
    
    # Write .env file
    env_path = Path(".env")
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    # Set permissions (Unix-like systems)
    try:
        os.chmod(env_path, 0o600)  # Read/write for owner only
    except:
        pass
    
    print("\n" + "="*60)
    print("✅ .env file created successfully!")
    print("="*60)
    print(f"\n📁 Location: {env_path.absolute()}")
    print(f"🔒 Permissions: Owner read/write only")
    print(f"\n🚀 NEXT STEPS:\n")
    print("   1. Wait 2-3 minutes for MATIC to arrive")
    print("   2. Check balance: python scripts/check_balance.py")
    print("   3. Install dependencies: npm install")
    print("   4. Deploy contract: python scripts/setup_testnet.py")
    print("   5. Run test: python test_blockchain.py\n")
    
    return True


if __name__ == "__main__":
    import sys
    success = create_env_file()
    sys.exit(0 if success else 1)
