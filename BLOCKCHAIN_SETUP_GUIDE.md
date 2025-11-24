# 🔗 Blockchain Integration Setup Guide

This guide will help you integrate real blockchain networks (Polygon/Ethereum) with the federated learning system.

## 📋 Prerequisites

### 1. Software Requirements

- **Python 3.12+** (already installed)
- **Node.js 18+** and npm (for Hardhat)
  ```bash
  # Check versions
  node --version  # Should be v18 or higher
  npm --version
  
  # Install from: https://nodejs.org/
  ```

### 2. Blockchain Account Setup

You need a wallet with a private key:

**Option A: Create new wallet (recommended for testing)**
1. Install [MetaMask](https://metamask.io/) browser extension
2. Create new account
3. Export private key (Settings → Security → Reveal Private Key)
4. **NEVER share this key or commit to git!**

**Option B: Use existing wallet**
- Export private key from your wallet
- **Use a TEST wallet for testnets, not your main wallet!**

### 3. Get Testnet Tokens (FREE)

**For Polygon Mumbai:**
- Visit [Polygon Faucet](https://faucet.polygon.technology/)
- Enter your wallet address
- Receive free test MATIC

**For Ethereum Sepolia:**
- Visit [Sepolia Faucet](https://sepoliafaucet.com/)
- Or [Sepolia Dev Faucet](https://faucet.sepolia.dev/)
- Receive free test ETH

---

## 🚀 Quick Start (Local Testing)

### Step 1: Install Dependencies

```bash
cd /Users/animesh/Downloads/blockchain_enabled_federated_learning-main

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

### Step 2: Start Local Blockchain

```bash
# Terminal 1: Start Hardhat local node
npx hardhat node

# This will create a local blockchain with 20 pre-funded accounts
# Keep this terminal running!
```

### Step 3: Deploy Contract Locally

```bash
# Terminal 2: Deploy contract
npx hardhat run scripts/deploy.js --network localhost

# You'll see output like:
# ✅ Contract deployed to: 0x5FbDB2315678afecb367f032d93F642f64180aa3
```

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file
nano .env  # or use your favorite editor

# For local testing, set:
BLOCKCHAIN_NETWORK=local
PRIVATE_KEY=ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3  # From deployment output
```

### Step 5: Run Integration Test

```bash
python test_blockchain.py
```

Expected output:
```
🧪 BLOCKCHAIN INTEGRATION TEST
================================
📡 Step 1: Initializing blockchain connector...
✅ Connected to local (Chain ID: 31337)
👥 Step 2: Registering clients...
✅ All 5 clients registered successfully
...
✅ ALL TESTS PASSED!
```

---

## 🌐 Testnet Deployment (Polygon Mumbai - FREE)

### Step 1: Get Testnet Tokens

1. Visit [https://faucet.polygon.technology/](https://faucet.polygon.technology/)
2. Select "Mumbai" network
3. Enter your wallet address
4. Click "Submit"
5. Wait 1-2 minutes for tokens

### Step 2: Configure for Mumbai

```bash
# Edit .env file
nano .env

# Set these values:
BLOCKCHAIN_NETWORK=mumbai
PRIVATE_KEY=your_private_key_here_without_0x
```

### Step 3: Deploy to Mumbai

```bash
# Run setup assistant (recommended)
python scripts/setup_testnet.py

# OR manually:
npx hardhat run scripts/deploy.js --network mumbai
```

### Step 4: Verify Deployment

Visit [Mumbai PolygonScan](https://mumbai.polygonscan.com/) and search for your contract address to see it on the blockchain!

### Step 5: Run Tests on Mumbai

```bash
# Update CONTRACT_ADDRESS in .env first, then:
python test_blockchain.py
```

---

## 💰 Cost Comparison

| Network | Cost per Transaction | 50 Rounds, 20 Clients | Speed |
|---------|---------------------|----------------------|-------|
| **Local** | FREE | FREE | Instant |
| **Mumbai (testnet)** | FREE | FREE | ~2 seconds |
| **Polygon Mainnet** | ~$0.01 | ~$10-20 | ~2 seconds |
| **Ethereum Sepolia** | FREE | FREE | ~12 seconds |
| **Ethereum Mainnet** | ~$5-50 | ~$5,000-50,000 | ~12 seconds |

**Recommendation**: Use **Polygon Mainnet** for production (cheap and fast)

---

## 📊 Running Federated Learning on Blockchain

### Basic Example

```python
from spectral_sentinel.blockchain import BlockchainConnector, BlockchainConfig

# Initialize
config = BlockchainConfig(network="mumbai")  # or "local" for testing
connector = BlockchainConnector(config)

# Register clients (do this once)
client_ids = [0, 1, 2, 3, 4]
connector.register_clients_batch(client_ids)

# Start federated learning round
connector.start_round()
round_num = connector.get_current_round()

# Clients submit model updates
for client_id in client_ids:
    model_dict = train_local_model()  # Your training function
    connector.submit_model_update(model_dict, client_id, round_num)

# Server retrieves and aggregates
models = []
for client_id in client_ids:
    model = connector.get_model_update(round_num, client_id)
    models.append(model)

aggregated = federated_average(models)  # Your aggregation function

# Finalize round
connector.finalize_round(aggregated, round_num)
```

---

## 🔧 Troubleshooting

### Error: "Failed to connect to RPC"

- Check your internet connection
- Try alternative RPC URLs in .env:
  ```
  MUMBAI_RPC_URL=https://rpc-mumbai.matic.today
  # or
  MUMBAI_RPC_URL=https://matic-mumbai.chainstacklabs.com
  ```

### Error: "Insufficient funds"

- Check wallet balance: Visit [Mumbai PolygonScan](https://mumbai.polygonscan.com/)
- Get more tokens from faucet
- For local network, make sure Hardhat node is running

### Error: "Contract not loaded"

- Make sure you deployed the contract
- Check CONTRACT_ADDRESS in .env matches deployment
- Run: `python scripts/setup_testnet.py` for guided setup

### Error: "Transaction failed"

- Check gas limit (increase in .env if needed)
- Check network congestion
- Verify you have sufficient balance
- Check transaction on block explorer for details

### NPM/Node.js Not Found

Install Node.js from [https://nodejs.org/](https://nodejs.org/)

---

## 🔐 Security Best Practices

1. **NEVER commit .env file to git** (already in .gitignore)
2. **Use different wallets** for testnet and mainnet
3. **Only fund mainnet wallet** with necessary amount
4. **Rotate keys regularly** for production
5. **Consider hardware wallet** for mainnet with large funds

---

## 📡 Network Information

### Polygon Mumbai (Testnet)
- Chain ID: 80001
- RPC: https://rpc-mumbai.maticvigil.com
- Explorer: https://mumbai.polygonscan.com/
- Faucet: https://faucet.polygon.technology/
- Currency: MATIC (free)

### Polygon Mainnet
- Chain ID: 137
- RPC: https://polygon-rpc.com
- Explorer: https://polygonscan.com/
- Currency: MATIC ($)

### Ethereum Sepolia (Testnet)
- Chain ID: 11155111
- RPC: https://rpc.sepolia.org
- Explorer: https://sepolia.etherscan.io/
- Faucet: https://sepoliafaucet.com/
- Currency: ETH (free)

### Ethereum Mainnet
- Chain ID: 1
- RPC: https://eth.llamarpc.com
- Explorer: https://etherscan.io/
- Currency: ETH ($$$)

---

## 🎯 Next Steps

1. ✅ **Test locally** - Make sure everything works on local blockchain
2. ✅ **Deploy to Mumbai** - Test on real testnet (free)
3. ✅ **Run small experiment** - 5 clients, 5 rounds
4. ✅ **Run full experiment** - 20 clients, 50 rounds
5. 🚀 **Deploy to Polygon Mainnet** - Production ready!

---

## 📚 Additional Resources

- [Hardhat Documentation](https://hardhat.org/docs)
- [Web3.py Documentation](https://web3py.readthedocs.io/)
- [Polygon Documentation](https://docs.polygon.technology/)
- [Ethereum Documentation](https://ethereum.org/en/developers/docs/)
- [Solidity Documentation](https://docs.soliditylang.org/)

---

## ❓ Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Check deployment.json for deployment info
3. Check block explorer for transaction details
4. Run `python scripts/setup_testnet.py` for guided setup
5. Review test_blockchain.py output for specific errors

---

**Happy blockchain federated learning! 🎉**
