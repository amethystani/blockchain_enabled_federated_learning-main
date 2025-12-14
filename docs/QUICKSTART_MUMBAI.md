# 🚀 QUICK START - Mumbai Testnet

## You're Almost Ready! Here's What to Do:

### ✅ Step 1: Export Your Private Key from MetaMask

**IMPORTANT**: You need your PRIVATE KEY, not your wallet address!

1. **Open MetaMask** browser extension
2. **Click the 3 dots (⋮)** next to your account name  
3. **Select "Account Details"**
4. **Click "Export Private Key"**
5. **Enter your MetaMask password**
6. **Copy the private key** (it looks like: `abc123def456...` - 64 characters)

⚠️ **Your wallet address is**: `0x0f8bbF3E6c8AA11C79Af8590B49144f09ed5d7d2`  
⚠️ **Private key looks different** - it's 64 hex characters!

---

### ✅ Step 2: Run Setup Script

```bash
cd /Users/animesh/Downloads/blockchain_enabled_federated_learning-main

# Run interactive setup
python3 scripts/quick_setup_mumbai.py
```

When prompted, **paste your PRIVATE KEY** (the 64-character one from MetaMask)

---

### ✅ Step 3: Wait for MATIC (2-3 minutes)

Check if MATIC has arrived:

```bash
python3 scripts/check_balance.py
```

Expected output:
```
💰 Balance: 0.5000 MATIC
✅ Sufficient balance for deployment!
```

---

### ✅ Step 4: Install Dependencies

```bash
# Install Node.js packages for Hardhat
npm install
```

---

### ✅ Step 5: Deploy to Mumbai Testnet

```bash
# This will deploy the smart contract to Mumbai
python3 scripts/setup_testnet.py
```

Expected output:
```
✅ Contract deployed to: 0x...
🔗 Explorer: https://mumbai.polygonscan.com/address/0x...
```

---

### ✅ Step 6: Run Blockchain FL Test

```bash
# Test the complete blockchain integration
python3 test_blockchain.py
```

This will:
- ✅ Connect to Mumbai testnet
- ✅ Register 5 test clients
- ✅ Submit model updates
- ✅ Verify on blockchain
- ✅ Show transaction URLs

---

## 🎯 What You'll See

All your transactions will be visible on **Mumbai PolygonScan**:
- https://mumbai.polygonscan.com/address/0x0f8bbF3E6c8AA11C79Af8590B49144f09ed5d7d2

You'll see:
- ✅ Contract deployment transaction
- ✅ Client registration transactions  
- ✅ Model submission transactions
- ✅ Round finalization transactions

---

## 💰 Costs

**Mumbai Testnet = 100% FREE!**
- All transactions are free
- MATIC from faucet is free
- Perfect for testing

---

## 🆘 Troubleshooting

### "Invalid private key length"
→ You're entering the wallet address instead of private key  
→ Private key is 64 characters (without 0x)  
→ Export from MetaMask → Account Details → Export Private Key

### "Insufficient balance"
→ Wait 2-3 minutes for faucet  
→ Run: `python3 scripts/check_balance.py`  
→ Get more at: https://faucet.polygon.technology/

### "npm: command not found"
→ Install Node.js from: https://nodejs.org/

---

## 📝 Manual .env Setup (Alternative)

If you prefer, create `.env` file manually:

```bash
# Create .env file
cat > .env << 'EOF'
BLOCKCHAIN_NETWORK=mumbai
MUMBAI_RPC_URL=https://rpc-mumbai.maticvigil.com
WALLET_ADDRESS=0x0f8bbF3E6c8AA11C79Af8590B49144f09ed5d7d2
PRIVATE_KEY=YOUR_64_CHARACTER_PRIVATE_KEY_HERE
CONTRACT_ADDRESS=
GAS_LIMIT=500000
CONFIRMATION_BLOCKS=1
USE_IPFS=false
EOF

# Replace YOUR_64_CHARACTER_PRIVATE_KEY_HERE with your actual key
nano .env
```

---

## ✅ Summary

1. **Export private key** from MetaMask (64 chars)
2. **Run**: `python3 scripts/quick_setup_mumbai.py`
3. **Wait** 2-3 min, then: `python3 scripts/check_balance.py`
4. **Install**: `npm install`
5. **Deploy**: `python3 scripts/setup_testnet.py`
6. **Test**: `python3 test_blockchain.py`

**That's it!** You'll have real blockchain federated learning running on Mumbai testnet! 🎉
