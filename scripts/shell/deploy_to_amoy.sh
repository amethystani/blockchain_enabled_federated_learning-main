#!/bin/bash
# Deploy contract to Amoy testnet

echo "🚀 Deploying FederatedLearning contract to Amoy testnet..."
echo ""

# Clean cache to avoid disk space issues
rm -rf cache

# Compile contract
echo "📦 Compiling contract..."
npx hardhat compile

# Deploy to Amoy
echo ""
echo "🚀 Deploying to Amoy testnet..."
npx hardhat run scripts/deploy.js --network amoy

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "1. Update .env file with the CONTRACT_ADDRESS from deployment.json"
echo "2. Run: python tests/test_amoy_testnet.py"

