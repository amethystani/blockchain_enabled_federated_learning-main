const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
    console.log("🚀 Deploying FederatedLearning contract...\n");

    // Get deployment network
    const network = hre.network.name;
    console.log(`📡 Network: ${network}`);

    // Get deployer account
    const [deployer] = await hre.ethers.getSigners();
    console.log(`👤 Deployer address: ${deployer.address}`);

    // Check balance
    const balance = await hre.ethers.provider.getBalance(deployer.address);
    console.log(`💰 Deployer balance: ${hre.ethers.formatEther(balance)} ${getNetworkCurrency(network)}\n`);

    // Check if balance is sufficient
    if (balance === 0n) {
        console.error("❌ ERROR: Deployer account has zero balance!");
        if (network === "mumbai") {
            console.log("💡 Get free MATIC from: https://faucet.polygon.technology/");
        } else if (network === "sepolia") {
            console.log("💡 Get free ETH from: https://sepoliafaucet.com/");
        }
        process.exit(1);
    }

    // Deploy contract
    console.log("⏳ Deploying contract...");
    const FederatedLearning = await hre.ethers.getContractFactory("FederatedLearning");
    const federatedLearning = await FederatedLearning.deploy();

    await federatedLearning.waitForDeployment();
    const contractAddress = await federatedLearning.getAddress();

    console.log(`✅ Contract deployed to: ${contractAddress}\n`);

    // Save deployment info
    const deploymentInfo = {
        network: network,
        contractAddress: contractAddress,
        deployerAddress: deployer.address,
        deploymentTime: new Date().toISOString(),
        blockNumber: await hre.ethers.provider.getBlockNumber(),
        explorerUrl: getExplorerUrl(network, contractAddress)
    };

    // Save to JSON file
    const deploymentPath = path.join(__dirname, "..", "deployment.json");
    fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
    console.log(`📄 Deployment info saved to: deployment.json\n`);

    // Save ABI
    const artifactPath = path.join(__dirname, "..", "artifacts", "contracts", "FederatedLearning.sol", "FederatedLearning.json");
    const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
    const abiPath = path.join(__dirname, "..", "contracts", "FederatedLearning.abi.json");
    fs.writeFileSync(abiPath, JSON.stringify(artifact.abi, null, 2));
    console.log(`📄 ABI saved to: contracts/FederatedLearning.abi.json\n`);

    // Print summary
    console.log("━".repeat(60));
    console.log("📊 DEPLOYMENT SUMMARY");
    console.log("━".repeat(60));
    console.log(`Network:         ${network}`);
    console.log(`Contract:        ${contractAddress}`);
    console.log(`Deployer:        ${deployer.address}`);
    console.log(`Explorer:        ${deploymentInfo.explorerUrl}`);
    console.log("━".repeat(60));

    // Print next steps
    console.log("\n✨ NEXT STEPS:\n");
    console.log("1. Update your .env file:");
    console.log(`   CONTRACT_ADDRESS=${contractAddress}\n`);

    if (network === "mumbai" || network === "sepolia") {
        console.log("2. Verify contract on block explorer (optional):");
        console.log(`   npx hardhat verify --network ${network} ${contractAddress}\n`);
    }

    console.log("3. Test the deployment:");
    console.log(`   python scripts/setup_testnet.py\n`);

    console.log("4. Run blockchain integration test:");
    console.log(`   python test_blockchain.py\n`);

    // If on testnet, print faucet links
    if (network === "amoy") {
        console.log("💡 Need more POL? Get from faucet:");
        console.log("   https://faucet.polygon.technology/\n");
    } else if (network === "mumbai") {
        console.log("💡 Need more MATIC? Get from faucet:");
        console.log("   https://faucet.polygon.technology/\n");
    } else if (network === "sepolia") {
        console.log("💡 Need more ETH? Get from faucets:");
        console.log("   https://sepoliafaucet.com/");
        console.log("   https://faucet.sepolia.dev/\n");
    }
}

function getNetworkCurrency(network) {
    switch (network) {
        case "amoy":
        case "mumbai":
        case "polygon":
            return "MATIC";
        case "sepolia":
        case "mainnet":
            return "ETH";
        default:
            return "ETH";
    }
}

function getExplorerUrl(network, address) {
    switch (network) {
        case "amoy":
            return `https://amoy.polygonscan.com/address/${address}`;
        case "mumbai":
            return `https://mumbai.polygonscan.com/address/${address}`;
        case "polygon":
            return `https://polygonscan.com/address/${address}`;
        case "sepolia":
            return `https://sepolia.etherscan.io/address/${address}`;
        case "mainnet":
            return `https://etherscan.io/address/${address}`;
        default:
            return "N/A (local network)";
    }
}

// Handle errors
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("\n❌ Deployment failed:");
        console.error(error);
        process.exit(1);
    });
