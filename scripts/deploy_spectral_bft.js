/**
 * deploy_spectral_bft.js
 * Deploy SpectralBFT.sol to local Hardhat node (or any configured network).
 * Saves deployment info to local_deployment.json for Python stack.
 *
 * Usage:
 *   npx hardhat run scripts/deploy_spectral_bft.js --network localhost
 *   npx hardhat run scripts/deploy_spectral_bft.js --network amoy
 */
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer, ...accounts] = await hre.ethers.getSigners();

  console.log("\n=== SpectralBFT Deployment ===");
  console.log("Network:", hre.network.name);
  console.log("Deployer:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Deployer balance:", hre.ethers.formatEther(balance), "ETH");

  if (hre.network.name !== "hardhat" && hre.network.name !== "localhost") {
    if (balance < hre.ethers.parseEther("0.01")) {
      throw new Error("Insufficient balance for deployment. Need at least 0.01 ETH.");
    }
  }

  // Deploy SpectralBFT
  console.log("\nDeploying SpectralBFT...");
  const SpectralBFT = await hre.ethers.getContractFactory("SpectralBFT");
  const contract = await SpectralBFT.deploy();
  await contract.waitForDeployment();
  const contractAddress = await contract.getAddress();
  console.log("SpectralBFT deployed at:", contractAddress);

  // Extract ABI
  const artifact = await hre.artifacts.readArtifact("SpectralBFT");
  const abiPath = path.join(__dirname, "../contracts/SpectralBFT.abi.json");
  fs.writeFileSync(abiPath, JSON.stringify(artifact.abi, null, 2));
  console.log("ABI saved to contracts/SpectralBFT.abi.json");

  // Build account list (Hardhat pre-funded accounts for local stack)
  const accountList = [];
  const allSigners = [deployer, ...accounts];
  for (let i = 0; i < Math.min(allSigners.length, 25); i++) {
    const signer = allSigners[i];
    const bal = await hre.ethers.provider.getBalance(signer.address);
    accountList.push({
      index: i,
      address: signer.address,
      balance_eth: hre.ethers.formatEther(bal)
    });
  }

  // Save deployment info
  const deploymentInfo = {
    network: hre.network.name,
    chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
    contractAddress: contractAddress,
    deployer: deployer.address,
    deployedAt: new Date().toISOString(),
    blockNumber: await hre.ethers.provider.getBlockNumber(),
    abiPath: "contracts/SpectralBFT.abi.json",
    accounts: accountList,
    rpcUrl: hre.network.name === "localhost" ? "http://127.0.0.1:8545" :
            hre.network.name === "hardhat" ? "http://127.0.0.1:8545" : null,
    // Protocol parameters (match contract defaults)
    params: {
      ksThreshold_e6: 50000,
      tailThreshold_e6: 2000000,
      challengeWindowBlocks: 50,
      voteWindowBlocks: 20,
      minStake_wei: "10000000000000000",  // 0.01 ETH
      slashBasisPoints: 5000,
      initialReputation: 800
    }
  };

  const deploymentPath = path.join(__dirname, "../local_deployment.json");
  fs.writeFileSync(deploymentPath, JSON.stringify(deploymentInfo, null, 2));
  console.log("Deployment info saved to local_deployment.json");

  // Summary
  console.log("\n=== Deployment Summary ===");
  console.log("Contract Address:", contractAddress);
  console.log("Available Accounts:", accountList.length);
  if (hre.network.name === "localhost" || hre.network.name === "hardhat") {
    console.log("\nLocal stack ready. Next steps:");
    console.log("  python scripts/start_local_stack.py   (if not already running)");
    console.log("  python experiments/blockchain/experiment_spectral_bft_local.py");
  } else {
    const explorers = {
      amoy: "https://amoy.polygonscan.com/address/",
      polygon: "https://polygonscan.com/address/",
      sepolia: "https://sepolia.etherscan.io/address/"
    };
    const explorer = explorers[hre.network.name];
    if (explorer) {
      console.log("Explorer:", explorer + contractAddress);
    }
  }

  return contractAddress;
}

main()
  .then((addr) => {
    console.log("\nDeployment successful:", addr);
    process.exit(0);
  })
  .catch((err) => {
    console.error("\nDeployment failed:", err);
    process.exit(1);
  });
