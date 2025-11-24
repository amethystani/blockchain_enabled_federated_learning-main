require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
    solidity: {
        version: "0.8.20",
        settings: {
            optimizer: {
                enabled: true,
                runs: 200
            }
        }
    },

    networks: {
        // Local Hardhat network (for testing)
        hardhat: {
            chainId: 31337,
            accounts: {
                count: 25, // Create 25 accounts for testing (1 owner + up to 24 clients)
                accountsBalance: "10000000000000000000000" // 10000 ETH per account
            }
        },

        // Localhost (for running local Hardhat node)
        localhost: {
            url: "http://127.0.0.1:8545",
            chainId: 31337
        },

        // Polygon Amoy Testnet (New)
        amoy: {
            url: process.env.AMOY_RPC_URL || "https://rpc-amoy.polygon.technology/",
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
            chainId: 80002,
            gasPrice: "auto"
        },

        // Polygon Mumbai Testnet (Deprecated)
        mumbai: {
            url: process.env.MUMBAI_RPC_URL || "https://rpc-mumbai.maticvigil.com",
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
            chainId: 80001,
            gasPrice: 20000000000, // 20 gwei
            timeout: 60000
        },

        // Polygon Mainnet (cheap - recommended for production)
        polygon: {
            url: process.env.POLYGON_RPC_URL || "https://polygon-rpc.com",
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
            chainId: 137,
            gasPrice: 50000000000, // 50 gwei
            timeout: 60000
        },

        // Ethereum Sepolia Testnet (FREE - slower but good for final validation)
        sepolia: {
            url: process.env.SEPOLIA_RPC_URL || "https://rpc.sepolia.org",
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
            chainId: 11155111,
            gasPrice: 20000000000, // 20 gwei
            timeout: 120000
        },

        // Ethereum Mainnet (EXPENSIVE - only use if absolutely necessary)
        mainnet: {
            url: process.env.MAINNET_RPC_URL || "https://eth.llamarpc.com",
            accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
            chainId: 1,
            gasPrice: 30000000000, // 30 gwei (check current gas prices!)
            timeout: 120000
        }
    },

    etherscan: {
        // Optional: for contract verification
        apiKey: {
            polygon: process.env.POLYGONSCAN_API_KEY || "",
            polygonMumbai: process.env.POLYGONSCAN_API_KEY || "",
            sepolia: process.env.ETHERSCAN_API_KEY || "",
            mainnet: process.env.ETHERSCAN_API_KEY || ""
        }
    },

    paths: {
        sources: "./contracts",
        tests: "./test",
        cache: "./cache",
        artifacts: "./artifacts"
    }
};
