HOW TO RUN THE CODE
===================

This project contains two main components:
1. Spectral Sentinel: A Byzantine-robust federated learning framework (PyTorch-based).
2. Blockchain Integration: A demo of federated learning on a local blockchain.

PREREQUISITES
-------------
- Python 3.10 or higher
- Node.js and npm (for blockchain components)

INSTALLATION
------------
1. Set up a Python virtual environment:
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install Python dependencies:
   # Install core dependencies for Spectral Sentinel (PyTorch)
   pip install -r requirements_spectral.txt
   
   # Install blockchain dependencies (Web3, etc.)
   pip install -r requirements.txt

3. Install Node.js dependencies (for blockchain):
   npm install

RUNNING THE BLOCKCHAIN DEMO
---------------------------
To run the federated learning demo with blockchain integration:

1. Start the local blockchain node (in a separate terminal):
   npx hardhat node

   Keep this terminal open. It simulates the blockchain network.

2. Run the demo script (in a new terminal):
   source venv/bin/activate
   python demos/demo_blockchain_fl.py

   This script will:
   - Connect to the local blockchain.
   - Register clients.
   - Simulate local training and model updates.
   - Aggregate models on-chain.

RUNNING SPECTRAL SENTINEL SIMULATIONS
-------------------------------------
The Spectral Sentinel framework allows you to simulate federated learning with various attacks and defenses.

1. Basic Simulation:
   python spectral_sentinel/experiments/simulate_basic.py

2. Customizing Experiments:
   You can customize the simulation using command-line arguments.

   a) Change Dataset:
      python spectral_sentinel/experiments/simulate_basic.py --dataset cifar10
      (Options: mnist, cifar10, cifar100)

   b) Change Attack Type:
      python spectral_sentinel/experiments/simulate_basic.py --attack_type labelflip
      (Options: minmax, labelflip, alie, adaptive, signflip, zero, gaussian, backdoor, model_poisoning, fall_of_empires, ipm)

   c) Change Aggregator (Defense):
      python spectral_sentinel/experiments/simulate_basic.py --aggregator krum
      (Options: spectral_sentinel, fedavg, krum, geometric_median, trimmed_mean, median, bulyan, signguard, fltrust, flame, crfl, byzshield)

   d) Adjust Byzantine Ratio:
      python spectral_sentinel/experiments/simulate_basic.py --byzantine_ratio 0.4
      (0.4 means 40% of clients are malicious)

   e) Enable Sketching (for large models/memory efficiency):
      python spectral_sentinel/experiments/simulate_basic.py --use_sketching --sketch_size 256

   f) Non-IID Data Distribution:
      python spectral_sentinel/experiments/simulate_basic.py --non_iid_alpha 0.1
      (Lower alpha means more heterogeneous data distribution)

3. Advanced Configuration:
   You can also modify `spectral_sentinel/config.py` to change default parameters like:
   - `num_rounds`: Number of training rounds (default: 50)
   - `local_epochs`: Number of local training epochs per round (default: 5)
   - `batch_size`: Batch size for local training (default: 32)
   - `learning_rate`: Learning rate (default: 0.01)
   - `model_type`: Model architecture (simple_cnn, lenet5, resnet18)

VISUALIZING RESULTS
-------------------
The simulation scripts automatically generate visualizations in the `results` directory (or specified `--save_path`).

- `training_curves.png`: Shows accuracy and loss over time.
- `detection_metrics.png`: Shows how many Byzantine clients were detected per round.
- `spectral_analysis.png`: (For Spectral Sentinel) Visualizes the spectral density and anomaly detection.

TROUBLESHOOTING
---------------
- If you encounter "Module not found" errors, ensure you have activated your virtual environment and installed all requirements.
- If the blockchain demo fails to connect, make sure `npx hardhat node` is running in a separate terminal window and is ready.
- If you see TensorFlow errors, note that the modern codebase primarily uses PyTorch. The TensorFlow code is legacy.
- For CUDA issues, the code automatically falls back to CPU if CUDA is not available.
