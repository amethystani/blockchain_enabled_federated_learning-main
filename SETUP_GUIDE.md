# Step-by-Step Setup and Execution Guide
## Blockchain-Enabled Federated Learning Project

This guide will help you run the project in three main parts as described in the README.

---

## **Part 0: Environment Setup**

### Step 0.1: Install Required Python Packages

```bash
# Install TensorFlow and TensorFlow Federated
pip3 install tensorflow==2.9.0
pip3 install tensorflow-federated==0.31.0

# Install other dependencies
pip3 install numpy matplotlib
```

**Note:** TensorFlow Federated compatibility depends on specific TensorFlow versions. The versions above are compatible.

### Step 0.2: Verify Installation

```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python3 -c "import tensorflow_federated as tff; print('TFF version:', tff.__version__)"
```

---

## **Part 1: Batch Service Queue Analysis** 

This part analyzes the queueing delay in the Blockchain using a separate simulator.

### Step 1.1: Clone the Batch Service Queue Simulator

```bash
cd /Users/animesh/Downloads
git clone https://github.com/fwilhelmi/batch_service_queue_simulator.git
cd batch_service_queue_simulator
git checkout f846b66
```

### Step 1.2: Run Queue Simulations

Copy the queue scripts from this project to the simulator:

```bash
# Navigate to the Queue code directory
cd "/Users/animesh/Downloads/blockchain_enabled_federated_learning-main/Code & Results/Queue code"

# Review the available queue scripts
ls -la
```

The output results will be saved to `Matlab code/output_queue_simulator/`.

---

## **Part 2: FLchain Analysis (TensorFlow Federated)**

This is the main part that runs the federated learning simulations.

### Step 2.1: Navigate to TensorFlow Code Directory

```bash
cd "/Users/animesh/Downloads/blockchain_enabled_federated_learning-main/Code & Results/TensorFlow code"
```

### Step 2.2: Run Centralized Baseline

This establishes baseline results:

```bash
python3 centalized_baseline.py
```

**Expected:** This will train a centralized model on EMNIST dataset and output accuracy/loss metrics.

### Step 2.3: Run s-FLchain vs a-FLchain Comparison

This is the main script comparing synchronous and asynchronous FL:

```bash
python3 sFLchain_vs_aFLchain.py
```

**What it does:**
- Loads EMNIST dataset
- Creates non-IID data distribution (10 classes per user by default)
- Trains both synchronous (percentage=1) and asynchronous (percentage=0.75) FL models
- Runs for 200 communication rounds
- Uses 200 clients
- Saves results as `.txt` files

**Output files:**
- `train_loss_K200_0.75.txt` (async)
- `train_accuracy_K200_0.75.txt` (async)
- `test_loss_K200_0.75.txt` (async)
- `test_accuracy_K200_0.75.txt` (async)
- `eval_loss_K200_0.75.txt` (async)
- `eval_accuracy_K200_0.75.txt` (async)
- `iteration_time_K200_0.75.txt` (async)
- Similar files for `_1.txt` (sync)
- `test_figure.jpg` (data distribution visualization)

**Estimated runtime:** 2-6 hours depending on your hardware (CPU/GPU)

### Step 2.4: Optional - Run CNN-based Models

```bash
# For EMNIST with CNN
python3 federated_CNN_EMNIST.py

# For CIFAR-10
python3 federated_CIFAR.py
```

---

## **Part 3: End-to-End Analysis (MATLAB)**

This part processes the outputs from Parts 1 and 2.

### Step 3.1: Requirements

You'll need MATLAB installed on your system.

### Step 3.2: Navigate to MATLAB Code

```bash
cd "/Users/animesh/Downloads/blockchain_enabled_federated_learning-main/Code & Results/Matlab code"
```

### Step 3.3: Process Results

Open MATLAB and navigate to the simulation scripts:

```matlab
cd '/Users/animesh/Downloads/blockchain_enabled_federated_learning-main/Code & Results/Matlab code/simulation_scripts'
```

The directories contain:
- `0_preliminary_results/` - Initial FL parameter evaluation
- `1_blockchain_analysis/` - Blockchain queuing delay analysis
- `2_flchain/` - FL accuracy and end-to-end latency analysis

Run the MATLAB scripts in each directory to generate figures and final results.

---

## **Quick Start (Recommended)**

If you want to start immediately with the core federated learning part:

```bash
# 1. Install dependencies
pip3 install tensorflow==2.9.0 tensorflow-federated==0.31.0 numpy matplotlib

# 2. Navigate to TensorFlow code
cd "/Users/animesh/Downloads/blockchain_enabled_federated_learning-main/Code & Results/TensorFlow code"

# 3. Run the main comparison script
python3 sFLchain_vs_aFLchain.py
```

This will start the federated learning training and generate results you can analyze.

---

## **Monitoring Progress**

While `sFLchain_vs_aFLchain.py` is running, you can monitor:
- Terminal output showing round numbers
- Generated `.txt` files in the TensorFlow code directory
- The `test_figure.jpg` file showing data distribution

---

## **Troubleshooting**

### Issue: TensorFlow/TFF Version Conflicts
- Try: `pip3 install tensorflow==2.8.0 tensorflow-federated==0.30.0`

### Issue: Out of Memory
- Reduce `NUM_CLIENTS` in the scripts
- Reduce `NUM_ROUNDS_FL` for faster testing
- Use CPU-only TensorFlow: `pip3 install tensorflow-cpu`

### Issue: Missing MATLAB
- You can skip Part 3 initially and focus on Parts 1 and 2
- Use Python/matplotlib to visualize the `.txt` output files

---

## **Next Steps After Completion**

1. Review the generated `.txt` files with training/testing metrics
2. Compare synchronous (percentage=1) vs asynchronous (percentage=0.75) results
3. Visualize accuracy over time using the data
4. If you have MATLAB, run Part 3 for comprehensive analysis

---

## **Customization**

You can modify parameters in `sFLchain_vs_aFLchain.py`:
- `NUM_CLIENTS`: Change from 200 to experiment with different network sizes
- `NUM_ROUNDS_FL`: Adjust from 200 for longer/shorter training
- `PARTITIONS`: Test different dataset sizes
- `PERCENTAGES`: Modify async participation rates
- `NUM_CLASSES_PER_USER`: Change data heterogeneity (currently 10 = IID)
