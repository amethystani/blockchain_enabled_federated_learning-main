# Modern Setup Guide (Python 3.12+ Compatible)
## Blockchain-Enabled Federated Learning Project

**Updated: November 2024**

This guide has been updated to work with **Python 3.12 and 3.13**, replacing the deprecated TensorFlow Federated library with a custom implementation.

---

## 🔄 Major Changes

### What Changed?
1. **Removed TensorFlow Federated (TFF)** - Not compatible with Python 3.12+
2. **Implemented Custom FedAvg** - Pure TensorFlow implementation of the Federated Averaging algorithm
3. **Updated Dependencies** - All packages now compatible with Python 3.12+
4. **Modern TensorFlow** - Using TensorFlow 2.15+ with tensorflow-datasets

### Why These Changes?
- TensorFlow Federated only supports Python 3.9-3.11
- Python 3.12+ offers better performance and security
- Custom implementation provides more control and flexibility

---

## 📋 Prerequisites

- **Python 3.12 or 3.13** (check with `python3 --version`)
- **pip** package manager
- At least 4GB RAM (8GB recommended)
- 2GB free disk space

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd /Users/animesh/Downloads/blockchain_enabled_federated_learning-main

# Make the setup script executable
chmod +x setup_and_run_modern.sh

# Run the setup script
./setup_and_run_modern.sh
```

### Option 2: Manual Setup

```bash
# Navigate to project directory
cd /Users/animesh/Downloads/blockchain_enabled_federated_learning-main

# Create virtual environment
python3 -m venv fed_learning_modern_env

# Activate virtual environment
source fed_learning_modern_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import tensorflow_datasets as tfds; print('TF Datasets:', tfds.__version__)"
```

---

## 📦 Dependencies

The project now uses these modern, Python 3.12+ compatible packages:

- **TensorFlow 2.15+** - Core ML framework
- **TensorFlow Datasets 4.9+** - Dataset loading (replaces TFF datasets)
- **NumPy 1.24+** - Numerical computing
- **Matplotlib 3.7+** - Visualization
- **tqdm** - Progress bars
- **nest-asyncio** - Async utilities

See `requirements.txt` for the complete list.

---

## 🏃 Running the Code

### 1. Centralized Baseline (Quick Test)

Run this first to verify everything works:

```bash
cd "Code & Results/TensorFlow code"
python centalized_baseline_modern.py
```

**What it does:**
- Trains a centralized neural network on EMNIST digits
- Takes ~5-10 minutes on CPU
- Outputs accuracy and loss metrics
- Saves model to `centralized_baseline_model.h5`

**Expected output:**
```
Final Training Accuracy: ~0.95
Test Accuracy: ~0.94
```

### 2. Federated Learning (Main Experiment)

```bash
cd "Code & Results/TensorFlow code"
python sFLchain_vs_aFLchain_modern.py
```

**What it does:**
- Simulates 200 clients with distributed data
- Runs both synchronous (100% participation) and asynchronous (75% participation) FL
- Trains for 200 communication rounds
- Saves detailed metrics for analysis

**Runtime:** 2-6 hours depending on hardware

**Output files:**
- `train_loss_K200_0.75.txt` - Training loss (async)
- `train_accuracy_K200_0.75.txt` - Training accuracy (async)
- `test_loss_K200_0.75.txt` - Test loss (async)
- `test_accuracy_K200_0.75.txt` - Test accuracy (async)
- `eval_loss_K200_0.75.txt` - Evaluation loss (async)
- `eval_accuracy_K200_0.75.txt` - Evaluation accuracy (async)
- `iteration_time_K200_0.75.txt` - Time per round (async)
- Similar files for `_1.txt` (synchronous, 100% participation)
- `test_figure.jpg` - Data distribution visualization

---

## 🔬 Understanding the Implementation

### Federated Averaging (FedAvg) Algorithm

The custom implementation follows the original FedAvg algorithm:

1. **Initialize** global model on server
2. **For each round:**
   - Select random subset of clients (based on participation rate)
   - Send global model to selected clients
   - **Local training:** Each client trains on their local data
   - **Aggregation:** Server averages client models (weighted by dataset size)
   - Update global model with averaged weights
3. **Evaluate** on test data

### Key Classes

- **`FederatedAveraging`** - Main FL orchestrator
  - `train_round()` - Execute one FL round
  - `client_update()` - Train on client's local data
  - `federated_averaging()` - Weighted averaging of models
  - `evaluate()` - Evaluate global model

### Data Distribution

- **IID (Independent and Identically Distributed):** Each client has data from all 10 classes
- **Non-IID:** Each client has data from only `NUM_CLASSES_PER_USER` classes
- Current setting: `NUM_CLASSES_PER_USER = 10` (IID)

---

## ⚙️ Configuration

Edit these parameters in the scripts to customize experiments:

### In `sFLchain_vs_aFLchain_modern.py`:

```python
# Number of federated learning rounds
NUM_ROUNDS_FL = 200

# Local training epochs per round
NUM_EPOCHS = 5

# Batch size for local training
BATCH_SIZE = 20

# Client learning rate
LEARNING_RATE_CLIENT = 0.01

# Server learning rate (for FedAvg)
LEARNING_RATE_SERVER = 1.00

# Data heterogeneity (10 = IID, <10 = non-IID)
NUM_CLASSES_PER_USER = 10

# Number of clients
PARTITIONS = [200]

# Participation rates (0.75 = 75%, 1 = 100%)
PERCENTAGES = [0.75, 1]
```

---

## 📊 Monitoring Progress

While training, you'll see:

```
FL Rounds: 45%|████▌     | 90/200 [12:34<13:56, 7.6s/it]

Round 90/200:
  Train Loss: 0.3245, Train Acc: 0.9123
  Test Loss: 0.3421, Test Acc: 0.9087
  Eval Loss: 0.3389, Eval Acc: 0.9102
  Round Time: 7.42s
```

---

## 🐛 Troubleshooting

### Issue: ImportError for TensorFlow

**Solution:**
```bash
pip install --upgrade tensorflow
```

### Issue: Out of Memory

**Solutions:**
- Reduce `BATCH_SIZE` (try 10 instead of 20)
- Reduce `NUM_CLIENTS` (try 100 instead of 200)
- Reduce `NUM_ROUNDS_FL` for testing (try 50 instead of 200)

### Issue: Slow Training

**Solutions:**
- Use GPU if available (TensorFlow will auto-detect)
- Reduce `NUM_ROUNDS_FL` for initial testing
- Increase `BATCH_SIZE` if you have enough memory

### Issue: Dataset Download Fails

**Solution:**
```bash
# Manually download EMNIST
python -c "import tensorflow_datasets as tfds; tfds.load('emnist/digits', download=True)"
```

---

## 🔍 Comparing with Original Code

### Original (TFF-based):
```python
import tensorflow_federated as tff

# Load data
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# Build FL process
iterative_process = tff.learning.build_federated_averaging_process(...)
```

### Modern (Pure TensorFlow):
```python
import tensorflow_datasets as tfds

# Load data
train_ds, test_ds = tfds.load('emnist/digits', ...)

# Custom FL implementation
fed_avg = FederatedAveraging(create_keras_model, num_clients)
```

**Benefits of custom implementation:**
- ✅ Python 3.12+ compatible
- ✅ More transparent and customizable
- ✅ Better error messages and debugging
- ✅ Easier to extend and modify

---

## 📈 Expected Results

### Centralized Baseline
- **Training Accuracy:** ~95%
- **Test Accuracy:** ~94%
- **Training Time:** ~5-10 minutes

### Federated Learning (200 clients, 200 rounds)
- **Synchronous (100% participation):**
  - Final Test Accuracy: ~92-94%
  - Average Round Time: ~15-20 seconds
  
- **Asynchronous (75% participation):**
  - Final Test Accuracy: ~90-92%
  - Average Round Time: ~12-15 seconds
  - Faster convergence in later rounds

---

## 🎯 Next Steps

1. **Run centralized baseline** to verify setup
2. **Run federated learning** with default settings
3. **Analyze results** using the generated `.txt` files
4. **Experiment with parameters** (participation rates, non-IID data, etc.)
5. **Visualize results** using the MATLAB code (optional)

---

## 💡 Tips

- Start with fewer rounds (`NUM_ROUNDS_FL = 50`) for testing
- Monitor GPU usage with `nvidia-smi` if using GPU
- Use `tmux` or `screen` for long training sessions
- Save checkpoints for very long experiments (modify the code to add this)

---

## 📚 References

- **FedAvg Algorithm:** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- **TensorFlow Datasets:** https://www.tensorflow.org/datasets
- **EMNIST Dataset:** Cohen et al., "EMNIST: Extending MNIST to handwritten letters" (2017)

---

## ❓ FAQ

**Q: Can I use the old TFF-based code?**  
A: Only if you downgrade to Python 3.11 or earlier. Not recommended.

**Q: Is the custom implementation equivalent to TFF?**  
A: Yes, it implements the same FedAvg algorithm. Results should be comparable.

**Q: Can I add more clients?**  
A: Yes, modify `PARTITIONS = [500]` for 500 clients. Note: will take longer to train.

**Q: How do I make data non-IID?**  
A: Set `NUM_CLASSES_PER_USER = 5` for each client to have only 5 classes.

---

## 📞 Support

If you encounter issues:
1. Check Python version: `python3 --version` (should be 3.12+)
2. Verify dependencies: `pip list | grep tensorflow`
3. Check error messages carefully
4. Try running centralized baseline first

---

**Good luck with your federated learning experiments! 🚀**
