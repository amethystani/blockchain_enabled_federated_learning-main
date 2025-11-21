# Running the Blockchain-Enabled Federated Learning Project

## Current Status

✅ **Completed Steps:**
1. Created virtual environment with Python 3.12
2. Installed TensorFlow 2.20.0
3. Installed numpy and matplotlib

⚠️ **Issue Encountered:**
- TensorFlow Federated version compatibility issues
- The original code was written for TensorFlow (~2.9) with TensorFlow Federated (~0.31)
- These versions require Python 3.9-3.10
- Your system has Python 3.12 and 3.13, which are too new for those older versions

## **Solution Options**

### **Option 1: Install Python 3.10 (Recommended)**

This is the most compatible approach to run the code as-is.

#### Step 1.1: Install Python 3.10 with Homebrew

```bash
brew install python@3.10
```

#### Step 1.2: Create virtual environment with Python 3.10

```bash
cd /Users/animesh/Downloads/blockchain_enabled_federated_learning-main
rm -rf fed_learning_env
python3.10 -m venv fed_learning_env
source fed_learning_env/bin/activate
```

#### Step 1.3: Install compatible dependencies

```bash
pip install --upgrade pip
pip install tensorflow==2.9.3
pip install tensorflow-federated==0.31.0
pip install numpy==1.23.5 matplotlib
```

#### Step 1.4: Run the main script

```bash
cd "Code & Results/TensorFlow code"
python sFLchain_vs_aFLchain.py
```

---

### **Option 2: Update the Code for Modern TensorFlow** 

Modify the existing code to work with TensorFlow 2.20 and a modern approach.

The main changes needed in `sFLchain_vs_aFLchain.py`:

1. Remove deprecated `tf.compat.v1` calls
2. Update the preprocessing function
3. Update the model building approach

**Status:** I can help you update the code if you choose this option.

---

### **Option 3: Use Docker (Most Isolated)**

Create a Docker container with the exact Python 3.9/3.10 environment.

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install tensorflow==2.9.3 \\
    tensorflow-federated==0.31.0 \\
    numpy==1.23.5 \\
    matplotlib

CMD ["python", "Code & Results/TensorFlow code/sFLchain_vs_aFLchain.py"]
```

---

## **Recommended Next Steps**

### **For Quick Start (Choose Option 1):**

```bash
# Install Python 3.10
brew install python@3.10

# Recreate environment
cd /Users/animesh/Downloads/blockchain_enabled_federated_learning-main
rm -rf fed_learning_env
python3.10 -m venv fed_learning_env
source fed_learning_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install tensorflow==2.9.3 tensorflow-federated==0.31.0 numpy==1.23.5 matplotlib

# Verify installation
python -c "import tensorflow as tf; import tensorflow_federated as tff; print('TF:', tf.__version__, 'TFF:', tff.__version__)"

# Run the main script
cd "Code & Results/TensorFlow code"
python sFLchain_vs_aFLchain.py
```

---

## **What the Script Will Do**

When you run `sFLchain_vs_aFLchain.py`:

1. **Load Data**: Downloads and loads the EMNIST dataset (this may take a few minutes the first time)
2. **Create Non-IID Distribution**: Creates datasets with 10 classes per user
3. **Generate Visualization**: Creates `test_figure.jpg` showing label distribution for 6 sample clients
4. **Train Models**:
   - **Asynchronous FL** (75% participation): Trains for 200 rounds
   - **Synchronous FL** (100% participation): Trains for 200 rounds
5. **Save Results**: Generates `.txt` files with:
   - Training loss and accuracy
   - Test loss and accuracy
   - Evaluation metrics
   - Iteration times

**Expected Runtime**: 2-6 hours (depending on hardware)

**Output Files** (in the TensorFlow code directory):
```
train_loss_K200_0.75.txt
train_accuracy_K200_0.75.txt
test_loss_K200_0.75.txt
test_accuracy_K200_0.75.txt
eval_loss_K200_0.75.txt
eval_accuracy_K200_0.75.txt
iteration_time_K200_0.75.txt

train_loss_K200_1.txt
train_accuracy_K200_1.txt
(...and so on for synchronous mode)

test_figure.jpg
```

---

## **Monitoring Progress**

While the script runs, you can:

1. Watch terminal output for round progress
2. Check generated `.txt` files:
   ```bash
   tail -f train_accuracy_K200_0.75.txt
   ```
3. View the data distribution:
   ```bash
   open test_figure.jpg
   ```

---

## **After Completion**

Once the training is complete, you can:

1. **Analyze Results**: Compare async vs sync performance
2. **Visualize Metrics**: Plot accuracy/loss over time
3. **Run Part 3 (MATLAB)**: Process results for end-to-end analysis

Would you like me to:
- Help you install Python 3.10 and set everything up?
- Update the code to work with modern TensorFlow?
- Create a simple visualization script for the results?
