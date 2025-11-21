#!/bin/bash

# Blockchain-Enabled Federated Learning - Setup and Run Script
# This script sets up the environment and runs the federated learning experiments

echo "=========================================="
echo " Blockchain-Enabled Federated Learning"
echo " Setup and Execution Script"
echo "=========================================="
echo ""

# Step 1: Activate virtual environment
echo "[Step 1] Activating virtual environment..."
source fed_learning_env/bin/activate

# Step 2: Upgrade pip
echo "[Step 2] Upgrading pip..."
pip install --upgrade pip

# Step 3: Install required packages
echo "[Step 3] Installing TensorFlow and dependencies..."
echo "This may take several minutes..."

pip install tensorflow==2.9.0
pip install tensorflow-federated==0.31.0
pip install numpy matplotlib

# Step 4: Verify installation
echo ""
echo "[Step 4] Verifying installation..."
python -c "import tensorflow as tf; print('✓ TensorFlow version:', tf.__version__)"
python -c "import tensorflow_federated as tff; print('✓ TensorFlow Federated version:', tff.__version__)"
python -c "import numpy as np; print('✓ NumPy version:', np.__version__)"
python -c "import matplotlib; print('✓ Matplotlib version:', matplotlib.__version__)"

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run the federated learning scripts:"
echo ""
echo "1. For the main s-FLchain vs a-FLchain comparison:"
echo "   cd 'Code & Results/TensorFlow code'"
echo "   python sFLchain_vs_aFLchain.py"
echo ""
echo "2. For centralized baseline:"
echo "   cd 'Code & Results/TensorFlow code'"
echo "   python centalized_baseline.py"
echo ""
echo "=========================================="
echo ""

# Ask user if they want to run the main script
read -p "Would you like to run the main s-FLchain vs a-FLchain script now? (y/n): " choice
if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo ""
    echo "[Running] s-FLchain vs a-FLchain comparison..."
    echo "This will take 2-6 hours depending on your hardware."
    echo "Results will be saved as .txt files in the TensorFlow code directory."
    echo ""
    cd "Code & Results/TensorFlow code"
    python sFLchain_vs_aFLchain.py
else
    echo ""
    echo "You can run the scripts manually when ready."
    echo "Don't forget to activate the virtual environment first:"
    echo "  source fed_learning_env/bin/activate"
fi
