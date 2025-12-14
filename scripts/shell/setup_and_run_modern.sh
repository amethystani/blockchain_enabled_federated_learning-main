#!/bin/bash

# Modern Blockchain-Enabled Federated Learning Setup (Python 3.12+ Compatible)
# This script sets up the environment with updated dependencies

echo "=========================================="
echo " Blockchain-Enabled Federated Learning"
echo " Modern Setup Script (Python 3.12+)"
echo "=========================================="
echo ""

# Check Python version
echo "[Step 1] Checking Python version..."
python3 --version
echo ""

# Create virtual environment
echo "[Step 2] Creating virtual environment..."
python3 -m venv fed_learning_modern_env
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "[Step 3] Activating virtual environment..."
source fed_learning_modern_env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "[Step 4] Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "[Step 5] Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo ""

# Verify installation
echo "[Step 6] Verifying installation..."
python -c "import tensorflow as tf; print('✓ TensorFlow version:', tf.__version__)"
python -c "import tensorflow_datasets as tfds; print('✓ TensorFlow Datasets version:', tfds.__version__)"
python -c "import numpy as np; print('✓ NumPy version:', np.__version__)"
python -c "import matplotlib; print('✓ Matplotlib version:', matplotlib.__version__)"
echo ""

echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Important Changes:"
echo "  • Removed TensorFlow Federated (incompatible with Python 3.12+)"
echo "  • Implemented custom FedAvg algorithm using pure TensorFlow"
echo "  • All code now compatible with Python 3.12 and 3.13"
echo ""
echo "To run the modern federated learning code:"
echo "  cd 'Code & Results/TensorFlow code'"
echo "  python sFLchain_vs_aFLchain_modern.py"
echo ""
echo "To run the centralized baseline:"
echo "  cd 'Code & Results/TensorFlow code'"
echo "  python centalized_baseline_modern.py"
echo ""
echo "=========================================="
echo ""

# Ask user if they want to run
read -p "Would you like to run the centralized baseline now? (y/n): " choice
if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo ""
    echo "[Running] Centralized baseline..."
    cd "Code & Results/TensorFlow code"
    python centalized_baseline_modern.py
else
    echo ""
    echo "You can run the scripts manually when ready."
    echo "Don't forget to activate the virtual environment first:"
    echo "  source fed_learning_modern_env/bin/activate"
fi
