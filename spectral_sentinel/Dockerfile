# Spectral Sentinel - Byzantine-Robust Federated Learning
# Production Dockerfile for reproducible deployments

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace/spectral_sentinel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements
COPY requirements_spectral.txt /workspace/requirements_spectral.txt
COPY requirements_phase3.txt /workspace/requirements_phase3.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /workspace/requirements_spectral.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements_phase3.txt

# Install additional production dependencies
RUN pip3 install \
    tensorboard \
    pytest \
    black \
    flake8

# Copy project files
COPY . /workspace/spectral_sentinel/

# Set Python path
ENV PYTHONPATH="/workspace/spectral_sentinel:${PYTHONPATH}"

# Create directories for data and results
RUN mkdir -p /workspace/spectral_sentinel/data \
    /workspace/spectral_sentinel/results \
    /workspace/spectral_sentinel/checkpoints

# Expose ports for TensorBoard and distributed training
EXPOSE 6006 12355

# Default command: print GPU info and help
CMD ["python3", "-c", "from spectral_sentinel.utils.multi_gpu import print_gpu_info; print_gpu_info(); print('\n✓ Spectral Sentinel Docker container ready!\nRun experiments with: python3 spectral_sentinel/experiments/...')"]
