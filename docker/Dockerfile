# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.0.1 with CUDA 11.8
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and compatible CUDA wheels
RUN pip install torch-geometric==2.3.1
RUN pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Set working directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project source code
COPY . .

RUN mkdir -p /workspace/data /workspace/results /workspace/logs

# Default command
CMD ["/bin/bash"]