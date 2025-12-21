#!/bin/bash
# Parallax Linux Deployment Script
# For testing on kt980m (Arch Linux + NVIDIA 980M)

set -e

echo "==================================="
echo "Parallax Linux Deployment"
echo "==================================="

# Check if running on remote machine
if [ -z "$SSH_CLIENT" ]; then
    echo "This script should be run on the Linux test machine"
    echo "Usage: ssh kt980m 'bash -s' < deploy_linux.sh"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
sudo pacman -S --needed --noconfirm cmake gcc vulkan-headers vulkan-icd-loader nvidia-utils

# Clone or update repository
if [ -d "parallax-runtime" ]; then
    echo "Updating existing repository..."
    cd parallax-runtime
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/parallax-compiler/parallax-runtime.git
    cd parallax-runtime
fi

# Build
echo "Building Parallax..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPARALLAX_ENABLE_VALIDATION=OFF
make -j$(nproc)

# Run tests
echo "Running tests..."
./tests/test_vulkan_backend

echo ""
echo "✓ Deployment complete!"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "Vulkan Information:"
vulkaninfo --summary | head -20
