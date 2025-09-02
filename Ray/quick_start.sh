#!/bin/bash

# Quick Start Script for Ray FSDP Training (PyTorch-only version)
# This script will install dependencies, test the setup, and run the PyTorch-only example

set -e  # Exit on any error

echo "🚀 Quick Start for Ray FSDP Training (PyTorch-only)"
echo "=================================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"
echo

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt
echo "✅ Dependencies installed"
echo

# Test the setup
echo "🧪 Testing setup..."
python3 test_ray_setup.py
if [ $? -eq 0 ]; then
    echo "✅ Setup test passed"
else
    echo "❌ Setup test failed. Please check the errors above."
    exit 1
fi
echo

# Ask user if they want to run a quick training example
read -p "🤔 Would you like to run a quick training example? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🏃 Running PyTorch-only FSDP training example..."
    echo "This will train for 1 epoch with 2 workers on MNIST dataset"
    echo "Using ray_fsdp_pytorch_only.py (your original PyTorch FSDP code + Ray process management)"
    echo
    
    # Run PyTorch-only example
    python3 ray_fsdp_pytorch_only.py --num_workers 2 --num_epochs 1 --batch_size 64
    
    if [ $? -eq 0 ]; then
        echo
        echo "🎉 PyTorch-only FSDP training completed successfully!"
        echo
        echo "Next steps:"
        echo "1. Run longer training: python3 ray_fsdp_pytorch_only.py --num_workers 2 --num_epochs 10"
        echo "2. Scale up workers: python3 ray_fsdp_pytorch_only.py --num_workers 4 --batch_size 128"
        echo "3. Customize model: python3 ray_fsdp_pytorch_only.py --num_layers 8 --hidden_size 1024"
        echo
        echo "💡 This approach keeps your original PyTorch FSDP code unchanged while using Ray for process management!"
    else
        echo "❌ Training failed. Check the error messages above."
        exit 1
    fi
else
    echo "👋 No problem! You can run training examples manually:"
    echo "1. Basic: python3 ray_fsdp_pytorch_only.py --num_workers 2"
    echo "2. Custom: python3 ray_fsdp_pytorch_only.py --num_workers 4 --num_epochs 5 --batch_size 256"
fi

echo
echo "🎯 Setup complete! You're using the PyTorch-only Ray FSDP approach."
echo "This gives you maximum control over your FSDP code while leveraging Ray's process management."
echo
echo "Files available:"
echo "- ray_fsdp_pytorch_only.py: Your main training script"
echo "- fsdp_main.py: Original manual FSDP implementation (for comparison)"
echo "- test_ray_setup.py: Setup verification script"
echo "- requirements.txt: Dependencies"
