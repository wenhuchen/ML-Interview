#!/bin/bash

echo "🧹 Cleaning up Ray processes and temporary files..."

# Kill any existing Ray processes
echo "Stopping Ray processes..."
pkill -f "ray" || echo "No Ray processes found"

# Remove Ray temporary directories
echo "Removing Ray temporary directories..."
rm -rf /tmp/ray/session_* 2>/dev/null || echo "No Ray session directories found"
rm -rf /tmp/ray_new_session 2>/dev/null || echo "No new Ray session directory found"
rm -rf /tmp/ray_local 2>/dev/null || echo "No local Ray directory found"

# Clean up any Ray lock files
echo "Removing Ray lock files..."
find /tmp -name "*.lock" -path "*/ray/*" -delete 2>/dev/null || echo "No Ray lock files found"

echo "✅ Ray cleanup completed!"
echo "You can now run your training script: python3 ray_fsdp_pytorch_only.py --num_workers 2"
