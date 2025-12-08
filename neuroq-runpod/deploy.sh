#!/bin/bash
# NeuroQ RunPod Deployment Script
# =================================
# This script rebuilds and pushes the Docker image with the temperature parameter fix

set -e  # Exit on any error

# Configuration (EDIT THESE)
DOCKER_REGISTRY="your-registry"  # e.g., "docker.io/username" or "ghcr.io/username"
IMAGE_NAME="neuroq-runpod"
VERSION_TAG="latest"

# Full image name
FULL_IMAGE="${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION_TAG}"

echo "🚀 NeuroQ RunPod Deployment"
echo "================================"
echo "Image: ${FULL_IMAGE}"
echo ""

# Verify we're in the right directory
if [ ! -f "handler.py" ]; then
    echo "❌ Error: handler.py not found. Please run this script from neuroq-runpod/ directory"
    exit 1
fi

# Verify the fix is in place
echo "🔍 Verifying temperature parameter fix..."
if grep -q "temp_min=temperature \* 0.8" handler.py; then
    echo "✅ Fix confirmed: Layered mode uses temp_min/temp_max"
else
    echo "❌ Warning: Fix not found in handler.py"
    exit 1
fi

if grep -q "temperature_min=temperature \* 0.8" handler.py; then
    echo "✅ Fix confirmed: Brain mode uses temperature_min/temperature_max"
else
    echo "❌ Warning: Fix not found in handler.py"
    exit 1
fi

echo ""
echo "📦 Building Docker image..."
docker build -t "${FULL_IMAGE}" .

echo ""
echo "✅ Build complete!"
echo ""
echo "🚢 Pushing to registry..."
docker push "${FULL_IMAGE}"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Go to your RunPod dashboard: https://www.runpod.io/console/serverless"
echo "2. Update your serverless template to use: ${FULL_IMAGE}"
echo "3. Deploy the updated template"
echo ""
echo "🧪 Test with this payload:"
echo '{'
echo '  "input": {'
echo '    "action": "generate",'
echo '    "prompt": "こんにちは",'
echo '    "mode": "layered",'
echo '    "max_length": 100,'
echo '    "temperature": 0.8,'
echo '    "pretrain": false'
echo '  }'
echo '}'
echo ""
echo "Expected: status: success (no temperature parameter error)"
