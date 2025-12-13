#!/bin/bash
# NeuroQ Docker Build with Git LFS Pull
# ======================================
# This script builds the Docker image by cloning the repo inside Docker
# and pulling LFS files there. Use this when local git lfs pull fails.

set -e

echo "🚀 Building NeuroQ Docker image with Git LFS auto-pull..."
echo ""
echo "This method will:"
echo "  ✓ Clone the repository inside Docker"
echo "  ✓ Install Git LFS in the container"
echo "  ✓ Automatically pull the LFS model file"
echo "  ✓ Bypass local LFS network issues"
echo ""

# Repository configuration
GIT_REPO_URL="https://github.com/tapiocaTakeshi/NeuroQ.git"
GIT_BRANCH="${GIT_BRANCH:-main}"
IMAGE_TAG="${IMAGE_TAG:-neuroq:latest}"

echo "Repository: $GIT_REPO_URL"
echo "Branch: $GIT_BRANCH"
echo "Image tag: $IMAGE_TAG"
echo ""

# Get script directory and move to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📦 Starting Docker build..."
echo ""

docker build \
  --build-arg GIT_REPO_URL="$GIT_REPO_URL" \
  --build-arg GIT_BRANCH="$GIT_BRANCH" \
  -t "$IMAGE_TAG" \
  .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "To run the container:"
echo "   docker run --gpus all -p 8000:8000 $IMAGE_TAG"
echo ""
echo "To push to Docker Hub:"
echo "   docker tag $IMAGE_TAG yourusername/neuroq:latest"
echo "   docker push yourusername/neuroq:latest"
echo ""
