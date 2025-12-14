#!/bin/bash
# NeuroQ Docker Build Helper Script
# ===================================
# Git LFSファイルの状態をチェックしてからDockerイメージをビルドするヘルパースクリプト

set -e

echo "🔍 Checking Git LFS files before building Docker image..."
echo ""

# 現在のディレクトリを保存
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Git LFSがインストールされているかチェック
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed!"
    echo "   Please install Git LFS first:"
    echo "   - Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "   - macOS: brew install git-lfs"
    echo "   - Windows: Download from https://git-lfs.github.com/"
    echo ""
    exit 1
fi

# Git LFSが初期化されているかチェック
if ! git lfs env &> /dev/null; then
    echo "⚠️  Git LFS is not initialized. Initializing now..."
    git lfs install
fi

# Check if git remote is using local proxy and fix if needed
REMOTE_URL=$(git config --get remote.origin.url)
if [[ "$REMOTE_URL" == *"127.0.0.1"* ]] || [[ "$REMOTE_URL" == *"local_proxy"* ]]; then
    echo "⚠️  Detected local proxy in git remote URL"
    echo "   Switching to GitHub URL for LFS file access..."
    git remote set-url origin https://github.com/tapiocaTakeshi/NeuroQ.git
    echo "✅ Git remote updated to GitHub"
fi

# neuroq_pretrained.ptのサイズをチェック
MODEL_FILE="neuroq_pretrained.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ $MODEL_FILE not found!"
    echo "   Please ensure the file exists in the repository root."
    exit 1
fi

FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null)
echo "📊 Current file size: $FILE_SIZE bytes"

if [ "$FILE_SIZE" -lt 10000 ]; then
    echo ""
    echo "❌ $MODEL_FILE is too small ($FILE_SIZE bytes)"
    echo "   This is a Git LFS pointer file, not the actual model!"
    echo ""
    echo "🔧 Attempting to pull LFS files..."

    if git lfs pull; then
        echo "✅ Git LFS files pulled successfully!"
        NEW_FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null)
        echo "📊 New file size: $NEW_FILE_SIZE bytes"

        if [ "$NEW_FILE_SIZE" -lt 10000 ]; then
            echo "❌ Failed to pull LFS files properly."
            echo ""
            echo "Alternative build method:"
            echo "   docker build -f neuroq-runpod/Dockerfile \\"
            echo "                --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \\"
            echo "                --build-arg GIT_BRANCH=main \\"
            echo "                -t neuroq:latest ."
            exit 1
        fi
    else
        echo "❌ Failed to pull LFS files."
        echo ""
        echo "You have two options:"
        echo ""
        echo "1. Fix Git LFS and try again:"
        echo "   - Check your network connection"
        echo "   - Verify Git LFS server is accessible"
        echo "   - Try: git lfs fetch --all"
        echo ""
        echo "2. Use repository URL to build (Docker will clone and pull LFS):"
        echo "   docker build -f neuroq-runpod/Dockerfile \\"
        echo "                --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \\"
        echo "                --build-arg GIT_BRANCH=main \\"
        echo "                -t neuroq:latest ."
        exit 1
    fi
else
    echo "✅ $MODEL_FILE size OK!"
fi

echo ""
echo "🚀 Starting Docker build..."
echo ""

# Build from parent directory to include .git for LFS
cd "$SCRIPT_DIR/.."
docker build -f neuroq-runpod/Dockerfile -t neuroq:latest .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "To run the container:"
echo "   docker run --gpus all -p 8000:8000 neuroq:latest"
