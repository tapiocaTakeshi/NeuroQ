#!/usr/bin/env python3
"""
NeuroQ Model Downloader
=======================
Downloads the pretrained model file when Git LFS is not available or fails.

This script provides an alternative method to obtain the neuroq_pretrained.pt
model file when `git lfs pull` doesn't work due to network/proxy issues.

Usage:
    python download_model.py
    python download_model.py --output /custom/path/neuroq_pretrained.pt
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def is_lfs_pointer(file_path: str) -> bool:
    """Check if a file is a Git LFS pointer file."""
    if not os.path.exists(file_path):
        return False

    file_size = os.path.getsize(file_path)
    if file_size > 1024:
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(200)
            return content.startswith('version https://git-lfs.github.com/spec/')
    except:
        return False


def check_git_lfs_available() -> bool:
    """Check if Git LFS is installed."""
    try:
        result = subprocess.run(['git', 'lfs', 'version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def try_git_lfs_pull(file_path: str) -> bool:
    """Try to pull the file using Git LFS."""
    try:
        print("🔄 Attempting to pull from Git LFS...")
        result = subprocess.run(['git', 'lfs', 'pull', '--include', file_path],
                              capture_output=True,
                              text=True,
                              timeout=60)

        if result.returncode == 0:
            print("✅ Git LFS pull successful!")
            return True
        else:
            print(f"❌ Git LFS pull failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Git LFS pull timed out")
        return False
    except Exception as e:
        print(f"❌ Git LFS pull error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download NeuroQ pretrained model')
    parser.add_argument('--output', type=str,
                       default='neuroq_pretrained.pt',
                       help='Output file path')
    parser.add_argument('--force', action='store_true',
                       help='Force download even if file exists')

    args = parser.parse_args()
    output_path = Path(args.output)

    print("=" * 70)
    print("📦 NeuroQ Pretrained Model Downloader")
    print("=" * 70)

    # Check if file exists and is valid
    if output_path.exists() and not args.force:
        file_size = output_path.stat().st_size

        if is_lfs_pointer(str(output_path)):
            print(f"⚠️  File exists but is a Git LFS pointer ({file_size} bytes)")
            print(f"    Path: {output_path.absolute()}")
        elif file_size < 1024:
            print(f"⚠️  File exists but is too small ({file_size} bytes)")
            print(f"    Path: {output_path.absolute()}")
        else:
            print(f"✅ Valid model file already exists!")
            print(f"    Path: {output_path.absolute()}")
            print(f"    Size: {file_size / (1024 * 1024):.2f} MB")
            print()
            print("Use --force to download again.")
            return 0

    # Try Git LFS first if available
    if check_git_lfs_available():
        print("✅ Git LFS is installed")

        if try_git_lfs_pull(str(output_path)):
            # Verify the file
            if output_path.exists():
                file_size = output_path.stat().st_size
                if file_size > 10000:  # > 10KB
                    print(f"✅ Model file downloaded successfully via Git LFS!")
                    print(f"    Path: {output_path.absolute()}")
                    print(f"    Size: {file_size / (1024 * 1024):.2f} MB")
                    return 0
                else:
                    print(f"⚠️  File still too small after LFS pull ({file_size} bytes)")
    else:
        print("⚠️  Git LFS is not installed")

    # If Git LFS failed or not available, provide instructions
    print()
    print("=" * 70)
    print("📋 Alternative Methods to Obtain the Model File")
    print("=" * 70)
    print()
    print("Since Git LFS pull failed, you have these options:")
    print()
    print("1. Install Git LFS and retry:")
    print("   $ apt-get install git-lfs  # Ubuntu/Debian")
    print("   $ brew install git-lfs      # macOS")
    print("   $ git lfs install")
    print("   $ git lfs pull")
    print()
    print("2. Download directly from GitHub (if you have access):")
    print("   Visit the repository on GitHub and download the file manually")
    print("   Place it at: " + str(output_path.absolute()))
    print()
    print("3. Use Docker build with Git LFS (recommended for deployment):")
    print("   $ cd neuroq-runpod")
    print("   $ docker build \\")
    print("       --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \\")
    print("       --build-arg GIT_BRANCH=main \\")
    print("       -t neuroq:latest .")
    print()
    print("4. Copy from another machine where Git LFS works:")
    print("   $ scp user@other-machine:/path/to/neuroq_pretrained.pt ./")
    print()
    print("=" * 70)
    print()
    print("Expected file properties:")
    print("  - Size: ~58 MB (58,051,523 bytes)")
    print("  - SHA256: 58e432b209fc8843986dccde566ff2b11612dfd83fa016a793d0c73c1e86ed03")
    print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
