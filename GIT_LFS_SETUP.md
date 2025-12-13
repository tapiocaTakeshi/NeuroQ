# Git LFS Setup Guide for NeuroQ

## Problem: Git LFS Pointer File Instead of Actual Model

If you see errors like:
```
❌ File too small (133 bytes). Possibly corrupted.
📊 File size: 0.00 MB
```

This means you have a **Git LFS pointer file** instead of the actual model file (`neuroq_pretrained.pt`).

## What is Git LFS?

Git Large File Storage (LFS) is a Git extension that stores large files (like our 58MB model file) on a separate server, keeping only small "pointer files" in the Git repository. This prevents the repository from becoming too large.

## Solutions

### Option 1: Use the Helper Script (Recommended)

We've created a helper script that attempts to download the model file:

```bash
python download_model.py
```

This script will:
1. Check if the file is a Git LFS pointer
2. Try to pull from Git LFS if available
3. Provide clear instructions if manual action is needed

### Option 2: Manual Git LFS Pull

If you have Git LFS installed or can install it:

```bash
# Install Git LFS (choose your platform)
# Ubuntu/Debian:
apt-get install git-lfs

# macOS:
brew install git-lfs

# Windows:
# Download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Pull the actual files
git lfs pull
```

### Option 3: Docker Build with Automatic LFS Pull (Best for Deployment)

Build the Docker image with automatic LFS pull:

```bash
cd neuroq-runpod

docker build \
  --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \
  --build-arg GIT_BRANCH=main \
  -t neuroq:latest .
```

This method:
- Clones the repository inside Docker
- Pulls LFS files automatically during build
- Validates the model file size
- Fails fast if LFS pull doesn't work

### Option 4: Copy from Another Machine

If you have access to another machine where Git LFS works:

```bash
# On the machine with working LFS
scp neuroq_pretrained.pt user@target-machine:/path/to/NeuroQ/

# Or use any other file transfer method
```

## Verification

After obtaining the actual model file, verify it:

```bash
ls -lh neuroq_pretrained.pt
```

Expected output:
```
-rw-r--r-- 1 user user 55M Dec 13 15:30 neuroq_pretrained.pt
```

**Expected Properties:**
- Size: ~58 MB (58,051,523 bytes)
- SHA256: `58e432b209fc8843986dccde566ff2b11612dfd83fa016a793d0c73c1e86ed03`

**NOT like this (pointer file):**
- Size: 133 bytes

## How to Check if You Have a Pointer File

```bash
# Check file size
ls -lh neuroq_pretrained.pt

# If it's very small (< 1KB), check the contents
head neuroq_pretrained.pt
```

If you see something like:
```
version https://git-lfs.github.com/spec/v1
oid sha256:58e432b209fc8843986dccde566ff2b11612dfd83fa016a793d0c73c1e86ed03
size 58051523
```

This confirms it's a Git LFS pointer file.

## Troubleshooting

### Error: "git: 'lfs' is not a git command"

Git LFS is not installed. Install it using the commands in Option 2 above.

### Error: "batch response: Server error HTTP 502"

This is a network/proxy issue preventing Git LFS from downloading files. Try:
1. Use a different network
2. Use Option 3 (Docker build)
3. Copy the file from another machine (Option 4)

### Error: "Permission denied"

You may need to run with sudo:
```bash
sudo apt-get install git-lfs
```

Or use Docker (Option 3) which handles this automatically.

## For Developers

### Adding New LFS Files

If you need to add new large files to the repository:

```bash
# Track all .pt files with LFS (already configured)
git lfs track "*.pt"

# Add your file
git add your_large_file.pt

# Commit and push
git commit -m "Add large model file"
git push
```

### Checking LFS Status

```bash
# See which files are tracked by LFS
git lfs ls-files

# See LFS configuration
cat .gitattributes
```

## Related Files

- `download_model.py` - Helper script to download the model
- `.gitattributes` - LFS configuration (tracks `*.pt` files)
- `neuroq-runpod/Dockerfile` - Docker build with LFS support
- `neuroq-runpod/neuroq_pretrained.py` - Model loading with LFS pointer detection
- `neuroq-runpod/handler.py` - RunPod handler with validation

## Support

If you continue to have issues:
1. Check the error messages - they now provide specific guidance
2. Run `python download_model.py` for diagnostic information
3. Use Docker build (Option 3) which is the most reliable method
