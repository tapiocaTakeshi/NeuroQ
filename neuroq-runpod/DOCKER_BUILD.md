# Docker Build Instructions for NeuroQ RunPod

## Important: Use the Build Script

**Recommended approach:** Use the provided build script which handles all checks automatically:

```bash
./neuroq-runpod/build.sh
```

The build script will:
- Check if Git LFS is installed
- Verify the model file is pulled (not a pointer)
- Pull LFS files if needed
- Build the Docker image with the correct context

## Git LFS Support

The Dockerfile supports two methods for handling Git LFS files (like `neuroq_pretrained.pt`):

### Method 1: Local Build (Default)

Make sure to pull LFS files **before** building the Docker image:

```bash
# Pull LFS files locally first
git lfs pull

# Build the Docker image
docker build -t neuroq:latest .
```

The Dockerfile will:
1. Copy the local `neuroq_pretrained.pt` file
2. Verify it's not an LFS pointer file (checks if size > 10KB)
3. Warn if it appears to be a pointer file

### Method 2: Clone from Git Repository

Build with the repository URL to automatically clone and pull LFS files during build:

```bash
docker build \
  --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \
  --build-arg GIT_BRANCH=main \
  -t neuroq:latest .
```

This method:
1. Clones the repository during Docker build
2. Runs `git lfs pull` to fetch LFS files
3. Copies the LFS-pulled model file to the image
4. Cleans up the cloned repo to save space

**Parameters:**
- `GIT_REPO_URL`: The git repository URL (required for this method)
- `GIT_BRANCH`: The branch to clone (default: `main`)

### Method 3: Clone from Specific Branch

To build from a specific branch (e.g., a feature branch):

```bash
docker build \
  --build-arg GIT_REPO_URL=https://github.com/tapiocaTakeshi/NeuroQ.git \
  --build-arg GIT_BRANCH=claude/fix-japanese-text-generation-01DgfDoTcCJhE9qdos3eGWuC \
  -t neuroq:latest-fix .
```

## Verification

The Dockerfile automatically checks if the pretrained model file is valid:

```
✅ neuroq_pretrained.pt size OK: 12345678 bytes
```

Or warns if it might be an LFS pointer:

```
❌ WARNING: neuroq_pretrained.pt is very small (132 bytes)
   This might be a Git LFS pointer file, not the actual model!
   Run 'git lfs pull' locally before building, or use --build-arg GIT_REPO_URL=<url>
```

## Troubleshooting

### Error: "/.git": not found

**Symptoms:**
```
ERROR: failed to compute cache key: "/.git": not found
```

**Cause:** Earlier versions tried to copy the `.git` directory during Docker build, which is incompatible with Docker BuildKit.

**Solution:** This has been fixed in the latest version. The Dockerfile no longer copies `.git`. Instead:
1. Use the `build.sh` script (recommended)
2. Or ensure LFS files are pulled before manual build
3. Or use the `GIT_REPO_URL` build arg to clone fresh

### Error: "neuroq_pretrained.pt is very small"

This means you're copying an LFS pointer file instead of the actual model.

**Solution:**
```bash
# Install git-lfs if not already installed
git lfs install

# Pull LFS files
git lfs pull

# Rebuild using the build script
./neuroq-runpod/build.sh
```

### Error: "Failed to clone repository"

If using Method 2 and the clone fails:

1. Check the repository URL is correct
2. Ensure you have network access
3. For private repos, you may need to use SSH keys or access tokens

### Manual Build (Advanced)

If you need to build manually without the build script:

```bash
# From repository root
docker build -f neuroq-runpod/Dockerfile -t neuroq:latest .
```

**Important:** Make sure the model file is pulled from LFS first!

## CI/CD Usage

For GitHub Actions or other CI/CD:

```yaml
# .github/workflows/docker-build.yml
- name: Checkout code
  uses: actions/checkout@v3
  with:
    lfs: true  # Important: Enable LFS

- name: Pull LFS files
  run: git lfs pull

- name: Build Docker image
  run: docker build -t neuroq:latest .
```

Or use the git clone method:

```yaml
- name: Build Docker image with git clone
  run: |
    docker build \
      --build-arg GIT_REPO_URL=${{ github.repositoryUrl }} \
      --build-arg GIT_BRANCH=${{ github.ref_name }} \
      -t neuroq:latest .
```

## File Size Reference

Typical file sizes:
- **LFS pointer file:** ~130 bytes (❌ wrong)
- **Actual pretrained model:** Several MB to GB (✅ correct)

## Additional Notes

- Git LFS is already installed in the base image (`runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`)
- The Dockerfile runs `git lfs install` to ensure LFS is configured
- The cloned repository is cleaned up after copying files to minimize image size
- The verification step helps catch LFS pointer issues early in the build process
