# NeuroQ RunPod Deployment Guide

## 🚨 CRITICAL: Docker Image Must Be Rebuilt

**If you're seeing the `train_on_texts` error, you MUST rebuild and redeploy the Docker image!**

The code has been updated, but **the deployed RunPod instance is still running an old version**.

### Quick Fix (3 Commands)

```bash
cd neuroq-runpod
docker build -t YOUR_DOCKER_USERNAME/neuroq-runpod:latest .
docker push YOUR_DOCKER_USERNAME/neuroq-runpod:latest
# Then update your RunPod endpoint to use the new image
```

### Errors You Might See:

**Error 1: train_on_texts attribute error**
```json
{
  "error": "'NeuroQuantumAI' object has no attribute 'train_on_texts'"
}
```

**Error 2: temperature parameter error**
```json
{
  "error": "NeuroQuantumAI.generate() got an unexpected keyword argument 'temperature'"
}
```

These errors occur because the old code (before the fixes) is still deployed on RunPod. **You need to rebuild and redeploy the Docker image.**

## ✅ What Was Fixed

### Fix 1: Added train_on_texts Method

The `train_on_texts` method has been added to both `NeuroQuantumAI` and `NeuroQuantumBrainAI` classes as a backward compatibility alias for the `train()` method.

**Location:** `neuroquantum_layered.py:1163` and `neuroquantum_brain.py:939`

```python
def train_on_texts(self, texts: List[str], epochs: int = 50, batch_size: int = 16,
                   lr: float = 0.001, seq_len: int = 64):
    """
    train()へのエイリアス（後方互換性のため）
    """
    return self.train(texts, epochs=epochs, batch_size=batch_size, lr=lr, seq_len=seq_len)
```

### Fix 2: Temperature Parameter Handling

The handler.py has been updated to correctly convert the `temperature` parameter:

**Layered Mode (lines 248-254):**
```python
result = model.generate(
    prompt=prompt,
    max_length=max_length,
    temp_min=temperature * 0.8,  # ✓ Correct
    temp_max=temperature * 1.2,
    top_k=top_k,
    top_p=top_p
)
```

**Brain Mode (lines 266-270):**
```python
result = model.generate(
    prompt=prompt,
    max_length=max_length,
    temperature_min=temperature * 0.8,  # ✓ Correct
    temperature_max=temperature * 1.2
)
```

## 📋 Deployment Steps

### 1. Build the Docker Image

Navigate to the neuroq-runpod directory and build the Docker image:

```bash
cd neuroq-runpod
docker build -t YOUR_DOCKER_USERNAME/neuroq-runpod:latest .
```

Or with a specific tag for this fix:

```bash
docker build -t YOUR_DOCKER_USERNAME/neuroq-runpod:temperature-fix .
```

### 2. Push to Docker Registry

Push the image to Docker Hub (or your preferred registry):

```bash
# Login to Docker Hub
docker login

# Push the image
docker push YOUR_DOCKER_USERNAME/neuroq-runpod:latest

# Or with specific tag
docker push YOUR_DOCKER_USERNAME/neuroq-runpod:temperature-fix
```

### 3. Update RunPod Endpoint

#### Option A: Via RunPod Web UI

1. Go to https://www.runpod.io/console/serverless
2. Select your endpoint
3. Click "Edit Template"
4. Update the Docker Image field to:
   - `YOUR_DOCKER_USERNAME/neuroq-runpod:latest`
   - or `YOUR_DOCKER_USERNAME/neuroq-runpod:temperature-fix`
5. Save and redeploy

#### Option B: Via RunPod CLI (if you have it installed)

```bash
runpod endpoint update YOUR_ENDPOINT_ID \
  --image YOUR_DOCKER_USERNAME/neuroq-runpod:latest
```

### 4. Verify the Deployment

Test the endpoint with a simple request:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "input": {
      "action": "generate",
      "mode": "layered",
      "prompt": "Hello",
      "max_length": 50,
      "temperature": 0.8
    }
  }'
```

**Expected result:** Should return generated text without the temperature error.

### 5. Run Verification Tests

You can also run the verification scripts included in this directory:

```bash
# Test the handler locally
python test_temperature_fix.py

# Verify parameter mapping
python verify_temperature_params.py
```

## 🔧 Alternative: Using GitHub Container Registry

If you prefer using GitHub Container Registry (ghcr.io):

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Build with GHCR tag
docker build -t ghcr.io/YOUR_GITHUB_USERNAME/neuroq-runpod:latest .

# Push to GHCR
docker push ghcr.io/YOUR_GITHUB_USERNAME/neuroq-runpod:latest

# Update RunPod to use: ghcr.io/YOUR_GITHUB_USERNAME/neuroq-runpod:latest
```

## 📝 Notes

- Make sure all required files are present in the neuroq-runpod directory:
  - handler.py (with the fix)
  - neuroquantum_layered.py
  - neuroquantum_brain.py
  - qbnn_layered.py
  - qbnn_brain.py
  - neuroq_tokenizer.model
  - neuroq_tokenizer.vocab
  - requirements.txt
  - Dockerfile

- The Dockerfile copies these files during the build process (lines 12-23)

## 🐛 Troubleshooting

### Still Getting the Temperature Error?

1. **Check RunPod Status**: Make sure the endpoint has fully restarted
2. **Verify Image**: Confirm the endpoint is using the new image tag
3. **Check Logs**: View RunPod logs to see which version of handler.py is running
4. **Cache Issues**: Try using a new tag (e.g., `temperature-fix-v2`) to force a fresh pull

### Docker Build Issues?

Make sure you're in the neuroq-runpod directory:
```bash
pwd  # Should show: /path/to/NeuroQ/neuroq-runpod
ls -la  # Should show all required files
```

## ✅ Success Criteria

You'll know the deployment succeeded when:
- ✅ API requests with `temperature` parameter work without errors
- ✅ Generated text is returned successfully
- ✅ No "unexpected keyword argument" errors in the response

## 📚 Related Documentation

- [Temperature Fix Summary](./TEMPERATURE_FIX_SUMMARY.md)
- [Temperature Fix Details](./TEMPERATURE_FIX.md)
- [API Request Examples](./API_REQUESTS.md)
