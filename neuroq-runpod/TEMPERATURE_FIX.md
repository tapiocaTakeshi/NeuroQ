# Temperature Parameter Fix

## Problem

The RunPod serverless worker was failing with the following error:

```json
{
  "error": "NeuroQuantumAI.generate() got an unexpected keyword argument 'temperature'",
  "status": "FAILED"
}
```

## Root Cause

The `NeuroQuantumAI.generate()` method in `neuroquantum_layered.py` does **not** accept a `temperature` parameter directly. Instead, it uses:

- **Layered mode**: `temp_min` and `temp_max` parameters
- **Brain mode**: `temperature_min` and `temperature_max` parameters

The old handler code was incorrectly passing `temperature` directly to the `generate()` method.

## Solution

The handler code has been updated to properly map the `temperature` input parameter to the correct method parameters:

### Layered Mode (neuroquantum_layered.py)

```python
# handler.py lines 248-254
result = model.generate(
    prompt=prompt,
    max_length=max_length,
    temp_min=temperature * 0.8,  # Convert temperature to temp_min/max range
    temp_max=temperature * 1.2,
    top_k=top_k,
    top_p=top_p
)
```

### Brain Mode (neuroquantum_brain.py)

```python
# handler.py lines 266-271
result = model.generate(
    prompt=prompt,
    max_length=max_length,
    temperature_min=temperature * 0.8,  # Convert temperature to temperature_min/max range
    temperature_max=temperature * 1.2
)
```

## What Was Changed

### Before (OLD CODE - BROKEN)

```python
# Layered mode
result = model.generate(
    prompt=prompt,
    max_length=max_length,
    temperature=temperature,  # ❌ WRONG - this parameter doesn't exist
    top_k=top_k,
    top_p=top_p
)

# Brain mode
result = model.generate(
    prompt=prompt,
    max_length=max_length,
    temperature=temperature  # ❌ WRONG - this parameter doesn't exist
)
```

### After (NEW CODE - FIXED)

```python
# Layered mode
result = model.generate(
    prompt=prompt,
    max_length=max_length,
    temp_min=temperature * 0.8,  # ✅ CORRECT
    temp_max=temperature * 1.2,   # ✅ CORRECT
    top_k=top_k,
    top_p=top_p
)

# Brain mode
result = model.generate(
    prompt=prompt,
    max_length=max_length,
    temperature_min=temperature * 0.8,  # ✅ CORRECT
    temperature_max=temperature * 1.2   # ✅ CORRECT
)
```

## Method Signatures

### NeuroQuantumAI.generate() (Layered Mode)

From `neuroquantum_layered.py` line 1160:

```python
def generate(
    self,
    prompt: str = "",
    max_length: int = 100,
    temp_min: float = 0.4,       # Temperature lower bound
    temp_max: float = 0.8,       # Temperature upper bound
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 2.0,
) -> str:
```

### NeuroQuantumBrainAI.generate() (Brain Mode)

Uses `temperature_min` and `temperature_max` parameters.

## Next Steps to Deploy the Fix

Since the code has been fixed in the repository but the RunPod worker is still failing, you need to **rebuild and redeploy** the Docker container:

### 1. Rebuild the Docker Image

```bash
cd neuroq-runpod
docker build -t <your-registry>/neuroq-runpod:latest .
```

### 2. Push to Container Registry

```bash
docker push <your-registry>/neuroq-runpod:latest
```

### 3. Update RunPod Template

- Go to your RunPod dashboard
- Update your serverless template to use the new image tag
- Deploy the updated template

### 4. Test the Fix

Send a test request:

```json
{
  "input": {
    "action": "generate",
    "prompt": "こんにちは",
    "mode": "layered",
    "max_length": 100,
    "temperature": 0.8,
    "pretrain": false
  }
}
```

Expected response:

```json
{
  "status": "success",
  "mode": "layered",
  "prompt": "こんにちは",
  "generated_text": "...",
  "is_pretrained": false
}
```

## Temperature Mapping Logic

The handler converts a single `temperature` value (e.g., 0.8) to a range:

- `temp_min = temperature * 0.8` (e.g., 0.64)
- `temp_max = temperature * 1.2` (e.g., 0.96)

This allows for dynamic temperature variation during generation, which is important for the APQB quantum theory implementation where θ (theta) varies based on temperature T.

## Files Modified

- `neuroq-runpod/handler.py` - Fixed temperature parameter mapping for both layered and brain modes

## Related Commits

- `22639a0` - Fix temperature parameter mapping in RunPod handler
- `a071435` - Fix temperature params, update .gitignore
- `ac0effe` - commit1

## Verification

The repository code is now correct. The error you're seeing is because the **deployed RunPod container** is still running the **old code**. You must rebuild and redeploy the container to apply this fix.

To verify the fix is in place locally:

```bash
cd neuroq-runpod
grep -A 7 "if mode == \"layered\"" handler.py
```

You should see `temp_min` and `temp_max` being used, not `temperature`.
