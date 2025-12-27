#!/usr/bin/env python3
"""
Test script to verify the gradient error fix
"""
import sys
import os
sys.path.insert(0, '/home/user/NeuroQ/neuroq-runpod')

import torch
from neuroquantum_brain import NeuroQuantumBrainAI
from neuroquantum_layered import NeuroQuantumTokenizer

print("=" * 60)
print("Testing generation fix for gradient error")
print("=" * 60)

# Initialize model
print("\n1. Initializing model...")
model = NeuroQuantumBrainAI(
    embed_dim=128,
    num_heads=4,
    num_layers=3,
    num_neurons=100,
    max_vocab=8000,
    use_sentencepiece=True
)

# Load tokenizer
tokenizer_path = "/home/user/NeuroQ/neuroq-runpod/neuroq_tokenizer.model"
if os.path.exists(tokenizer_path):
    print(f"✅ Loading tokenizer from {tokenizer_path}")
    model.tokenizer = NeuroQuantumTokenizer(
        vocab_size=8000,
        model_file=tokenizer_path
    )
else:
    print(f"❌ Tokenizer not found at {tokenizer_path}")
    sys.exit(1)

# Create a simple model for testing (without training)
print("\n2. Creating test model...")
from neuroquantum_brain import NeuroQuantumBrain

model.model = NeuroQuantumBrain(
    vocab_size=8000,
    embed_dim=128,
    num_heads=4,
    num_layers=3,
    num_neurons=100,
    max_seq_len=256,
    dropout=0.1
).to(model.device)

# Set to eval mode
model.model.eval()

# Test generation
print("\n3. Testing generation...")
test_prompts = [
    "こんにちは",
    "人工知能とは",
    "量子コンピュータは"
]

for prompt in test_prompts:
    print(f"\n   Testing prompt: '{prompt}'")
    try:
        # Test with torch.no_grad() wrapper (simulating handler behavior)
        with torch.no_grad():
            result = model.generate(
                prompt=prompt,
                max_length=50,
                temperature_min=0.4,
                temperature_max=0.7,
                top_k=40,
                top_p=0.9
            )

        print(f"   ✅ Success! Generated: {result[:100]}...")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
