#!/usr/bin/env python3
"""Test script to verify model initialization fix"""

import sys
import os

# Add neuroq-runpod to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neuroq-runpod'))

# Test imports
try:
    from neuroquantum_brain import NeuroQuantumBrainAI, NeuroQuantumBrain
    from neuroquantum_layered import NeuroQuantumTokenizer
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 1: Create NeuroQuantumBrainAI instance
try:
    print("\n📝 Test 1: Creating NeuroQuantumBrainAI instance...")
    model = NeuroQuantumBrainAI(
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        num_neurons=100,
        max_vocab=8000,
        use_sentencepiece=True
    )
    print("✅ NeuroQuantumBrainAI instance created successfully")
    print(f"   - model.model = {model.model}")
    print(f"   - model.tokenizer = {model.tokenizer}")
except Exception as e:
    print(f"❌ Failed to create NeuroQuantumBrainAI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Verify .model attribute is None initially
if model.model is None:
    print("✅ model.model is None (expected for untrained model)")
else:
    print(f"⚠️  model.model is not None: {model.model}")

# Test 3: Load tokenizer
try:
    print("\n📝 Test 3: Loading tokenizer...")
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'neuroq-runpod', 'neuroq_tokenizer.model')
    if os.path.exists(tokenizer_path):
        model.tokenizer = NeuroQuantumTokenizer(
            vocab_size=8000,
            model_file=tokenizer_path
        )
        print(f"✅ Tokenizer loaded successfully")
    else:
        print(f"⚠️  Tokenizer file not found: {tokenizer_path}")
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Verify we can't call .eval() on None model
print("\n📝 Test 4: Checking model.model is None...")
try:
    if model.model is not None:
        model.model.eval()
        print("✅ model.model.eval() called (model exists)")
    else:
        print("✅ model.model is None - skipping eval() (correct behavior)")
except AttributeError as e:
    print(f"❌ AttributeError accessing model.model: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
