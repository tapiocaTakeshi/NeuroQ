#!/usr/bin/env python3
"""
Test script for JSON-based generation requests
Simulates the RunPod handler locally
"""

import sys
import os
import json
from pathlib import Path

# Add neuroq-runpod to path
sys.path.insert(0, str(Path(__file__).parent / 'neuroq-runpod'))

try:
    from neuroquantum_layered import NeuroQuantumAI
    print("✅ Successfully imported NeuroQuantumAI")
except ImportError as e:
    print(f"❌ Failed to import NeuroQuantumAI: {e}")
    sys.exit(1)


def generate_response(input_data):
    """
    Process a generation request with JSON input

    Args:
        input_data: Dictionary with 'action', 'prompt', 'max_length', etc.

    Returns:
        Dictionary with generation result
    """
    try:
        action = input_data.get('action', 'generate')
        prompt = input_data.get('prompt', '')
        max_length = input_data.get('max_length', 100)

        # Optional parameters
        temp_min = input_data.get('temp_min', 0.5)
        temp_max = input_data.get('temp_max', 0.8)
        top_k = input_data.get('top_k', 40)
        top_p = input_data.get('top_p', 0.9)
        repetition_penalty = input_data.get('repetition_penalty', 2.0)
        no_repeat_ngram_size = input_data.get('no_repeat_ngram_size', 3)

        print("\n" + "=" * 70)
        print("🧠 NeuroQ Text Generation")
        print("=" * 70)
        print(f"📝 Prompt: {prompt}")
        print(f"📏 Max length: {max_length}")
        print(f"🌡️  Temperature: {temp_min} - {temp_max}")
        print(f"🎯 Top-k: {top_k}, Top-p: {top_p}")
        print(f"🔄 Repetition penalty: {repetition_penalty}")
        print("=" * 70)

        # Initialize model
        print("\n🔄 Initializing model...")
        model = NeuroQuantumAI(
            embed_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=6,
            max_seq_len=256,
            dropout=0.1,
            lambda_entangle=0.5,
        )

        # Try to load pretrained model
        pretrained_path = Path(__file__).parent / "neuroq_pretrained.pt"
        if pretrained_path.exists():
            print(f"📦 Loading pretrained model: {pretrained_path}")
            try:
                import torch
                checkpoint = torch.load(str(pretrained_path), map_location='cpu')
                if model.model is not None:
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Model loaded successfully")
                else:
                    print("⚠️  Model not initialized. Using fresh model.")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                print("   Using fresh model instead.")
        else:
            print(f"⚠️  Pretrained model not found: {pretrained_path}")
            print("   Using fresh model.")

        # Generate response
        print("\n🔄 Generating response...")
        result = model.generate(
            prompt=prompt,
            max_length=max_length,
            temp_min=temp_min,
            temp_max=temp_max,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        print("\n" + "=" * 70)
        print("✨ Generated Response:")
        print("=" * 70)
        print(result)
        print("=" * 70)

        # Check for repetition
        words = result.split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            repeated_words = [(word, count) for word, count in word_counts.items() if count > 3]
            if repeated_words:
                print(f"\n⚠️  Repetition detected:")
                for word, count in repeated_words:
                    print(f"   '{word}': {count} times")
            else:
                print(f"\n✅ No excessive repetition detected")

        # Return result in RunPod-compatible format
        return {
            "status": "success",
            "output": result,
            "metadata": {
                "prompt": prompt,
                "max_length": max_length,
                "output_length": len(result.split()),
                "parameters": {
                    "temp_min": temp_min,
                    "temp_max": temp_max,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                }
            }
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    """Main entry point"""

    # Default test input (can be overridden by command line JSON)
    default_input = {
        "action": "generate",
        "prompt": "こんにちは",
        "max_length": 100
    }

    # Check if JSON input provided via command line
    if len(sys.argv) > 1:
        try:
            input_json = sys.argv[1]
            input_data = json.loads(input_json)
            if 'input' in input_data:
                input_data = input_data['input']
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON input: {e}")
            print("Using default input instead.")
            input_data = default_input
    else:
        input_data = default_input

    # Process the request
    result = generate_response(input_data)

    # Print final JSON result
    print("\n" + "=" * 70)
    print("📤 JSON Output:")
    print("=" * 70)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 70)

    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
