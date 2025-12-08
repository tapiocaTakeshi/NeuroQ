#!/usr/bin/env python3
"""
Test script to verify the temperature parameter fix in handler.py
This simulates a RunPod request to ensure the fix works correctly.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Mock the runpod module since we're testing locally
class MockRunPod:
    class serverless:
        @staticmethod
        def start(config):
            print("✅ RunPod serverless would start with handler")
            return None

sys.modules['runpod'] = MockRunPod()

# Import the handler
from handler import handler

def test_layered_mode():
    """Test layered mode with temperature parameter"""
    print("\n🧪 Testing Layered Mode")
    print("=" * 60)

    event = {
        "input": {
            "action": "generate",
            "prompt": "こんにちは",
            "mode": "layered",
            "max_length": 50,
            "temperature": 0.8,  # This should be converted to temp_min/temp_max
            "top_k": 50,
            "top_p": 0.9,
            "pretrain": False
        }
    }

    print(f"Request: {event['input']}")

    try:
        result = handler(event)
        print(f"\n✅ Success!")
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'error':
            print(f"❌ Error: {result.get('error')}")
            return False
        else:
            print(f"Generated text available: {bool(result.get('generated_text'))}")
            return True
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_brain_mode():
    """Test brain mode with temperature parameter"""
    print("\n🧪 Testing Brain Mode")
    print("=" * 60)

    event = {
        "input": {
            "action": "generate",
            "prompt": "こんにちは",
            "mode": "brain",
            "max_length": 50,
            "temperature": 0.8,  # This should be converted to temperature_min/temperature_max
            "pretrain": False
        }
    }

    print(f"Request: {event['input']}")

    try:
        result = handler(event)
        print(f"\n✅ Success!")
        print(f"Status: {result.get('status')}")
        if result.get('status') == 'error':
            print(f"❌ Error: {result.get('error')}")
            return False
        else:
            print(f"Generated text available: {bool(result.get('generated_text'))}")
            return True
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_check():
    """Test health check endpoint"""
    print("\n🧪 Testing Health Check")
    print("=" * 60)

    event = {
        "input": {
            "action": "health"
        }
    }

    print(f"Request: {event['input']}")

    try:
        result = handler(event)
        print(f"\n✅ Success!")
        print(f"Status: {result.get('status')}")
        print(f"Layered available: {result.get('layered_available')}")
        print(f"Brain available: {result.get('brain_available')}")
        print(f"CUDA available: {result.get('cuda_available')}")
        return True
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 NeuroQ Handler Temperature Fix Verification")
    print("=" * 60)

    results = []

    # Test health check
    results.append(("Health Check", test_health_check()))

    # Test layered mode
    results.append(("Layered Mode", test_layered_mode()))

    # Test brain mode
    results.append(("Brain Mode", test_brain_mode()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! The temperature parameter fix is working correctly.")
        print("\n⚠️  Next Steps:")
        print("   1. Rebuild your RunPod Docker image with the updated code")
        print("   2. Deploy the new image to RunPod")
        print("   3. The 'temperature' parameter will be correctly mapped to:")
        print("      - Layered mode: temp_min and temp_max")
        print("      - Brain mode: temperature_min and temperature_max")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    sys.exit(0 if all_passed else 1)
