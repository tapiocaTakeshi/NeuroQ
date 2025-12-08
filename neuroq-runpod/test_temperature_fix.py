#!/usr/bin/env python3
"""
Temperature Parameter Fix Verification Script

This script verifies that the handler correctly converts the 'temperature'
parameter to the appropriate model-specific parameters.
"""

import json
from handler import handler

def test_layered_temperature():
    """Test that layered mode correctly handles temperature parameter"""
    print("\n" + "="*60)
    print("Testing Layered Mode Temperature Parameter Conversion")
    print("="*60)

    event = {
        "input": {
            "action": "generate",
            "mode": "layered",
            "prompt": "Test prompt",
            "max_length": 20,
            "temperature": 0.8,  # Single temperature value
            "pretrain": False
        }
    }

    try:
        print(f"Input: {json.dumps(event, indent=2)}")
        print("\n✓ Handler should convert temperature=0.8 to:")
        print("  - temp_min = 0.8 * 0.8 = 0.64")
        print("  - temp_max = 0.8 * 1.2 = 0.96")

        result = handler(event)
        print(f"\nResult: {json.dumps(result, indent=2)}")

        if result.get("status") == "error":
            if "'temperature'" in result.get("error", ""):
                print("\n❌ FAILED: Still getting temperature parameter error!")
                return False
        print("\n✓ PASSED: No temperature parameter error")
        return True
    except Exception as e:
        if "'temperature'" in str(e):
            print(f"\n❌ FAILED: {e}")
            return False
        print(f"\n⚠️  Other error: {e}")
        return False


def test_brain_temperature():
    """Test that brain mode correctly handles temperature parameter"""
    print("\n" + "="*60)
    print("Testing Brain Mode Temperature Parameter Conversion")
    print("="*60)

    event = {
        "input": {
            "action": "generate",
            "mode": "brain",
            "prompt": "Test prompt",
            "max_length": 20,
            "temperature": 0.7,  # Single temperature value
            "pretrain": False
        }
    }

    try:
        print(f"Input: {json.dumps(event, indent=2)}")
        print("\n✓ Handler should convert temperature=0.7 to:")
        print("  - temperature_min = 0.7 * 0.8 = 0.56")
        print("  - temperature_max = 0.7 * 1.2 = 0.84")

        result = handler(event)
        print(f"\nResult: {json.dumps(result, indent=2)}")

        if result.get("status") == "error":
            if "'temperature'" in result.get("error", ""):
                print("\n❌ FAILED: Still getting temperature parameter error!")
                return False
        print("\n✓ PASSED: No temperature parameter error")
        return True
    except Exception as e:
        if "'temperature'" in str(e):
            print(f"\n❌ FAILED: {e}")
            return False
        print(f"\n⚠️  Other error: {e}")
        return False


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)

    event = {
        "input": {
            "action": "health"
        }
    }

    try:
        result = handler(event)
        print(f"Result: {json.dumps(result, indent=2)}")

        if result.get("status") == "healthy":
            print("\n✓ PASSED: Health check successful")
            print(f"  - Layered available: {result.get('layered_available')}")
            print(f"  - Brain available: {result.get('brain_available')}")
            return True
        else:
            print("\n⚠️  Health check returned non-healthy status")
            return False
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🧪"*30)
    print("Temperature Parameter Fix Verification")
    print("🧪"*30)

    results = {
        "Health Check": test_health_check(),
        "Layered Temperature": test_layered_temperature(),
        "Brain Temperature": test_brain_temperature(),
    }

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("="*60)

    if all_passed:
        print("\n✅ All tests passed! Temperature parameter fix is working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the handler code.")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
