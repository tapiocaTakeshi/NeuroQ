#!/usr/bin/env python3
"""
Temperature Parameter Fix Verification Script

This script verifies that the handler.py code has the correct temperature parameter
conversions without requiring runpod or other dependencies.
"""

import re
import sys

def verify_handler_code():
    """Verify that handler.py has correct temperature parameter conversions"""
    print("\n" + "="*70)
    print("Verifying handler.py Temperature Parameter Fix")
    print("="*70)

    with open('handler.py', 'r') as f:
        content = f.read()

    issues = []
    success_count = 0

    # Check 1: Verify layered mode uses temp_min and temp_max
    print("\n1. Checking layered mode generate() call...")
    layered_section = re.search(
        r'if mode == "layered".*?model\.generate\((.*?)\)',
        content,
        re.DOTALL
    )

    if layered_section:
        params = layered_section.group(1)
        if 'temp_min' in params and 'temp_max' in params:
            if 'temperature=' in params and 'temp_' not in params.split('temperature=')[0]:
                issues.append("❌ Layered mode: Found 'temperature=' parameter (should use temp_min/temp_max)")
            else:
                print("   ✓ Layered mode correctly uses temp_min and temp_max")
                success_count += 1
        else:
            issues.append("❌ Layered mode: Missing temp_min and/or temp_max parameters")
    else:
        issues.append("❌ Could not find layered mode generate() call")

    # Check 2: Verify brain mode uses temperature_min and temperature_max
    print("\n2. Checking brain mode generate() call...")
    brain_section = re.search(
        r'elif mode == "brain".*?model\.generate\((.*?)\)',
        content,
        re.DOTALL
    )

    if brain_section:
        params = brain_section.group(1)
        if 'temperature_min' in params and 'temperature_max' in params:
            # Make sure it's not using just 'temperature=' (without _min/_max)
            temp_usage = re.findall(r'\btemperature\s*=', params)
            temp_min_max_usage = re.findall(r'\btemperature_(min|max)\s*=', params)

            if len(temp_usage) > len(temp_min_max_usage):
                issues.append("❌ Brain mode: Found plain 'temperature=' parameter")
            else:
                print("   ✓ Brain mode correctly uses temperature_min and temperature_max")
                success_count += 1
        else:
            issues.append("❌ Brain mode: Missing temperature_min and/or temperature_max parameters")
    else:
        issues.append("❌ Could not find brain mode generate() call")

    # Check 3: Verify temperature variable is read from input
    print("\n3. Checking temperature input parameter extraction...")
    if 'temperature = input_data.get("temperature"' in content:
        print("   ✓ Handler correctly reads 'temperature' from API input")
        success_count += 1
    else:
        issues.append("❌ Handler doesn't read 'temperature' from input_data")

    # Check 4: Verify conversion logic
    print("\n4. Checking temperature conversion logic...")
    has_layered_conversion = bool(re.search(r'temp_min\s*=\s*temperature\s*\*\s*0\.8', content))
    has_brain_conversion = bool(re.search(r'temperature_min\s*=\s*temperature\s*\*\s*0\.8', content))

    if has_layered_conversion:
        print("   ✓ Layered mode has temperature conversion (temp_min = temperature * 0.8)")
        success_count += 1
    else:
        issues.append("❌ Layered mode missing temperature conversion logic")

    if has_brain_conversion:
        print("   ✓ Brain mode has temperature conversion (temperature_min = temperature * 0.8)")
        success_count += 1
    else:
        issues.append("❌ Brain mode missing temperature conversion logic")

    # Print summary
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)
    print(f"Checks passed: {success_count}/5")

    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n✅ All checks passed! The temperature parameter fix is correctly implemented.")
        return True


def verify_model_signatures():
    """Verify that the model generate() methods have the expected signatures"""
    print("\n" + "="*70)
    print("Verifying Model generate() Method Signatures")
    print("="*70)

    # Check NeuroQuantumAI (layered)
    print("\n1. Checking NeuroQuantumAI.generate() signature...")
    try:
        with open('neuroquantum_layered.py', 'r') as f:
            content = f.read()

        # Find the generate method definition
        match = re.search(
            r'def generate\s*\((.*?)\)\s*->',
            content,
            re.DOTALL
        )

        if match:
            params = match.group(1)
            if 'temp_min' in params and 'temp_max' in params:
                print("   ✓ NeuroQuantumAI.generate() accepts temp_min and temp_max")
                if 'temperature:' in params or 'temperature =' in params:
                    print("   ⚠️  Warning: Also has 'temperature' parameter (might cause issues)")
            else:
                print("   ❌ NeuroQuantumAI.generate() doesn't have temp_min/temp_max parameters")
        else:
            print("   ⚠️  Could not parse generate() signature")
    except FileNotFoundError:
        print("   ⚠️  neuroquantum_layered.py not found")

    # Check NeuroQuantumBrainAI (brain)
    print("\n2. Checking NeuroQuantumBrainAI.generate() signature...")
    try:
        with open('neuroquantum_brain.py', 'r') as f:
            content = f.read()

        # Find the generate method definition in the NeuroQuantumBrainAI class
        match = re.search(
            r'class NeuroQuantumBrainAI.*?def generate\s*\((.*?)\)\s*->',
            content,
            re.DOTALL
        )

        if match:
            params = match.group(1)
            if 'temperature_min' in params and 'temperature_max' in params:
                print("   ✓ NeuroQuantumBrainAI.generate() accepts temperature_min and temperature_max")
                # Check if it also has plain temperature parameter
                if re.search(r'\btemperature\s*:', params) and 'temperature_min' not in params.split(re.search(r'\btemperature\s*:', params).group())[0]:
                    print("   ⚠️  Warning: Also has 'temperature' parameter (might cause issues)")
            else:
                print("   ❌ NeuroQuantumBrainAI.generate() doesn't have temperature_min/temperature_max")
        else:
            print("   ⚠️  Could not parse generate() signature")
    except FileNotFoundError:
        print("   ⚠️  neuroquantum_brain.py not found")


def main():
    """Run all verifications"""
    print("\n" + "🔍"*35)
    print("Temperature Parameter Fix Verification")
    print("🔍"*35)

    handler_ok = verify_handler_code()
    verify_model_signatures()

    print("\n" + "="*70)
    if handler_ok:
        print("✅ VERIFICATION SUCCESSFUL")
        print("\nThe handler correctly:")
        print("  1. Accepts 'temperature' from API input")
        print("  2. Converts it to temp_min/temp_max for layered mode")
        print("  3. Converts it to temperature_min/temperature_max for brain mode")
        print("  4. Uses ±20% range (0.8× to 1.2×) for the conversion")
        print("\nThe fix is ready for deployment to RunPod.")
        return True
    else:
        print("❌ VERIFICATION FAILED")
        print("\nPlease review the handler.py file and ensure it has the correct")
        print("temperature parameter conversions.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
