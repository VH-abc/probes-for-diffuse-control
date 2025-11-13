#!/usr/bin/env python3
"""
Quick test to verify that the updated save_generations_to_cache function
works correctly and saves both NumPy and JSON files.
"""

import os
import json
import numpy as np
from lib.generation import save_generations_to_cache, load_generations_from_cache

def test_json_caching():
    """Test that both NumPy and JSON files are created."""
    
    # Create test data
    test_full_texts = [
        "Question: What is 2+2? Answer: 4",
        "Question: What is the capital of France? Answer: Paris",
        "Question: Who wrote Hamlet? Answer: William Shakespeare"
    ]
    
    test_completions = [
        " Answer: 4",
        " Answer: Paris",
        " Answer: William Shakespeare"
    ]
    
    test_metadata = {
        "model_name": "test-model",
        "temperature": 1.0,
        "max_new_tokens": 100,
        "num_samples": 3,
        "timestamp": "2025-11-13 test"
    }
    
    # Create test cache path
    test_cache_dir = "experiments/test-model/generations/test"
    os.makedirs(test_cache_dir, exist_ok=True)
    test_cache_path = os.path.join(test_cache_dir, "n3_t1.0_maxtok100")
    
    print("Testing JSON caching functionality...")
    print(f"Cache path: {test_cache_path}\n")
    
    # Save to cache
    print("1. Saving generations...")
    save_generations_to_cache(test_cache_path, test_full_texts, test_completions, test_metadata)
    
    # Check that all files exist
    print("\n2. Checking files exist...")
    expected_files = [
        f"{test_cache_path}_full_texts.npy",
        f"{test_cache_path}_full_texts.json",
        f"{test_cache_path}_completions.npy",
        f"{test_cache_path}_completions.json",
        f"{test_cache_path}_metadata.json"
    ]
    
    all_exist = True
    for filepath in expected_files:
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n✗ Test FAILED: Not all files were created")
        return False
    
    # Load from cache (should use NumPy files)
    print("\n3. Loading from cache (NumPy)...")
    loaded_data = load_generations_from_cache(test_cache_path)
    
    if loaded_data is None:
        print("✗ Test FAILED: Could not load from cache")
        return False
    
    loaded_full_texts, loaded_completions, loaded_metadata = loaded_data
    
    # Verify data matches
    print("\n4. Verifying data integrity...")
    
    if loaded_full_texts != test_full_texts:
        print("✗ Test FAILED: Full texts don't match")
        return False
    print("  ✓ Full texts match")
    
    if loaded_completions != test_completions:
        print("✗ Test FAILED: Completions don't match")
        return False
    print("  ✓ Completions match")
    
    if loaded_metadata != test_metadata:
        print("✗ Test FAILED: Metadata doesn't match")
        return False
    print("  ✓ Metadata matches")
    
    # Verify JSON files are readable
    print("\n5. Verifying JSON files are valid...")
    
    with open(f"{test_cache_path}_full_texts.json", 'r') as f:
        json_full_texts = json.load(f)
    
    if json_full_texts != test_full_texts:
        print("✗ Test FAILED: JSON full texts don't match")
        return False
    print("  ✓ JSON full texts are valid and match")
    
    with open(f"{test_cache_path}_completions.json", 'r') as f:
        json_completions = json.load(f)
    
    if json_completions != test_completions:
        print("✗ Test FAILED: JSON completions don't match")
        return False
    print("  ✓ JSON completions are valid and match")
    
    # Clean up test files
    print("\n6. Cleaning up test files...")
    for filepath in expected_files:
        os.remove(filepath)
        print(f"  ✓ Removed {filepath}")
    
    # Remove test directory
    os.rmdir(test_cache_dir)
    os.rmdir("experiments/test-model/generations")
    os.rmdir("experiments/test-model")
    print("  ✓ Removed test directories")
    
    print("\n" + "="*60)
    print("✓ All tests PASSED!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_json_caching()
    exit(0 if success else 1)

