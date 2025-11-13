#!/usr/bin/env python3
"""
Quick test to verify JSON caching functionality without requiring all dependencies.
"""

import os
import json
import numpy as np

def save_test_generations(cache_path, full_texts, completions, metadata):
    """Test version of save_generations_to_cache."""
    print(f"  💾 Saving test generations to: {cache_path}")
    
    # Save as numpy arrays (more efficient for large datasets)
    np.save(f"{cache_path}_full_texts.npy", np.array(full_texts, dtype=object))
    np.save(f"{cache_path}_completions.npy", np.array(completions, dtype=object))
    
    # Save as JSON for human readability
    with open(f"{cache_path}_full_texts.json", 'w', encoding='utf-8') as f:
        json.dump(full_texts, f, indent=2, ensure_ascii=False)
    with open(f"{cache_path}_completions.json", 'w', encoding='utf-8') as f:
        json.dump(completions, f, indent=2, ensure_ascii=False)
    
    # Save metadata as JSON
    with open(f"{cache_path}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Cached {len(full_texts)} generations (numpy + JSON)")

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
    
    print("\n" + "="*60)
    print("Testing JSON Caching Functionality")
    print("="*60)
    print(f"Cache path: {test_cache_path}\n")
    
    # Save to cache
    print("1. Saving generations...")
    save_test_generations(test_cache_path, test_full_texts, test_completions, test_metadata)
    
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
        size = os.path.getsize(filepath) if exists else 0
        print(f"  {status} {os.path.basename(filepath):30} ({size:,} bytes)")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n✗ Test FAILED: Not all files were created")
        return False
    
    # Load from NumPy files
    print("\n3. Loading from NumPy cache...")
    loaded_full_texts = np.load(f"{test_cache_path}_full_texts.npy", allow_pickle=True).tolist()
    loaded_completions = np.load(f"{test_cache_path}_completions.npy", allow_pickle=True).tolist()
    
    with open(f"{test_cache_path}_metadata.json", 'r') as f:
        loaded_metadata = json.load(f)
    
    # Verify data matches
    print("\n4. Verifying NumPy data integrity...")
    
    if loaded_full_texts != test_full_texts:
        print("  ✗ Full texts don't match")
        return False
    print("  ✓ Full texts match")
    
    if loaded_completions != test_completions:
        print("  ✗ Completions don't match")
        return False
    print("  ✓ Completions match")
    
    if loaded_metadata != test_metadata:
        print("  ✗ Metadata doesn't match")
        return False
    print("  ✓ Metadata matches")
    
    # Verify JSON files are readable
    print("\n5. Verifying JSON files are valid...")
    
    with open(f"{test_cache_path}_full_texts.json", 'r', encoding='utf-8') as f:
        json_full_texts = json.load(f)
    
    if json_full_texts != test_full_texts:
        print("  ✗ JSON full texts don't match")
        return False
    print("  ✓ JSON full texts are valid and match")
    
    with open(f"{test_cache_path}_completions.json", 'r', encoding='utf-8') as f:
        json_completions = json.load(f)
    
    if json_completions != test_completions:
        print("  ✗ JSON completions don't match")
        return False
    print("  ✓ JSON completions are valid and match")
    
    # Show a sample of JSON content
    print("\n6. Sample JSON content:")
    print(f"  Full text [0]: {json_full_texts[0][:60]}...")
    print(f"  Completion [0]: {json_completions[0]}")
    
    # Clean up test files
    print("\n7. Cleaning up test files...")
    for filepath in expected_files:
        os.remove(filepath)
        print(f"  ✓ Removed {os.path.basename(filepath)}")
    
    # Remove test directory
    os.rmdir(test_cache_dir)
    os.rmdir("experiments/test-model/generations")
    os.rmdir("experiments/test-model")
    print("  ✓ Removed test directories")
    
    print("\n" + "="*60)
    print("✓ All tests PASSED!")
    print("="*60)
    print("\nConclusion:")
    print("  - Both NumPy and JSON files are created correctly")
    print("  - Data integrity is maintained")
    print("  - JSON files are human-readable")
    print("  - NumPy files are used for loading (efficient)")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    success = test_json_caching()
    exit(0 if success else 1)
