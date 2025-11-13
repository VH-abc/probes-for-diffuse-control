#!/usr/bin/env python3
"""
Decode cached numpy generation files to human-readable JSON format.

This script finds all cached generation files (*.npy) and converts them to JSON
for easy human inspection. The JSON files are saved alongside the numpy files
with the same naming convention.

Usage:
    python decode_generations.py                    # Decode all cached generations
    python decode_generations.py --model gemma-3-12b  # Decode specific model
    python decode_generations.py --limit 10         # Only save first 10 items per file
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from typing import Optional


def decode_generation_file(npy_path: str, limit: Optional[int] = None) -> bool:
    """
    Decode a single numpy generation file to JSON.
    
    Args:
        npy_path: Path to .npy file
        limit: Optional limit on number of items to save (for large files)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load numpy array
        data = np.load(npy_path, allow_pickle=True)
        
        # Convert to list
        data_list = data.tolist()
        
        # Apply limit if specified
        if limit is not None and len(data_list) > limit:
            print(f"    (limiting to first {limit} items)")
            data_list = data_list[:limit]
        
        # Determine output path
        json_path = npy_path.replace('.npy', '.json')
        
        # Save as JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Decoded {len(data_list)} items to {json_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error decoding {npy_path}: {e}")
        return False


def find_generation_files(base_dir: str = "experiments", model: Optional[str] = None):
    """
    Find all cached generation numpy files.
    
    Args:
        base_dir: Base directory to search
        model: Optional model name to filter by
    
    Returns:
        List of paths to generation .npy files
    """
    search_path = Path(base_dir)
    if model:
        search_path = search_path / model / "generations"
    
    if not search_path.exists():
        print(f"Warning: Path {search_path} does not exist")
        return []
    
    # Find all .npy files in generations directories
    npy_files = list(search_path.glob("**/*.npy"))
    
    # Filter to only generation files (full_texts and completions)
    generation_files = [
        str(f) for f in npy_files 
        if 'full_texts' in f.name or 'completions' in f.name
    ]
    
    return sorted(generation_files)


def decode_all_generations(
    base_dir: str = "experiments", 
    model: Optional[str] = None,
    limit: Optional[int] = None,
    dry_run: bool = False
):
    """
    Decode all cached generation files to JSON.
    
    Args:
        base_dir: Base directory to search
        model: Optional model name to filter by
        limit: Optional limit on items per file
        dry_run: If True, only print what would be done
    """
    print(f"\n{'='*60}")
    print(f"Decoding Cached Generation Files")
    print(f"{'='*60}")
    if model:
        print(f"Model: {model}")
    else:
        print(f"All models in {base_dir}")
    if limit:
        print(f"Limiting to first {limit} items per file")
    if dry_run:
        print("DRY RUN - no files will be written")
    print(f"{'='*60}\n")
    
    # Find all generation files
    generation_files = find_generation_files(base_dir, model)
    
    if not generation_files:
        print("No generation files found!")
        return
    
    print(f"Found {len(generation_files)} generation files\n")
    
    # Process each file
    success_count = 0
    skip_count = 0
    
    for npy_path in generation_files:
        # Check if JSON already exists
        json_path = npy_path.replace('.npy', '.json')
        
        # Skip metadata files (those are already JSON)
        if 'metadata' in npy_path:
            continue
        
        print(f"Processing: {npy_path}")
        
        if os.path.exists(json_path) and not dry_run:
            print(f"  ⊙ JSON already exists (skipping)")
            skip_count += 1
            continue
        
        if dry_run:
            print(f"  Would decode to: {json_path}")
            success_count += 1
        else:
            if decode_generation_file(npy_path, limit):
                success_count += 1
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  Successfully decoded: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Total processed: {success_count + skip_count}/{len(generation_files)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Decode cached numpy generation files to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='experiments',
        help='Base directory to search for generations (default: experiments)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Filter to specific model (e.g., gemma-3-12b)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of items to save per file (useful for large files)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without writing files'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing JSON files'
    )
    
    args = parser.parse_args()
    
    # If force is specified, temporarily rename existing files
    # (simpler: just delete .json files if --force)
    if args.force and not args.dry_run:
        generation_files = find_generation_files(args.base_dir, args.model)
        for npy_path in generation_files:
            json_path = npy_path.replace('.npy', '.json')
            if os.path.exists(json_path):
                os.remove(json_path)
                print(f"Removed existing: {json_path}")
    
    decode_all_generations(
        base_dir=args.base_dir,
        model=args.model,
        limit=args.limit,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()

