#!/usr/bin/env python3
"""
Migrate existing cached_activations and results to the new config-based structure.

This script helps transition from the old flat directory structure to the new
model-specific directory organization.
"""

import os
import shutil
from config import MODEL_SHORT_NAME, CACHED_ACTIVATIONS_DIR, RESULTS_DIR

def migrate_directories():
    """Migrate old directories to new structure."""
    
    # Old directory paths
    old_cache_dir = "cached_activations"
    old_results_dir = "results"
    
    # Check if old directories exist
    has_old_cache = os.path.exists(old_cache_dir) and os.path.isdir(old_cache_dir)
    has_old_results = os.path.exists(old_results_dir) and os.path.isdir(old_results_dir)
    
    if not has_old_cache and not has_old_results:
        print("No old directories found to migrate.")
        print("The new directory structure is already in place.")
        return
    
    print(f"Migrating to model-specific structure for: {MODEL_SHORT_NAME}")
    print("="*80)
    
    # Create new directories
    os.makedirs(CACHED_ACTIVATIONS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Migrate cached activations
    if has_old_cache:
        files = [f for f in os.listdir(old_cache_dir) if os.path.isfile(os.path.join(old_cache_dir, f))]
        if files:
            print(f"\nMigrating {len(files)} files from {old_cache_dir}/ to {CACHED_ACTIVATIONS_DIR}/")
            for filename in files:
                src = os.path.join(old_cache_dir, filename)
                dst = os.path.join(CACHED_ACTIVATIONS_DIR, filename)
                
                if os.path.exists(dst):
                    print(f"  SKIP (exists): {filename}")
                else:
                    shutil.copy2(src, dst)
                    print(f"  COPY: {filename}")
            
            print(f"\nOld directory preserved at: {old_cache_dir}/")
            print(f"You can safely delete it after verifying the migration.")
        else:
            print(f"\n{old_cache_dir}/ is empty, nothing to migrate.")
    
    # Migrate results
    if has_old_results:
        files = [f for f in os.listdir(old_results_dir) if os.path.isfile(os.path.join(old_results_dir, f))]
        if files:
            print(f"\nMigrating {len(files)} files from {old_results_dir}/ to {RESULTS_DIR}/")
            for filename in files:
                src = os.path.join(old_results_dir, filename)
                dst = os.path.join(RESULTS_DIR, filename)
                
                if os.path.exists(dst):
                    print(f"  SKIP (exists): {filename}")
                else:
                    shutil.copy2(src, dst)
                    print(f"  COPY: {filename}")
            
            print(f"\nOld directory preserved at: {old_results_dir}/")
            print(f"You can safely delete it after verifying the migration.")
        else:
            print(f"\n{old_results_dir}/ is empty, nothing to migrate.")
    
    print("\n" + "="*80)
    print("Migration complete!")
    print(f"New structure:")
    print(f"  Cached activations: {CACHED_ACTIVATIONS_DIR}/")
    print(f"  Results:            {RESULTS_DIR}/")
    print("\nTo clean up old directories:")
    if has_old_cache:
        print(f"  rm -rf {old_cache_dir}/")
    if has_old_results:
        print(f"  rm -rf {old_results_dir}/")

if __name__ == "__main__":
    migrate_directories()

