# Sweep Layers Cache Check Enhancement

## Summary
Updated `sweep_layers.py` to automatically detect and skip recomputing activations that already exist in the cache. This significantly speeds up sweep operations when running multiple experiments.

## Changes Made

### New Function: `check_cache_exists()`

Added a helper function that checks if cached activations exist for given parameters:

```python
def check_cache_exists(prompt_name: str, layer: int, token_position: str, 
                       num_examples: int, filter_reliable: bool = False) -> bool:
    """
    Check if cached activations already exist for the given parameters.
    Returns True if cache exists and is complete, False otherwise.
    """
```

The function checks for essential files:
- `{prefix}_activations.npy`
- `{prefix}_labels.npy`
- `{prefix}_metadata.json`

### Updated Sweep Functions

Both `sweep_layers()` and `sweep_positions()` now:
1. **Check cache before computing** - Automatically detect if activations exist
2. **Skip recomputation** - Only call `cache_mmlu_activations()` if cache is missing
3. **Clear logging** - Show whether using existing cache or computing new activations

### Updated Comparison Functions

Fixed `generate_layer_comparison()` and `generate_position_comparison()` to:
1. Use the `_filtered` or `_unfiltered` suffix when looking for result files
2. Include filtered status in output filenames
3. Add filtered status to report headers

## Behavior Changes

### Before:
```bash
python sweep_layers.py --layers 10 12 13 14
# Always recomputed activations, even if they existed
```

### After:
```bash
python sweep_layers.py --layers 10 12 13 14
# Checks each layer's cache first
# Only computes missing activations
# Much faster for partial reruns!
```

## Output Messages

**Cache exists:**
```
✓ Cache already exists for layer 12, skipping activation caching
```

**Cache doesn't exist:**
```
Caching activations for layer 12...
[... activation caching process ...]
```

**Skip cache flag:**
```
⏩ Skipping cache (--skip-cache flag)
```

## Examples

### Layer Sweep with Auto-Detection
```bash
# First run - computes all activations
python sweep_layers.py --prompt "50-50" --layers 10 12 13 14 --filtered

# Second run - skips existing, only computes new layers
python sweep_layers.py --prompt "50-50" --layers 10 12 13 14 16 --filtered
# Layers 10, 12, 13, 14 skipped (already cached)
# Only layer 16 computed
```

### Position Sweep with Auto-Detection
```bash
# First run
python sweep_layers.py --mode positions --layer 12 --positions last first --filtered

# Second run with additional position
python sweep_layers.py --mode positions --layer 12 --positions last first middle --filtered
# 'last' and 'first' skipped (already cached)
# Only 'middle' computed
```

### Force Recompute (if needed)
```bash
# Use --skip-cache to bypass all caching (use existing only)
python sweep_layers.py --layers 10 12 --skip-cache --skip-analysis
```

## Benefits

1. **⚡ Faster reruns** - No need to recompute existing activations
2. **💰 Cost savings** - Avoids unnecessary VLLM server usage
3. **🔄 Iterative workflow** - Easy to add new layers/positions to existing sweeps
4. **🛡️ Safe** - Only skips if essential files exist
5. **📝 Clear logging** - Always know what's being computed vs reused

## Summary Files

Summary files now include filtered status in filename:
- `layer_sweep_summary_filtered.txt` (for --filtered runs)
- `layer_sweep_summary_unfiltered.txt` (for normal runs)
- `position_sweep_layer{N}_summary_filtered.txt`
- `position_sweep_layer{N}_summary_unfiltered.txt`

