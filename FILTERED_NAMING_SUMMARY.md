# Filtered/Unfiltered Naming Convention

## Summary
All result filenames and cached activation filenames now include a `_filtered` or `_unfiltered` suffix to clearly indicate whether the run used filtered (reliable questions only) or unfiltered (all questions) data.

## Changes Made

### Code Updates

#### 1. `probe_analysis.py`
- Updated `load_cached_activations()` to include filter status in filename prefix
- Updated `run_probe_analysis()` to include filter status in result filenames

**New filename format:**
```python
fname = f"layer{layer}_pos-{token_position}_n{num_examples}_{filter_suffix}"
# Example: layer12_pos-last_n2000_filtered
```

#### 2. `cache_activations.py`
- Updated `cache_mmlu_activations()` to include filter status in output filenames

**New filename format:**
```python
output_prefix = f"mmlu_layer{layer_idx:02d}_pos-{token_position}_n{num_examples}_{filter_suffix}"
# Example: mmlu_layer12_pos-last_n2000_filtered
```

### File Naming Convention

#### Cached Activations
**Old:**
```
experiments/gemma-3-12b/cached_activations/50-50/50-50_filtered/
    mmlu_layer12_pos-last_n2000_activations.npy
    mmlu_layer12_pos-last_n2000_labels.npy
    ...
```

**New:**
```
experiments/gemma-3-12b/cached_activations/50-50/50-50_filtered/
    mmlu_layer12_pos-last_n2000_filtered_activations.npy
    mmlu_layer12_pos-last_n2000_filtered_labels.npy
    ...
```

#### Result Files
**Old:**
```
experiments/gemma-3-12b/results/
    roc_layer12_pos-last_n2000.png
    auroc_vs_n_layer12_pos-last_n2000.png
    pca_layer12_pos-last_n2000.png
    ...
```

**New:**
```
experiments/gemma-3-12b/results/
    roc_layer12_pos-last_n2000_filtered.png
    auroc_vs_n_layer12_pos-last_n2000_filtered.png
    pca_layer12_pos-last_n2000_filtered.png
    ...
```

## Benefits

1. **Clear at a glance** - Can immediately see if a result used filtered or unfiltered data
2. **No confusion** - Different runs with different filtering won't overwrite each other
3. **Better organization** - All files related to filtered/unfiltered runs are clearly labeled
4. **Comparison-friendly** - Easy to compare filtered vs unfiltered results side-by-side

## Examples

### Filtered Run (using reliable questions only)
```bash
python cache_activations.py --prompt "50-50" --layer 12 --filtered
```
**Creates:**
- `experiments/gemma-3-12b/cached_activations/50-50/50-50_filtered/mmlu_layer12_pos-last_n2000_filtered_activations.npy`

**Then running analysis:**
```bash
python probe_analysis.py --prompt "50-50" --layer 12 --filtered
```
**Creates:**
- `experiments/gemma-3-12b/results/roc_layer12_pos-last_n2000_filtered.png`
- `experiments/gemma-3-12b/results/auroc_vs_n_layer12_pos-last_n2000_filtered.png`
- etc.

### Unfiltered Run (using all questions)
```bash
python cache_activations.py --prompt "50-50" --layer 12
```
**Creates:**
- `experiments/gemma-3-12b/cached_activations/50-50/mmlu_layer12_pos-last_n2000_unfiltered_activations.npy`

**Then running analysis:**
```bash
python probe_analysis.py --prompt "50-50" --layer 12
```
**Creates:**
- `experiments/gemma-3-12b/results/roc_layer12_pos-last_n2000_unfiltered.png`
- `experiments/gemma-3-12b/results/auroc_vs_n_layer12_pos-last_n2000_unfiltered.png`
- etc.

## Migration

All existing files have been renamed to include the appropriate suffix:
- Cached activations in `50-50_filtered/` directories → `_filtered` suffix added
- Result files from filtered runs → `_filtered` suffix added

