# Organized Results Directory Structure

## Summary
Updated the codebase to save results in organized subdirectories based on the number of examples (`n`) and filter status (`filtered`/`unfiltered`). This makes it much easier to manage and compare different experimental configurations.

## New Directory Structure

### Before (all results in one directory):
```
experiments/gemma-3-12b/results/
├── roc_layer12_pos-last_n200_filtered.png
├── roc_layer12_pos-last_n2000_filtered.png
├── roc_layer12_pos-last_n200_unfiltered.png
├── auroc_layer12_pos-last_n200_filtered.json
├── auroc_layer12_pos-last_n2000_filtered.json
└── ... (all mixed together)
```

### After (organized by n and filter status):
```
experiments/gemma-3-12b/results/
├── n200_filtered/
│   ├── roc_layer12_pos-last_n200_filtered.png
│   ├── auroc_layer12_pos-last_n200_filtered.json
│   ├── pca_layer12_pos-last_n200_filtered.png
│   └── layer_sweep_summary_filtered.txt
├── n200_unfiltered/
│   ├── roc_layer12_pos-last_n200_unfiltered.png
│   ├── auroc_layer12_pos-last_n200_unfiltered.json
│   └── layer_sweep_summary_unfiltered.txt
├── n2000_filtered/
│   ├── roc_layer12_pos-last_n2000_filtered.png
│   ├── auroc_layer12_pos-last_n2000_filtered.json
│   ├── pca_layer12_pos-last_n2000_filtered.png
│   └── layer_sweep_summary_filtered.txt
└── n2000_unfiltered/
    ├── roc_layer12_pos-last_n2000_unfiltered.png
    └── ...
```

## Changes Made

### 1. Added `get_results_dir()` function to `config.py`

```python
def get_results_dir(num_examples: int = None, filter_reliable: bool = False) -> str:
    """
    Get results directory path organized by number of examples and filter status.
    Returns: Path like "results/n200_filtered"
    """
    num_examples = num_examples or DEFAULT_NUM_EXAMPLES
    filter_suffix = "filtered" if filter_reliable else "unfiltered"
    return f"{RESULTS_DIR}/n{num_examples}_{filter_suffix}"
```

### 2. Updated `probe_analysis.py`

- Calls `config.get_results_dir()` to get organized subdirectory
- All result files (ROC curves, PCA plots, AUROC JSON files, etc.) now saved to appropriate subdirectory
- Prints the results directory path for user reference

### 3. Updated `sweep_layers.py`

- `generate_layer_comparison()` looks for AUROC files in correct subdirectory
- `generate_position_comparison()` looks for AUROC files in correct subdirectory
- Summary files saved to appropriate subdirectory
- Final output shows correct results directory

## Benefits

### 1. **Clear Organization**
- Easy to find results for specific configurations
- No mixing of different experimental setups

### 2. **Easy Comparison**
- Compare n=200 vs n=2000 by looking at different directories
- Compare filtered vs unfiltered by looking at different directories

### 3. **No Name Conflicts**
- Different configurations can't overwrite each other's results
- Safe to run multiple experiments in parallel

### 4. **Better Workflow**
- Can delete results for one configuration without affecting others
- Easy to archive or share results for specific configurations

## Usage Examples

### Running with Default (n=200, filtered)
```bash
python probe_analysis.py --prompt "50-50" --layer 12
# Results saved to: experiments/gemma-3-12b/results/n200_filtered/
```

### Running with n=2000, filtered
```bash
python probe_analysis.py --prompt "50-50" --layer 12 --num-examples 2000
# Results saved to: experiments/gemma-3-12b/results/n2000_filtered/
```

### Running with n=200, unfiltered
```bash
python probe_analysis.py --prompt "50-50" --layer 12 --no-filter
# Results saved to: experiments/gemma-3-12b/results/n200_unfiltered/
```

### Layer Sweep
```bash
python sweep_layers.py --layers 10 12 13 14
# Results saved to: experiments/gemma-3-12b/results/n200_filtered/
# Summary: experiments/gemma-3-12b/results/n200_filtered/layer_sweep_summary_filtered.txt
```

## Result File Locations

For a given configuration with `n=200` and `filtered=True`:

**All files go to:** `experiments/gemma-3-12b/results/n200_filtered/`

Including:
- `roc_layer{N}_pos-{position}_n200_filtered.png`
- `pca_layer{N}_pos-{position}_n200_filtered.png`
- `auroc_layer{N}_pos-{position}_n200_filtered.json`
- `scores_layer{N}_pos-{position}_n200_filtered.png`
- `auroc_vs_n_layer{N}_pos-{position}_n200_filtered.png`
- `corruption_layer{N}_pos-{position}_n200_filtered.png`
- `training_analysis_layer{N}_pos-{position}_n200_filtered.txt`
- `layer_sweep_summary_filtered.txt` (from sweep)
- `position_sweep_layer{N}_summary_filtered.txt` (from position sweep)

## Migration

Existing results in the flat `results/` directory are not affected. New runs will create and use the organized subdirectories. If you want to organize existing results, you can manually move them to the appropriate subdirectories.

## Directory Naming Pattern

Format: `n{num_examples}_{filter_status}`

Examples:
- `n200_filtered` - 200 examples, filtered (reliable questions only)
- `n200_unfiltered` - 200 examples, unfiltered (all questions)
- `n2000_filtered` - 2000 examples, filtered
- `n2000_unfiltered` - 2000 examples, unfiltered
- `n1374_filtered` - 1374 examples (all reliable questions), filtered

