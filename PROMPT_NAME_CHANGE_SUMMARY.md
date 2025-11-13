# Prompt Name Change: "50/50" → "50-50"

## Summary
All references to the prompt name "50/50" have been changed to "50-50" throughout the codebase to avoid issues with forward slashes in directory and file names.

## Files Updated

### Python Source Files
- `probe_analysis.py` - Updated docstrings and argument help
- `prompts.py` - Changed prompt key from "50/50" to "50-50"
- `lib/generation.py` - Updated docstrings
- `sweep_layers.py` - Updated default values and argument help
- `cache_activations.py` - Updated docstrings and argument help
- `lib/data.py` - Updated docstrings
- `run_position_sweep.py` - Updated default values and help text

### Documentation Files
- `GENERATION_CACHING_SUMMARY.md`
- `DECODE_GENERATIONS.md`
- `GENERATION_CACHING.md`
- `SETUP_SUMMARY.md`
- `TROUBLESHOOTING_OOM.md`
- `GPU_ALLOCATION.md`

## Directories Renamed
- `experiments/gemma-3-12b/cached_activations/50/` → `experiments/gemma-3-12b/cached_activations/50-50/`
- `experiments/gemma-3-12b/cached_activations/50/50_filtered/` → `experiments/gemma-3-12b/cached_activations/50-50/50-50_filtered/`
- `experiments/gemma-3-12b/generations/50/` → `experiments/gemma-3-12b/generations/50-50/`
- `experiments/gemma-3-12b/generations/50/50_filtered/` → `experiments/gemma-3-12b/generations/50-50/50-50_filtered/`

## Usage Examples (Updated)

### Old Usage:
```bash
python cache_activations.py --prompt "50/50" --layer 10
python sweep_layers.py --prompt "50/50" --layers 10 11 12 13 14
```

### New Usage:
```bash
python cache_activations.py --prompt "50-50" --layer 10
python sweep_layers.py --prompt "50-50" --layers 10 11 12 13 14
```

## Note
The only remaining "50/50" reference is in `math_vs_nonmath_experiment.py` line 267, which refers to a train/test split ratio (not the prompt name), and was intentionally left unchanged.

## Verification
- All Python scripts now use "50-50" as the prompt name
- Directory structure has been updated to use "50-50"
- All cached activations and generations are preserved
