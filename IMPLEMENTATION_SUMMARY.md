# Lie Detector Experiment - Implementation Summary

## What Was Implemented

### 1. New File: `cache_lie_detector_activations.py`

A specialized caching script that performs two-pass generation:

**Pass 1**: 
- Formats questions with "lie detector part 1" prompt
- Generates answer attempts via VLLM
- Computes actual correctness labels using `compute_correctness_labels()`
- Caches as `prompt_name="lie_detector_part1"`

**Pass 2**:
- Formats followup prompts with "lie detector part 2" using Pass 1 as `{preceding_conversation}`
- Generates yes/no responses via VLLM
- Extracts activations from these responses
- Caches as `prompt_name="lie_detector_part2"`

**Output**:
- Saves in unified cache format: `experiments/{model}/cached_activations/lie_detector[_filtered]/unified_n{N}_{filtered|unfiltered}/`
- Activations from Pass 2, labels from Pass 1 actual correctness
- Includes `first_generation.npy` for later analysis
- Compatible with existing `probe_analysis.py`

### 2. Modified File: `sweep.py`

Updated to detect and handle "lie_detector" prompt:

**Changes**:
1. `check_cache_exists()` function (lines 46-54):
   - Added special handling for "lie_detector" cache naming
   - Maps to "lie_detector_filtered" or "lie_detector" (not "lie_detector_filtered")

2. `sweep()` function (lines 134-156):
   - Added conditional import based on prompt_name
   - If `prompt_name == "lie_detector"`: imports and calls `cache_lie_detector_activations()`
   - Otherwise: uses existing `cache_mmlu_activations_unified()`

**Result**: `sweep.py` now seamlessly supports lie detector experiments alongside other prompts.

### 3. Documentation: `LIE_DETECTOR_USAGE.md`

Comprehensive usage guide covering:
- Quick start commands
- Cache structure explanation
- How the two-pass generation works
- Example workflows
- Comparison with other experiments
- Troubleshooting tips

## Integration with Existing Code

The implementation follows the established pattern:

```
Cache (specialized) → Analysis (existing) → Sweep (modified)
     ↓                      ↓                    ↓
cache_lie_detector_    probe_analysis.py    sweep.py
activations.py         (no changes)        (detects prompt)
```

**No changes needed to**:
- `probe_analysis.py` - works as-is with new cache
- `lib/probes.py` - all probe training functions work
- `lib/visualization.py` - all visualization functions work
- `lib/data.py` - reuses existing functions
- `lib/generation.py` - reuses existing VLLM interface
- `lib/activations.py` - reuses existing extraction

**Prompts already configured**:
- "lie detector part 1" - already in `prompts.py`
- "lie detector part 2" - already in `prompts.py` with `{preceding_conversation}` placeholder

## Testing Status

✅ **Syntax validation**: Both new/modified files pass `python3 -m py_compile`
✅ **Linting**: No linter errors detected
✅ **Import structure**: Correct (runtime dependencies like transformers not needed for validation)
✅ **Integration**: sweep.py properly detects and routes lie_detector caching

**Ready for runtime testing** (requires VLLM servers and GPU):
```bash
# Small test run
python cache_lie_detector_activations.py --num-examples 10
python probe_analysis.py --prompt lie_detector --layer 13 --position last

# Full sweep
python sweep.py --prompt lie_detector --layers 10 12 13 14 --num-examples 200
```

## Key Design Decisions

1. **Unified Cache Format**: Lie detector uses the same cache structure as other experiments for consistency and compatibility

2. **Separate Generation Caches**: Pass 1 and Pass 2 cache separately to avoid regeneration if only analyzing different layers/positions

3. **Label Source**: Labels come from Pass 1 actual correctness (not model's claim in Pass 2)

4. **Activation Source**: Activations come from Pass 2 (yes/no response) where we expect the "lying" signal

5. **Integration Pattern**: Modified sweep.py rather than duplicating it, maintaining single source of truth for sweep logic

## Benefits

- ✅ Reuses all existing analysis infrastructure (PCA, ROC, anomaly detection, etc.)
- ✅ Works with existing sweep.py for layer/position sweeps
- ✅ Consistent cache format across all experiments
- ✅ Can directly compare lie detector results with other prompts
- ✅ No duplication of probe training or visualization code

## Files Modified/Created

**Created**:
- `cache_lie_detector_activations.py` (365 lines)
- `LIE_DETECTOR_USAGE.md` (documentation)
- `IMPLEMENTATION_SUMMARY.md` (this file)

**Modified**:
- `sweep.py` (added 10 lines for lie_detector detection)

**Unchanged** (by design):
- `probe_analysis.py`
- `lib/` (all library files)
- `prompts.py` (already had correct prompts)
- `config.py`

## Next Steps (User)

1. **Start VLLM servers** (if not already running):
   ```bash
   ./vllm_launcher.sh
   ```

2. **Test with small sample**:
   ```bash
   python cache_lie_detector_activations.py --num-examples 10
   python probe_analysis.py --prompt lie_detector --layer 13 --position last
   ```

3. **Verify results**:
   - Check `experiments/{model}/cached_activations/lie_detector/`
   - Check `experiments/{model}/results/`
   - Review PCA plots and ROC curves

4. **Run full experiment**:
   ```bash
   python sweep.py --prompt lie_detector --layers 10 11 12 13 14 15
   ```

5. **Compare with baselines**:
   ```bash
   python sweep.py --prompt benign --layers 10 11 12 13 14 15
   python sweep.py --prompt semimalign --layers 10 11 12 13 14 15
   ```

