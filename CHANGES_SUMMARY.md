# Changes Summary

## Recent Updates

### 1. Separate Results Directory for Lie Detector Experiment

**Modified**: `config.py`, `probe_analysis.py`, `sweep.py`

**Changes**:
- Updated `get_results_dir()` function in `config.py` to accept optional `prompt_name` parameter
- Lie detector results now saved to: `experiments/{model}/results/lie_detector/n{N}_{filtered|unfiltered}/`
- Other prompts continue using: `experiments/{model}/results/n{N}_{filtered|unfiltered}/`
- Updated all callers of `get_results_dir()` to pass `prompt_name` parameter

**Benefit**: Clean separation of lie detector experiment results from other experiments.

### 2. New Token Position: "yes-no"

**Modified**: `lib/activations.py`, `config.py`, `LIE_DETECTOR_USAGE.md`

**New Functionality**:
- Added `"yes-no"` token position type
- Created `find_yes_no_token_position()` function that:
  - Searches for "yes" or "no" tokens after "Assistant:" in the text
  - Specifically designed for lie detector experiments
  - Falls back to "last" position if yes/no not found
- Added handling in activation extraction code (similar to "letter" position)
- Requires `completions` parameter to be provided (like "letter" position)

**Usage**:
```bash
# Recommended for lie detector experiments
python probe_analysis.py --prompt lie_detector --layer 13 --position yes-no

# Or in sweeps
python sweep.py --prompt lie_detector --positions yes-no last --layer 13
```

**How it works**:
1. Tokenizes the full text
2. Finds "Assistant:" in the tokens
3. Looks for tokens containing "yes" or "no" within next 10 tokens
4. Returns the position of the first yes/no token found
5. Falls back to "last" position if not found

**Configuration Updates**:
- Added to `SUPPORTED_POSITIONS` in `config.py`
- Added to `DEFAULT_POSITION_SWEEP` in `config.py`
- Automatically available in all argument parsers that use `config.SUPPORTED_POSITIONS`

### 3. Documentation Updates

**Modified**: `LIE_DETECTOR_USAGE.md`

**Updates**:
- Added section explaining different token position options
- Recommended `yes-no` position for lie detector experiments
- Added instructions for verifying which tokens were probed
- Updated troubleshooting section with `yes-no` specific guidance
- Updated all example commands to show `yes-no` position usage

## File Changes Summary

### Modified Files
1. **`config.py`**:
   - Updated `get_results_dir()` signature to accept `prompt_name`
   - Added logic to route lie_detector to separate directory
   - Added `"yes-no"` to `SUPPORTED_POSITIONS` and `DEFAULT_POSITION_SWEEP`

2. **`lib/activations.py`**:
   - Added `"yes-no"` to `TokenPosition` Literal type
   - Created `find_yes_no_token_position()` function (33 lines)
   - Added validation for `yes-no` requiring completions
   - Added extraction logic for `yes-no` position (20 lines)

3. **`probe_analysis.py`**:
   - Updated `get_results_dir()` call to pass `prompt_name`

4. **`sweep.py`**:
   - Updated both `get_results_dir()` calls to pass `prompt_name`

5. **`LIE_DETECTOR_USAGE.md`**:
   - Added "Token Positions" section
   - Updated examples to use `yes-no` position
   - Enhanced troubleshooting section

## Benefits

1. **Better Organization**: Lie detector results are cleanly separated from other experiments
2. **More Precise Extraction**: `yes-no` position specifically targets the yes/no token
3. **Easier Debugging**: Can verify exact tokens probed with position-specific JSON files
4. **Consistent API**: New position works seamlessly with existing sweep and analysis tools

## Testing

All modified files pass:
- ✅ Syntax validation (`python3 -m py_compile`)
- ✅ Linter checks (no errors)
- ✅ Import structure (correct dependencies)

## Usage Examples

### Using yes-no position with lie detector

```bash
# Cache activations (includes yes-no if specified)
python cache_lie_detector_activations.py --num-examples 200

# Analyze with yes-no position
python probe_analysis.py --prompt lie_detector --layer 13 --position yes-no

# Sweep across layers
python sweep.py --prompt lie_detector --layers 10 12 13 14 --position yes-no

# Compare yes-no vs last position
python sweep.py --prompt lie_detector --pairs 13,yes-no 13,last --num-examples 200
```

### Verify token extraction

```bash
# Check what tokens were captured
cat experiments/gemma-3-12b/cached_activations/lie_detector/*/probed_tokens_pos-yes-no.json

# Compare with last position
cat experiments/gemma-3-12b/cached_activations/lie_detector/*/probed_tokens_pos-last.json
```

## Backward Compatibility

✅ All changes are backward compatible:
- Existing experiments continue to work without modification
- `get_results_dir()` has default `prompt_name=None` for backward compatibility
- New `yes-no` position is optional - existing positions still work
- Documentation clearly indicates when to use new features

## Future Considerations

The pattern established here (specialized token positions + separate results directories) can be easily extended for future specialized experiments:
- Other multi-turn dialogue experiments
- Chain-of-thought analysis
- Self-reflection experiments
- Any experiment requiring specific token extraction patterns

