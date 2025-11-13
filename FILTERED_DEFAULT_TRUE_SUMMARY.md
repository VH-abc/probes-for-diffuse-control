# Filtered Flag Now Defaults to True

## Summary
Changed the `--filtered` flag in `sweep_layers.py` to default to `True`. Now all sweep operations will use filtered (reliable) questions by default unless explicitly disabled.

## Changes Made

### Updated Argument Parser in `sweep_layers.py`

**Before:**
```python
parser.add_argument("--filtered", action="store_true",
                    help="Filter to only reliable questions...")
# Default: False (unfiltered)
```

**After:**
```python
filter_group = parser.add_mutually_exclusive_group()
filter_group.add_argument("--filtered", dest="filtered", action="store_true", default=True,
                    help="Filter to only reliable questions (questions with 100% pass rate) (default)")
filter_group.add_argument("--no-filter", "--unfiltered", dest="filtered", action="store_false",
                    help="Use all questions (unfiltered)")
# Default: True (filtered)
```

## Usage Changes

### Old Behavior (filtered=False by default):
```bash
# Uses unfiltered data
python sweep_layers.py --layers 10 12 13

# Need to explicitly add --filtered to use filtered data
python sweep_layers.py --layers 10 12 13 --filtered
```

### New Behavior (filtered=True by default):
```bash
# Uses filtered data (default)
python sweep_layers.py --layers 10 12 13

# Use --no-filter or --unfiltered to use all questions
python sweep_layers.py --layers 10 12 13 --no-filter
python sweep_layers.py --layers 10 12 13 --unfiltered
```

## Examples

### Layer Sweep (filtered by default):
```bash
python sweep_layers.py --prompt "50-50" --layers 10 12 13 14
# Automatically uses filtered questions
```

### Layer Sweep (explicitly unfiltered):
```bash
python sweep_layers.py --prompt "50-50" --layers 10 12 13 14 --no-filter
# Uses all questions
```

### Position Sweep (filtered by default):
```bash
python sweep_layers.py --mode positions --layer 12
# Automatically uses filtered questions
```

### Quick Sweep (filtered by default):
```bash
python sweep_layers.py --quick
# Automatically uses filtered questions
```

## Benefits

1. **✅ Safer default** - Uses reliable questions by default, ensuring better quality results
2. **🎯 More consistent** - Most experiments should use filtered data
3. **📝 Clear intent** - Need to explicitly opt-in to unfiltered data with `--no-filter`
4. **⚡ Fewer arguments** - No need to add `--filtered` to every command

## Flag Options

Both flags work to disable filtering:
- `--no-filter` - Shorter option
- `--unfiltered` - More descriptive option

Both are mutually exclusive with `--filtered`, so you can only use one at a time.

