# Setup Summary - OOM Prevention & Caching

## 🎉 What's Been Implemented

This document summarizes the improvements made to prevent CUDA OOM errors and speed up layer sweeps.

## ✅ Feature 1: Separate GPU Allocation

**Problem**: VLLM servers and activation extraction compete for GPU memory → OOM errors

**Solution**: Configurable GPU allocation in `config.py`

### Configuration

In `/workspace/probes-for-diffuse-control/config.py`:

```python
# VLLM servers use GPUs 0-3
VLLM_GPUS = [0, 1, 2, 3]

# Activation extraction uses GPUs 4-7 (no overlap!)
ACTIVATION_GPUS = [4, 5, 6, 7]
```

### How It Works

1. **VLLM Launcher** (`vllm_launcher.sh`):
   - Reads `VLLM_GPUS` from config
   - Launches servers on specified GPUs only
   - Cycles through GPUs if `NUM_SERVERS > len(VLLM_GPUS)`

2. **Activation Extraction** (`lib/activations.py`):
   - Reads `ACTIVATION_GPUS` from config
   - Only uses specified GPUs for model loading
   - Distributes work across designated GPUs

### Benefits

- ✅ **No OOM errors** - GPUs don't compete for memory
- ✅ **Run simultaneously** - VLLM and activation extraction can coexist
- ✅ **Fully configurable** - Adjust for your GPU setup

### Documentation

- See `GPU_ALLOCATION.md` for detailed configuration options
- See `TROUBLESHOOTING_OOM.md` for OOM solutions

---

## ✅ Feature 2: Generation Caching

**Problem**: Layer sweeps regenerate the same completions for every layer → slow & wasteful

**Solution**: Automatic caching of model generations

### How It Works

1. **First Generation**:
   ```bash
   python cache_activations.py --prompt "50-50" --layer 10
   ```
   - Generates 200 completions
   - Saves to: `experiments/gemma-3-12b/generations/50-50/{cache_key}_*`
   - Proceeds with activation extraction

2. **Subsequent Runs**:
   ```bash
   python cache_activations.py --prompt "50-50" --layer 11
   ```
   - **Detects cached generations** ✨
   - Loads from disk (instant!)
   - Skips VLLM generation entirely
   - Only runs activation extraction

### Cache Location

```
experiments/
└── gemma-3-12b/
    └── generations/
        ├── benign/
        │   └── {cache_key}_full_texts.npy
        │   └── {cache_key}_completions.npy
        │   └── {cache_key}_metadata.json
        └── 50-50/
            └── {cache_key}_full_texts.npy
            └── {cache_key}_completions.npy
            └── {cache_key}_metadata.json
```

### Cache Key

Hash of:
- Model name
- Temperature
- Max tokens
- All prompts

**Same parameters** = **Same cache** ✅

### Benefits

- ✅ **Massive speedup** for layer sweeps
- ✅ **Automatic** - no flags needed
- ✅ **Separate per prompt** - "benign" and "50-50" don't conflict
- ✅ **Persistent** - survives across sessions

### Example

**Without caching:**
```bash
# Layer sweep - 8 layers × 5 min = 40 min generation time
python sweep_layers.py --prompt "50-50" --layers 10 11 12 13 14 15 16 17
```

**With caching:**
```bash
# Layer sweep - 1 × 5 min generation, rest instant!
python sweep_layers.py --prompt "50-50" --layers 10 11 12 13 14 15 16 17
# First layer: 5 min generation + activation
# Layers 11-17: Instant load + activation only
```

### Documentation

- See `GENERATION_CACHING.md` for full details

---

## ✅ Feature 3: Reduced Batch Size & Memory Optimizations

**Changes:**
1. Default `ACTIVATION_BATCH_SIZE` reduced from 4 → 1
2. Added `--batch-size` command-line argument
3. Automatic CUDA cache clearing every 50 samples
4. Added command-line control for activation batch size

### Usage

```bash
# Use default (batch_size=1)
python cache_activations.py --prompt "50-50" --layer 10

# Or override if you have more memory
python cache_activations.py --prompt "50-50" --layer 10 --batch-size 2
```

---

## 🚀 Recommended Workflow

### For 8-GPU Systems

**Best setup** (separate GPUs):

```python
# config.py
VLLM_GPUS = [0, 1, 2, 3]        # 4 GPUs for generation
ACTIVATION_GPUS = [4, 5, 6, 7]  # 4 GPUs for activation
```

**Usage:**
```bash
# Start VLLM (stays running)
bash vllm_launcher.sh

# Run layer sweep - everything works simultaneously!
python sweep_layers.py --prompt "50-50" --layers 10 11 12 13 14 15 16

# First layer: Generates completions (cached) + extracts activations
# Layers 11-16: Loads cached completions + extracts activations
# All while VLLM servers stay running - no OOM!
```

### For 4-GPU Systems

**Best setup** (shared GPUs + caching):

```python
# config.py
VLLM_GPUS = [0, 1, 2, 3]        # All GPUs for generation
ACTIVATION_GPUS = [0, 1, 2, 3]  # Same GPUs for activation
```

**Usage:**
```bash
# Start VLLM
bash vllm_launcher.sh

# Generate completions (only happens once due to caching)
python cache_activations.py --prompt "50-50" --layer 10

# Kill VLLM to free memory
pkill -f vllm

# Extract activations for all other layers (uses cached generations)
python cache_activations.py --prompt "50-50" --layer 11
python cache_activations.py --prompt "50-50" --layer 12
python cache_activations.py --prompt "50-50" --layer 13
# All instant! No regeneration needed.
```

---

## 📁 Files Modified

### Core Changes
- ✅ `config.py` - Added `VLLM_GPUS` and `ACTIVATION_GPUS`
- ✅ `lib/generation.py` - Added generation caching system
- ✅ `lib/activations.py` - Added `gpu_ids` parameter
- ✅ `cache_activations.py` - Integrated GPU allocation & caching
- ✅ `vllm_launcher.sh` - Uses `VLLM_GPUS` from config
- ✅ `math_vs_nonmath_experiment.py` - Uses `ACTIVATION_GPUS`
- ✅ `identify_reliable_questions.py` - Uses generation caching

### Documentation Added
- 📄 `GPU_ALLOCATION.md` - GPU configuration guide
- 📄 `GENERATION_CACHING.md` - Caching system documentation
- 📄 `TROUBLESHOOTING_OOM.md` - OOM solutions (updated)
- 📄 `SETUP_SUMMARY.md` - This file

---

## 🎯 Quick Start

1. **Configure GPUs** (edit `config.py`):
   ```python
   VLLM_GPUS = [0, 1, 2, 3]      # Adjust for your system
   ACTIVATION_GPUS = [4, 5, 6, 7]  # Should not overlap (or kill VLLM first)
   ```

2. **Start VLLM servers**:
   ```bash
   bash vllm_launcher.sh
   ```

3. **Run experiments**:
   ```bash
   # First run - generates and caches
   python cache_activations.py --prompt "50-50" --layer 10
   
   # Subsequent runs - uses cache
   python cache_activations.py --prompt "50-50" --layer 11
   python cache_activations.py --prompt "50-50" --layer 12
   ```

4. **Layer sweeps**:
   ```bash
   # All layers use the same cached generations!
   python sweep_layers.py --prompt "50-50" --layers 10 11 12 13 14 15 16
   ```

---

## ✨ Key Benefits

1. **No More OOM** - Separate GPU allocation prevents memory conflicts
2. **Much Faster** - Generation caching eliminates redundant work
3. **Configurable** - Adjust for your specific hardware setup
4. **Automatic** - Caching works transparently, no flags needed
5. **Persistent** - Caches survive across sessions

---

## 📊 Performance Impact

**Example: 8-layer sweep with 200 examples**

**Before:**
- 8 layers × 5 min generation = 40 min
- 8 layers × 10 min activation = 80 min
- **Total: ~120 minutes**
- OOM errors likely ❌

**After:**
- 1 × 5 min generation (cached) = 5 min
- 8 layers × 10 min activation = 80 min
- **Total: ~85 minutes**
- No OOM errors ✅
- **Savings: ~35 minutes (29%)**

**For subsequent sweeps:**
- 0 min generation (cached!) = 0 min
- 8 layers × 10 min activation = 80 min
- **Total: ~80 minutes**
- **Savings: ~40 minutes (33%)**

---

## 🔧 Troubleshooting

**Still getting OOM?**
1. Check `GPU_ALLOCATION.md` for configuration options
2. Check `TROUBLESHOOTING_OOM.md` for solutions
3. Verify GPUs: `nvidia-smi`
4. Verify config: `python3 -c "import config; print(config.VLLM_GPUS, config.ACTIVATION_GPUS)"`

**Cache not working?**
1. Check cache location exists: `ls experiments/*/generations/`
2. Check for cache hits in output: Look for "✓ Found cached generations!"
3. Different parameters create different caches (expected)

---

## 📚 Full Documentation

- `GPU_ALLOCATION.md` - GPU configuration guide
- `GENERATION_CACHING.md` - Caching system details
- `TROUBLESHOOTING_OOM.md` - OOM error solutions
- `README.md` - General project documentation


