# Troubleshooting CUDA Out of Memory (OOM) Errors

## 🔴 Problem
Getting CUDA OOM errors during activation extraction step when caching activations.

## ✅ Solutions (Try in Order)

### 1. **Use Separate GPUs for VLLM and Activation** (BEST - Now Supported!)

Configure different GPUs for VLLM servers and activation extraction in `config.py`:

```python
# Use GPUs 0-3 for VLLM servers
VLLM_GPUS = [0, 1, 2, 3]

# Use GPUs 4-7 for activation extraction (no overlap!)
ACTIVATION_GPUS = [4, 5, 6, 7]
```

Then restart VLLM servers:
```bash
pkill -f vllm
bash vllm_launcher.sh
```

Now both can run simultaneously with no OOM! 🎉

See `GPU_ALLOCATION.md` for detailed configuration options.

### 2. **Use Reduced Batch Size** (EASIEST - Already Applied)
The default batch size has been changed from 4 to 1. You can also override it:

```bash
python cache_activations.py --prompt "50/50" --layer 10 --batch-size 1
```

### 3. **Shut Down VLLM Servers During Activation Extraction** (RECOMMENDED if GPUs overlap)
VLLM servers use 50% of GPU memory. Free it before extraction:

```bash
# Stop VLLM servers
pkill -f vllm

# Run activation caching
python cache_activations.py --prompt "50/50" --layer 10

# Restart VLLM servers when done
bash vllm_launcher.sh
```

### 4. **Use Fewer GPUs**
This gives each GPU more memory:

```bash
python cache_activations.py --prompt "50/50" --layer 10 --num-gpus 4
# Or even fewer:
python cache_activations.py --prompt "50/50" --layer 10 --num-gpus 2
```

### 5. **Process Fewer Examples**
Reduce the dataset size temporarily:

```bash
python cache_activations.py --prompt "50/50" --layer 10 --num-examples 100
```

### 6. **Reduce VLLM Memory Usage** (If you need VLLM running)
Edit `config.py`:

```python
VLLM_GPU_MEMORY_UTILIZATION = 0.3  # Reduce from 0.5 to 0.3
```

Then restart VLLM servers:
```bash
pkill -f vllm
bash vllm_launcher.sh
```

### 7. **Process in Multiple Passes**
Split your work into smaller batches:

```bash
# First 100 examples
python cache_activations.py --prompt "50/50" --layer 10 --num-examples 100

# Then combine the results manually or run on different layers
python cache_activations.py --prompt "50/50" --layer 11 --num-examples 100
```

## 📊 Memory Usage Breakdown

For `google/gemma-3-12b-it` model (12B parameters):

- **Model size in bfloat16**: ~24GB
- **VLLM servers (8 GPUs @ 50% each)**: ~12GB per GPU
- **Activation extraction**: 
  - Loading model: ~24GB
  - Forward pass with batch_size=1: ~2-4GB
  - Forward pass with batch_size=4: ~8-16GB

## 🎯 Recommended Workflow

**Option A: Sequential (Safest)**
```bash
# 1. Generate completions with VLLM
bash vllm_launcher.sh
python cache_activations.py --prompt "50/50" --layer 10 --num-examples 200
# (This will only run generation, then fail at activation extraction)

# 2. Stop VLLM servers
pkill -f vllm

# 3. Extract activations with saved generations (TODO: add skip-generation flag)
# For now, just re-run with VLLM down - it will skip generation if files exist
```

**Option B: Reduce VLLM Memory**
```bash
# Edit config.py: VLLM_GPU_MEMORY_UTILIZATION = 0.3
bash vllm_launcher.sh
python cache_activations.py --prompt "50/50" --layer 10 --batch-size 1
```

**Option C: Use Fewer GPUs**
```bash
python cache_activations.py --prompt "50/50" --layer 10 --num-gpus 2 --batch-size 1
```

## 🛠️ New Features Added

1. **Command-line `--batch-size` control**: Override default batch size
2. **Automatic CUDA cache clearing**: Clears GPU cache every 50 samples
3. **Reduced default batch size**: Changed from 4 to 1 in `config.py`

## 📝 Config Changes Made

`config.py`:
```python
ACTIVATION_BATCH_SIZE = 1  # Changed from 4
```

## 🔍 Monitoring GPU Memory

Check GPU memory usage:
```bash
# Watch GPU memory in real-time
watch -n 1 nvidia-smi

# Or just check once
nvidia-smi
```

## ⚠️ If Still Getting OOM

Try this nuclear option - use only 1 GPU:
```bash
pkill -f vllm  # Stop VLLM
CUDA_VISIBLE_DEVICES=0 python cache_activations.py --prompt "50/50" --layer 10 --num-gpus 1 --batch-size 1 --num-examples 50
```

