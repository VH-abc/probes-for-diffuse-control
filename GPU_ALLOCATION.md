# GPU Allocation Configuration

## 🎯 Overview

To prevent CUDA OOM errors, the codebase now supports **separate GPU allocation** for VLLM servers and activation extraction. This allows you to run both simultaneously without memory conflicts.

## 📊 Default Configuration

In `config.py`:

```python
# VLLM servers use GPUs 0-3
VLLM_GPUS = [0, 1, 2, 3]

# Activation extraction uses GPUs 4-7  
ACTIVATION_GPUS = [4, 5, 6, 7]
```

## 🔧 Configuration Options

### Option 1: Split GPUs Evenly (8 GPUs)

**Best for: Systems with 8+ GPUs**

```python
# config.py
VLLM_GPUS = [0, 1, 2, 3]          # 4 GPUs for generation
ACTIVATION_GPUS = [4, 5, 6, 7]    # 4 GPUs for activation extraction
```

**Advantages:**
- ✅ No OOM issues
- ✅ Run generation and activation extraction simultaneously
- ✅ Maximum parallelism

### Option 2: Overlapping (Shared GPUs)

**Best for: Systems with 4 GPUs, when using generation caching**

```python
# config.py
VLLM_GPUS = [0, 1, 2, 3]          # All 4 GPUs for generation
ACTIVATION_GPUS = [0, 1, 2, 3]    # Same 4 GPUs for activation
```

**Advantages:**
- ✅ Use all available GPUs for each task
- ✅ Works well with generation caching (generation only runs once)
- ⚠️  Kill VLLM servers before activation extraction to avoid OOM

**Usage:**
```bash
# 1. Generate completions (only happens once due to caching)
python cache_activations.py --prompt "50/50" --layer 10

# 2. Kill VLLM servers
pkill -f vllm

# 3. Extract activations for other layers (uses cached generations)
python cache_activations.py --prompt "50/50" --layer 11
python cache_activations.py --prompt "50/50" --layer 12
```

### Option 3: Minimal GPUs for VLLM

**Best for: Prioritizing activation extraction speed**

```python
# config.py
VLLM_GPUS = [0, 1]               # Just 2 GPUs for generation
ACTIVATION_GPUS = [2, 3, 4, 5, 6, 7]  # 6 GPUs for activation
```

**Advantages:**
- ✅ Faster activation extraction
- ✅ No OOM issues
- ⚠️  Slower generation (but only happens once with caching)

### Option 4: Single GPU Setup

**Best for: Limited GPU systems**

```python
# config.py
VLLM_GPUS = [0]                  # 1 GPU for generation
ACTIVATION_GPUS = [0]            # Same GPU for activation
VLLM_NUM_SERVERS = 1             # Only 1 VLLM server
```

**Usage:**
```bash
# Generate completions
python cache_activations.py --prompt "50/50" --layer 10

# Kill VLLM to free memory
pkill -f vllm

# Extract activations
# (Re-run same command - will use cached generations)
python cache_activations.py --prompt "50/50" --layer 10
```

## 🚀 How It Works

### VLLM Server Launch

When you run `bash vllm_launcher.sh`:
1. Script reads `VLLM_GPUS` from config
2. Launches servers cycling through specified GPUs
3. Example with `VLLM_GPUS = [0, 1, 2, 3]` and 8 servers:
   - Server 0 → GPU 0
   - Server 1 → GPU 1
   - Server 2 → GPU 2
   - Server 3 → GPU 3
   - Server 4 → GPU 0 (cycles back)
   - Server 5 → GPU 1
   - Server 6 → GPU 2
   - Server 7 → GPU 3

### Activation Extraction

When extracting activations:
1. Code reads `ACTIVATION_GPUS` from config
2. Distributes work across specified GPUs
3. Example with `ACTIVATION_GPUS = [4, 5, 6, 7]`:
   - Chunk 1 → GPU 4
   - Chunk 2 → GPU 5
   - Chunk 3 → GPU 6
   - Chunk 4 → GPU 7

## 📈 Performance Comparison

### 8-GPU Setup Example

**Before (All GPUs shared):**
- VLLM uses GPUs 0-7 (50% memory each)
- Activation extraction tries to use GPUs 0-7
- Result: **CUDA OOM** ❌

**After (Split allocation):**
- VLLM uses GPUs 0-3 (50% memory each)
- Activation extraction uses GPUs 4-7 (100% memory each)
- Result: **No OOM, runs simultaneously** ✅

### Memory Usage Example (12B Model)

**GPU 0-3 (VLLM):**
- ~12GB per GPU (50% utilization)

**GPU 4-7 (Activation):**
- ~30GB per GPU during extraction
- 24GB for model + 6GB for forward pass

## 🛠️ Recommended Setups

### 8 GPUs (40GB each)
```python
VLLM_GPUS = [0, 1, 2, 3]
ACTIVATION_GPUS = [4, 5, 6, 7]
VLLM_NUM_SERVERS = 8
```
**Perfect! Maximum parallelism, no OOM issues.**

### 4 GPUs (80GB each)
```python
VLLM_GPUS = [0, 1]
ACTIVATION_GPUS = [2, 3]
VLLM_NUM_SERVERS = 4
```
**Good balance. Can run simultaneously.**

### 4 GPUs (40GB each) - With Caching
```python
VLLM_GPUS = [0, 1, 2, 3]
ACTIVATION_GPUS = [0, 1, 2, 3]
VLLM_NUM_SERVERS = 4
```
**Kill VLLM before activation extraction, but generation caching makes this efficient.**

### 2 GPUs (80GB each)
```python
VLLM_GPUS = [0]
ACTIVATION_GPUS = [1]
VLLM_NUM_SERVERS = 2
```
**Can run simultaneously on separate GPUs.**

## 🔍 Verifying GPU Usage

Check which processes are using which GPUs:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# See detailed process info
nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_bus_id --format=csv

# Check VLLM servers
ps aux | grep vllm | grep "CUDA_VISIBLE_DEVICES"
```

## 📝 Configuration Checklist

Before running experiments:

1. **Check your GPU count:**
   ```bash
   nvidia-smi --list-gpus
   ```

2. **Check GPU memory:**
   ```bash
   nvidia-smi --query-gpu=name,memory.total --format=csv
   ```

3. **Update config.py based on your setup**

4. **Verify no overlap if running simultaneously:**
   ```python
   # Make sure these don't overlap:
   VLLM_GPUS = [0, 1, 2, 3]
   ACTIVATION_GPUS = [4, 5, 6, 7]
   ```

5. **Or ensure you'll kill VLLM first:**
   ```python
   # Can overlap if you kill VLLM before activation:
   VLLM_GPUS = [0, 1, 2, 3]
   ACTIVATION_GPUS = [0, 1, 2, 3]
   ```

## 💡 Pro Tips

1. **Use generation caching**: With caching, generation only happens once, so you can:
   - Use all GPUs for generation initially
   - Kill VLLM servers
   - Use all GPUs for activation extraction
   - Subsequent layer sweeps are instant (cached)

2. **Monitor memory**: Use `nvidia-smi` to check actual usage and adjust allocations

3. **Adjust VLLM memory**: Lower `VLLM_GPU_MEMORY_UTILIZATION` if still getting OOM:
   ```python
   VLLM_GPU_MEMORY_UTILIZATION = 0.3  # Down from 0.5
   ```

4. **Scale servers to GPUs**: Match server count to GPU availability:
   ```python
   VLLM_GPUS = [0, 1, 2, 3]
   VLLM_NUM_SERVERS = 4  # or 8 (will cycle through GPUs)
   ```

## ⚠️ Common Issues

### Issue: "CUDA out of memory" during activation extraction

**Solution 1**: Ensure GPUs don't overlap
```python
VLLM_GPUS = [0, 1, 2, 3]
ACTIVATION_GPUS = [4, 5, 6, 7]  # Different GPUs
```

**Solution 2**: Kill VLLM servers first
```bash
pkill -f vllm
```

**Solution 3**: Reduce batch size
```bash
python cache_activations.py --prompt "50/50" --layer 10 --batch-size 1
```

### Issue: VLLM servers not starting

**Solution**: Check if GPUs exist
```bash
nvidia-smi
# Make sure VLLM_GPUS references valid GPU IDs
```

### Issue: Activation extraction using wrong GPUs

**Solution**: Verify config is being used
```python
# Add debug print to cache_activations.py
print(f"Using activation GPUs: {config.ACTIVATION_GPUS}")
```

## 🎓 Example Workflows

### Workflow 1: Separate GPUs (Simultaneous)

```bash
# config.py: VLLM_GPUS = [0,1,2,3], ACTIVATION_GPUS = [4,5,6,7]

# Start VLLM servers (will stay running)
bash vllm_launcher.sh

# Run layer sweep (generation + activation for each layer)
python sweep_layers.py --prompt "50/50" --layers 10 11 12 13 14

# VLLM and activation run simultaneously with no conflicts!
```

### Workflow 2: Shared GPUs (Sequential with Caching)

```bash
# config.py: VLLM_GPUS = [0,1,2,3], ACTIVATION_GPUS = [0,1,2,3]

# Start VLLM servers
bash vllm_launcher.sh

# First layer (generates and caches completions)
python cache_activations.py --prompt "50/50" --layer 10

# Kill VLLM to free memory
pkill -f vllm

# Subsequent layers (use cached completions, no VLLM needed)
python cache_activations.py --prompt "50/50" --layer 11
python cache_activations.py --prompt "50/50" --layer 12
python cache_activations.py --prompt "50/50" --layer 13

# Generation happened once, activation extraction 4 times
```

