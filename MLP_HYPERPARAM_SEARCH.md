# MLP Hyperparameter Search

## Overview

The `mlp_hyperparameter_search.py` script systematically tests different MLP configurations to find optimal settings for probe training. This is useful when larger/deeper networks underperform and you need to tune regularization and learning rates.

## Problem Being Solved

**Observation**: Bigger MLPs (e.g., multiple layers with 12k hidden units) perform **worse** than expected.

**Likely causes**:
- Overfitting (insufficient regularization)
- Poor learning rate (too high or too low)
- Early stopping too aggressive or too lenient
- Suboptimal architecture choices

**Solution**: Systematic hyperparameter search to find optimal configuration.

## What It Tests

The script performs **comprehensive sweeps** across multiple dimensions:

### 1. **Architecture Sweep**
Tests different network sizes and depths:
```python
(12000,)             # Single large layer
(8000,)              # Single medium layer
(6000,), (4000,)     # Smaller single layers
(12000, 6000)        # Two layers, decreasing
(8000, 4000)         # Two layers, medium
(12000, 12000)       # Two layers, same size
(12000, 6000, 3000)  # Three layers
```

### 2. **Learning Rate Sweep**
```python
[0.001, 0.0005, 0.0001, 0.00005]
```
Lower LRs may help with stability for large networks.

### 3. **Weight Decay Sweep** (L2 Regularization)
```python
[1e-4, 1e-3, 1e-2, 5e-2]
```
Higher values = more regularization = less overfitting.

### 4. **Dropout Sweep**
```python
[0.1, 0.2, 0.3]
```
Higher dropout = more regularization.

### 5. **Patience Sweep** (Early Stopping)
```python
[5, 10, 15, 20]
```
Higher patience = train longer before stopping.

### 6. **LR Scheduler Sweep**
```python
[None, "plateau", "step"]
```
- **None**: Fixed learning rate
- **"plateau"**: Reduce LR when validation loss plateaus (factor=0.5, patience=5)
- **"step"**: Reduce LR every 50 epochs (factor=0.5)

### 7. **Combined Configurations**
Tests hand-picked combinations:
- **Conservative**: Small network, low LR, high regularization, plateau scheduler
- **Aggressive**: Large network, high LR, low regularization, no scheduler
- **Balanced**: Medium network, medium settings, plateau scheduler

## Usage

### Basic Usage

```bash
python mlp_hyperparameter_search.py \
    --layer 20 \
    --position letter \
    --num-examples 2000
```

### With Lie Detector Experiment

```bash
python mlp_hyperparameter_search.py \
    --layer 20 \
    --position letter \
    --num-examples 2000 \
    --prompt lie_detector
```

### With Different Model

```bash
python mlp_hyperparameter_search.py \
    --layer 30 \
    --position last \
    --num-examples 2000 \
    --model "meta-llama/Llama-2-7b-chat-hf"
```

### Without Filtering

```bash
python mlp_hyperparameter_search.py \
    --layer 20 \
    --position letter \
    --num-examples 2000 \
    --no-filter
```

## Requirements

### Prerequisites

1. **Cached activations must exist** for the specified layer/position:
   ```bash
   # Run caching first
   python cache_activations_unified.py --num-examples 2000
   # Or for lie detector
   python cache_lie_detector_activations.py --num-examples 2000
   ```

2. **GPU recommended** (will use CPU if unavailable, but much slower)

## Output

### 1. Incremental Results File (JSONL) - NEW!

`experiments/{model}/results/n{N}_{filter}/mlp_hyperparam_incremental_layer{L}_pos-{P}.jsonl`

**Key features:**
- ✅ **Never cleared** - always appends
- ✅ **Saves after each test** - results preserved if interrupted
- ✅ **Multiple runs tracked** - each run adds a new section
- ✅ **Timestamped** - every result has a timestamp

Format (JSONL - one JSON object per line):
```
================================================================================
NEW RUN: 2025-11-17 10:30:45
Layer: 20, Position: letter, Examples: 2000
Configurations to test: 42
================================================================================

{"auroc": 0.8012, "train_time": 42.3, "sweep_name": "arch_sweep", "config_number": 1, "timestamp": "2025-11-17 10:31:12", "config": {...}}
{"auroc": 0.8156, "train_time": 38.1, "sweep_name": "arch_sweep", "config_number": 2, "timestamp": "2025-11-17 10:32:05", "config": {...}}
...
{"auroc": 0.8234, "train_time": 45.3, "sweep_name": "balanced", "config_number": 42, "timestamp": "2025-11-17 11:15:32", "config": {...}}

================================================================================
RUN COMPLETE: 2025-11-17 11:15:35
Tested 42 configurations
Best AUROC: 0.8234
Best config: {...}
================================================================================
```

**Why JSONL?**
- Each line is a complete JSON object
- Easy to append without parsing entire file
- Can be read line-by-line if needed
- Preserves history across multiple runs

### 2. JSON Results File (Summary)

`experiments/{model}/results/n{N}_{filter}/mlp_hyperparam_search_layer{L}_pos-{P}.json`

Contains:
```json
{
  "layer": 20,
  "position": "letter",
  "num_examples": 2000,
  "filter_reliable": true,
  "prompt_name": null,
  "num_configs_tested": 42,
  "best_config": {
    "auroc": 0.8234,
    "train_time": 45.3,
    "config": {
      "hidden_layer_sizes": [8000],
      "learning_rate": 0.0005,
      "weight_decay": 0.001,
      "dropout": 0.2,
      "patience": 10,
      "lr_scheduler": "plateau"
    }
  },
  "all_results": [...]
}
```

**Note:** This file is **overwritten** each run. Use the incremental file for permanent history.

### 3. Text Report

`experiments/{model}/results/n{N}_{filter}/mlp_hyperparam_report_layer{L}_pos-{P}.txt`

Human-readable report with:
- **Top 10 configurations** ranked by AUROC
- **Results by sweep type** (architecture, LR, weight decay, etc.)
- Full configuration details for each run

Example:
```
================================================================================
TOP 10 CONFIGURATIONS
================================================================================
Rank   AUROC    Time(s)    Architecture         LR         WD         Drop   Pat   Sched     
------------------------------------------------------------------------------------------------
1      0.8234   45.3       (8000,)              0.000500   0.001000   0.20   10    plateau   
2      0.8201   38.7       (6000,)              0.000100   0.001000   0.20   15    plateau   
3      0.8189   52.1       (12000,)             0.000500   0.001000   0.20   10    plateau   
...
```

### 3. Console Output

Real-time progress for each configuration:
```
================================================================================
Configuration 1/42: arch_sweep
================================================================================
Architecture: (12000,)
LR: 0.001, WD: 0.0001, Dropout: 0.1
Patience: 10, Scheduler: None

    Using device: cuda
    GPU: NVIDIA A100-SXM4-80GB
    Model parameters: 190,092,001 (trainable: 190,092,001)
    Architecture: Input(3840) → Hidden(12000) → Output(1)
    Hyperparameters: LR=0.001, WD=0.0001, Dropout=0.1, Patience=10
    LR Scheduler: None
    Training samples: 1600, Validation: 200

    Starting training (max 1000 epochs, early stopping patience=10)...
    Epoch    Train Loss   Val Loss     LR           Status              
    -------- ------------ ------------ ------------ --------------------
    0        0.682347     0.678912     0.00100000   ✓ New best!
    10       0.621456     0.635821     0.00100000   ✓ New best!
    20       0.589234     0.612453     0.00100000   No improve (1/10)
    ...

    Early stopping at epoch 150 (best was epoch 140)
    Best validation loss: 0.601234 (epoch 140)
    Training complete!

  → AUROC: 0.8012
  → Train time: 42.3s
```

## Configuration Details

### Updated ResidualMLPClassifier

The `ResidualMLPClassifier` now supports all these hyperparameters:

```python
clf = ResidualMLPClassifier(
    hidden_layer_sizes=(12000,),      # Architecture
    learning_rate=0.001,               # NEW: Configurable LR
    weight_decay=1e-4,                 # NEW: Configurable L2 reg
    dropout=0.1,                       # NEW: Configurable dropout
    patience=10,                       # NEW: Configurable early stop
    lr_scheduler=None,                 # NEW: Optional scheduler
    max_iter=1000,                     # Max epochs
    random_state=42                    # Reproducibility
)
```

### LR Schedulers

**ReduceLROnPlateau** (`lr_scheduler="plateau"`):
- Monitors validation loss
- Reduces LR by 50% if no improvement for 5 checks
- Good for adaptive learning rate

**StepLR** (`lr_scheduler="step"`):
- Reduces LR by 50% every 50 epochs
- Predictable schedule
- Good for long training runs

## Interpreting Results

### What to Look For

1. **Best AUROC**: Primary metric
2. **Train time**: Secondary consideration (faster is better if AUROC similar)
3. **Overfitting indicators**:
   - Very high train AUROC, lower test AUROC
   - Large networks with low regularization often overfit

### Common Patterns

**If larger networks underperform**:
- ✅ Increase regularization (higher weight decay, higher dropout)
- ✅ Decrease learning rate
- ✅ Use plateau scheduler
- ✅ Increase patience (let it train longer)

**If training is unstable**:
- ✅ Lower learning rate
- ✅ Use plateau scheduler
- ✅ Increase dropout

**If training stops too early**:
- ✅ Increase patience
- ✅ Lower weight decay
- ✅ Lower dropout

**If small networks do better**:
- May not need large capacity for this task
- Use simpler architecture to save compute

## Example Workflow

### Step 1: Run Search

```bash
python mlp_hyperparameter_search.py \
    --layer 20 \
    --position letter \
    --num-examples 2000
```

This will test ~42 configurations (takes 30-60 minutes on GPU).

### Step 2: Review Results

```bash
# Check best configuration
cat experiments/gemma-3-12b/results/n2000_filtered/mlp_hyperparam_report_layer20_pos-letter.txt | head -20

# Or open the JSON
python -c "import json; print(json.dumps(json.load(open('experiments/gemma-3-12b/results/n2000_filtered/mlp_hyperparam_search_layer20_pos-letter.json'))['best_config'], indent=2))"
```

### Step 3: Apply Best Settings

Update `probe_analysis.py` or your training script with the best configuration:

```python
# Before (defaults)
clf = ResidualMLPClassifier(
    hidden_layer_sizes=(12000,),
    max_iter=1000,
    random_state=42
)

# After (optimized from search)
clf = ResidualMLPClassifier(
    hidden_layer_sizes=(8000,),         # From search results
    learning_rate=0.0005,               # From search results
    weight_decay=0.001,                 # From search results
    dropout=0.2,                        # From search results
    patience=10,                        # From search results
    lr_scheduler="plateau",             # From search results
    max_iter=1000,
    random_state=42
)
```

### Step 4: Validate on Other Layers/Positions

Test the best configuration on other layers/positions to ensure it generalizes:

```bash
# Test best config on layer 30
python mlp_hyperparameter_search.py --layer 30 --position letter --num-examples 2000

# Test best config on position "last"
python mlp_hyperparameter_search.py --layer 20 --position last --num-examples 2000
```

## Tips

### Speed Up Search

1. **Test fewer configurations**: Edit the script to remove some sweeps
2. **Reduce max_iter**: Change from 1000 to 500 epochs
3. **Use smaller validation set**: Faster evaluation (edit script)

### For Production

After finding optimal settings:
1. Document the configuration
2. Add to `config.py` as defaults
3. Update documentation/papers with final hyperparameters

### For Papers

Report:
- Full search space explored
- Best configuration found
- AUROC improvement vs. default
- Total configurations tested
- Compute time for search

## Recovering from Interruptions

One of the key features of the incremental saving is **crash recovery**. If the script is interrupted:

### Reading the Incremental File

```python
import json

# Read all results from incremental file
results = []
with open('mlp_hyperparam_incremental_layer20_pos-letter.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if line and line.startswith('{'):  # JSON lines only
            results.append(json.loads(line))

print(f"Found {len(results)} completed configurations")

# Find best so far
best = max(results, key=lambda x: x['auroc'])
print(f"Best AUROC so far: {best['auroc']:.4f}")
print(f"Config: {best['config']}")
```

### What's Preserved

✅ **Every completed configuration** - saved immediately after training  
✅ **Timestamps** - know when each test ran  
✅ **Multiple runs** - if you restart, both runs are in the file  
✅ **Best result** - can always find the best configuration tested so far

### What's Lost

❌ **Current training progress** - if interrupted mid-training, that test is lost  
❌ **Remaining configurations** - need to manually skip completed ones if resuming

## Troubleshooting

**Q: Script fails with "Activations not found"**
- Run caching script first: `python cache_activations_unified.py --num-examples 2000`

**Q: Script was interrupted, can I resume?**
- Check the incremental file to see what completed
- Edit the script to remove already-tested configurations
- Or just re-run (results append, duplicates are okay for analysis)

**Q: Incremental file is huge**
- Normal - it never clears across runs
- You can manually archive/delete old sections if needed
- Each run is clearly marked with timestamps

**Q: Out of memory**
- Edit script to test smaller architectures only
- Reduce batch size (edit script to use mini-batches)

**Q: Takes too long**
- Reduce `max_iter` from 1000 to 500
- Test fewer configurations (edit script)
- Use faster GPU
- **Interrupt anytime** - incremental file has results so far

**Q: All configurations have similar AUROC**
- Task may be relatively easy/linear
- Smaller networks likely sufficient
- Consider simpler models (linear probe)

**Q: Validation loss is noisy**
- Increase validation set size (edit script)
- Average over multiple runs
- Use more conservative early stopping

## Advanced: Custom Configurations

Edit `mlp_hyperparameter_search.py` to test your own configurations:

```python
# Add custom configuration to test
configs_to_test.append({
    "name": "my_custom_config",
    "hidden_layer_sizes": (10000, 5000, 2500),
    "learning_rate": 0.0003,
    "weight_decay": 5e-3,
    "dropout": 0.25,
    "patience": 15,
    "max_iter": 1000,
    "lr_scheduler": "plateau"
})
```

## Summary

This script helps you:
- ✅ Find optimal MLP settings systematically
- ✅ Understand why bigger networks underperform
- ✅ Tune regularization and learning rates
- ✅ Compare many configurations efficiently
- ✅ Document best practices for your specific task

Use it when default settings don't work well and you need principled hyperparameter tuning.

