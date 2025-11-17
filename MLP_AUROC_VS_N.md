# MLP AUROC vs Training Set Size Experiment

## Overview

This experiment measures how the MLP probe's AUROC varies with the training set size, similar to the existing linear probe experiment but using a neural network classifier instead.

## What It Does

For different training set sizes (N = 16, 32, 64, 128, 256, 512, 1024, 1800):
1. Randomly samples N examples for training
2. Uses remaining examples for testing
3. Trains an MLP probe on the N training examples
4. Measures test AUROC
5. Repeats 10 times with different random seeds
6. Computes mean and standard deviation

**Output**: Plot showing Error (1 - AUROC) vs Training Set Size

## Usage

### Run the Experiment

```bash
# Run probe analysis with MLP auroc_vs_n experiment
python probe_analysis.py \
    --layer 20 \
    --position letter \
    --num-examples 2000 \
    --prompt semimalign \
    --experiments auroc_vs_n_mlp

# Or run both linear and MLP versions
python probe_analysis.py \
    --layer 20 \
    --position letter \
    --num-examples 2000 \
    --prompt semimalign \
    --experiments auroc_vs_n auroc_vs_n_mlp
```

### Output Files

- **`auroc_vs_n_mlp_layer{L}_pos-{P}_n{N}_filtered.png`**: Plot of MLP error vs training size
- Console output showing error rates at each N

Example output:
```
==============================================================
Experiment 5b: AUROC vs Training Set Size (MLP Probe)
==============================================================
Using Residual MLP with architecture: (12000,)
Training Residual MLP probes with 16 examples
  Trial 1/10: AUROC = 0.7234
  Trial 2/10: AUROC = 0.7456
  ...
Training Residual MLP probes with 32 examples
  ...
  N=16: Error = 0.2891 ± 0.0234
  N=32: Error = 0.2345 ± 0.0189
  N=64: Error = 0.1987 ± 0.0156
  N=128: Error = 0.1678 ± 0.0134
  N=256: Error = 0.1456 ± 0.0112
  N=512: Error = 0.1289 ± 0.0098
  N=1024: Error = 0.1167 ± 0.0087
  N=1800: Error = 0.1089 ± 0.0079
```

## Configuration Options

The experiment is highly configurable through `measure_auroc_vs_training_size_mlp()`:

```python
results_mlp = measure_auroc_vs_training_size_mlp(
    activations, labels,
    n_values=None,                    # Auto-determined if None
    n_trials=10,                      # Number of trials per N
    hidden_layer_sizes=(12000,),      # MLP architecture
    max_iter=1000,                    # Max training epochs
    learning_rate=0.001,              # Learning rate
    weight_decay=1e-4,                # L2 regularization
    dropout=0.1,                      # Dropout rate
    patience=10,                      # Early stopping patience
    lr_scheduler=None,                # LR scheduler ("plateau", "step", None)
    use_constant_residual=False,      # Use constant residual MLP
    random_state=42                   # Random seed
)
```

### Customizing Architecture

Edit `probe_analysis.py` to change the MLP architecture:

```python
results_mlp = measure_auroc_vs_training_size_mlp(
    activations, labels,
    hidden_layer_sizes=(8000,),       # Smaller network
    # or
    hidden_layer_sizes=(12000, 6000), # Two-layer network
    # or
    use_constant_residual=True,       # Use constant residual architecture
    ...
)
```

### Customizing N Values

By default, tests N = 16, 32, 64, 128, 256, 512, 1024, ~1800 (90% of data)

To customize:

```python
results_mlp = measure_auroc_vs_training_size_mlp(
    activations, labels,
    n_values=[50, 100, 200, 500, 1000],  # Custom N values
    ...
)
```

## Comparison: Linear vs MLP

Run both experiments to compare:

```bash
python probe_analysis.py \
    --layer 20 \
    --position letter \
    --num-examples 2000 \
    --experiments auroc_vs_n auroc_vs_n_mlp
```

This produces two plots:
- `auroc_vs_n_layer20_pos-letter_n2000_filtered.png` (Linear)
- `auroc_vs_n_mlp_layer20_pos-letter_n2000_filtered.png` (MLP)

### Expected Differences

**Linear Probe:**
- Faster to train (seconds per trial)
- More sample efficient (good AUROC with fewer samples)
- Lower capacity - may plateau earlier
- Error curve: smooth decrease

**MLP Probe:**
- Slower to train (minutes per trial)  
- Requires more samples to train effectively
- Higher capacity - may achieve lower error with enough data
- Error curve: may have higher variance at low N, lower error at high N

### Typical Pattern

```
Error (1 - AUROC)
     |
0.30 |  L\
     |    \M
0.25 |     L\
     |       \M
0.20 |        L\M
     |          \M
0.15 |           L\M
     |             \\
0.10 |              LM
     |___________________
       16  64  256  1024
           Training Size N

L = Linear Probe
M = MLP Probe
```

At low N: Linear often better (more sample efficient)
At high N: MLP may catch up or surpass (higher capacity)

## Implementation Details

### Functions Added

**`lib/probes.py`:**
- `measure_auroc_vs_training_size_mlp()`: MLP equivalent of linear probe experiment
- Updated `train_mlp_probe()`: Now accepts all hyperparameters

**`probe_analysis.py`:**
- Experiment 5b: AUROC vs Training Set Size (MLP Probe)
- Runs after Experiment 5 (linear version)

### Code Flow

1. `measure_auroc_vs_training_size_mlp()` called with data
2. For each N in n_values:
   - For each trial:
     - Random split: N train, rest test
     - Call `train_mlp_probe()` with specified hyperparameters
     - Record AUROC
   - Store all AUROCs for this N
3. Return dict: `{N: [auroc1, auroc2, ..., auroc10]}`
4. Plot using `plot_auroc_vs_training_size()` (same as linear)

## Use Cases

### 1. **Sample Efficiency Analysis**

How many examples does MLP need to match linear probe performance?

```bash
python probe_analysis.py --experiments auroc_vs_n auroc_vs_n_mlp ...
```

Compare the plots to see crossover point.

### 2. **Architecture Comparison**

Does deeper network require more samples?

```python
# Test single layer
results_1layer = measure_auroc_vs_training_size_mlp(
    ..., hidden_layer_sizes=(12000,)
)

# Test three layers
results_3layer = measure_auroc_vs_training_size_mlp(
    ..., hidden_layer_sizes=(12000, 6000, 3000)
)
```

### 3. **Regularization Impact**

Does higher regularization help at low N?

```python
# Low regularization
results_low_reg = measure_auroc_vs_training_size_mlp(
    ..., weight_decay=1e-4, dropout=0.1
)

# High regularization
results_high_reg = measure_auroc_vs_training_size_mlp(
    ..., weight_decay=1e-2, dropout=0.3
)
```

### 4. **Constant vs Standard Residual**

Which architecture is more sample efficient?

```python
# Standard residual
results_std = measure_auroc_vs_training_size_mlp(
    ..., use_constant_residual=False
)

# Constant residual
results_const = measure_auroc_vs_training_size_mlp(
    ..., use_constant_residual=True
)
```

## Tips

### Faster Execution

Reduce trials for quicker results:

```python
results_mlp = measure_auroc_vs_training_size_mlp(
    ...,
    n_trials=5,  # Instead of 10
    max_iter=500  # Instead of 1000
)
```

### GPU Acceleration

The MLP automatically uses GPU if available. Check output:
```
Using device: cuda
GPU: NVIDIA A100-SXM4-80GB
```

### Progress Monitoring

The experiment prints progress for each trial:
```
Training Residual MLP probes with 128 examples
  Trial 1/10: AUROC = 0.8234
  Trial 2/10: AUROC = 0.8156
  ...
```

### Interpreting Results

**High variance at low N**: Normal - small training sets are unstable
**Error not decreasing**: May need more regularization or different architecture
**MLP worse than linear everywhere**: Task may not benefit from non-linearity
**MLP better at high N**: Task benefits from higher capacity

## Troubleshooting

**Q: Training is very slow**
- Reduce `max_iter` (e.g., 500 instead of 1000)
- Reduce `n_trials` (e.g., 5 instead of 10)
- Use smaller architecture (e.g., `(8000,)` instead of `(12000,)`)
- Check GPU is being used

**Q: High error even at large N**
- Try different hyperparameters (lower LR, higher regularization)
- Try different architecture
- Check if linear probe also has high error (may be task difficulty)

**Q: Out of memory**
- Use smaller architecture
- Reduce batch size in classifier (requires code edit)

**Q: Results don't match between trials**
- Increase `n_trials` for more stable estimates
- Check if early stopping is too aggressive (increase `patience`)

## Related Experiments

- **Experiment 5**: Linear probe version of this experiment
- **Experiment 6**: Label corruption robustness (tests noise tolerance)
- **Hyperparameter Search**: `mlp_hyperparameter_search.py` for finding best MLP config

## Example Analysis Workflow

1. **Baseline**: Run linear probe auroc_vs_n
   ```bash
   python probe_analysis.py --experiments auroc_vs_n ...
   ```

2. **MLP comparison**: Run MLP version
   ```bash
   python probe_analysis.py --experiments auroc_vs_n_mlp ...
   ```

3. **Find best config**: Run hyperparameter search
   ```bash
   python mlp_hyperparameter_search.py --layer 20 --position letter
   ```

4. **Re-test with best config**: Update probe_analysis.py with best hyperparameters, re-run
   
5. **Compare plots**: Determine if MLP provides benefit and at what sample size

## Summary

The MLP AUROC vs N experiment:
- ✅ **Measures sample efficiency** of MLP probes
- ✅ **Compares to linear probes** using same experimental setup
- ✅ **Highly configurable** - architecture, hyperparameters, trial count
- ✅ **Integrated into probe_analysis.py** - easy to run
- ✅ **Compatible with all token positions and layers**

Use it to understand when and how much data MLPs need to outperform simpler linear probes!

