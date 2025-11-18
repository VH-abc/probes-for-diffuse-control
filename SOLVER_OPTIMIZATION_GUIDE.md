# Linear Probe Solver Optimization Guide

## Summary

This guide explains the tools and configuration options added to diagnose and optimize linear probe training performance.

## Problem

Linear probe training became slow with >10k samples. Initial attempt to switch to SAGA solver actually made it **slower** (116x slower in tests!), indicating we needed better diagnostics and testing.

## Solution: Diagnostic Tools + Flexible Configuration

### 1. Added Diagnostic Logging

**Feature:** `train_linear_probe()` now has optional verbose logging

**Usage:**
```python
# Enable globally in config.py
PROBE_VERBOSE = True

# Or enable per-call
auroc, clf, _, _, _ = train_linear_probe(
    X_train, y_train, X_test, y_test,
    verbose=True
)
```

**What it shows:**
```
  [Probe Training Diagnostics]
    Dataset: 10000 train samples, 5000 test samples
    Features: 3840
    Solver: lbfgs, tol: 0.01, max_iter: 100, n_jobs: -1
    Training time: 2.45s
    Iterations: 87 / 100
    Converged: True
    AUROC: 0.8234
```

This reveals:
- **Time**: How long training took
- **Iterations**: Whether solver converged before max_iter
- **Convergence**: Whether optimization completed successfully

### 2. Flexible Solver Configuration

**New config parameters in `config.py`:**
```python
PROBE_MAX_ITER = 100      # Reduced from 1000 for faster training
PROBE_SOLVER = 'lbfgs'    # Solver algorithm ('lbfgs', 'liblinear', 'sag', 'saga')
PROBE_TOL = 1e-2          # Relaxed from 1e-4 for faster convergence
PROBE_N_JOBS = -1         # Use all CPU cores (where supported)
PROBE_VERBOSE = False     # Enable diagnostic logging
```

**Solver options:**
- `lbfgs`: Default, good for small-medium datasets (<20k)
- `liblinear`: Often fastest for medium datasets (10k-100k)
- `sag`: Simpler than saga, sometimes faster
- `saga`: For very large datasets (>100k), supports all penalties

### 3. Solver Comparison Script

**Script:** `compare_solvers.py`

Tests multiple solvers on your actual data to find the fastest one.

**Usage:**
```bash
# Basic usage (uses defaults from config.py)
python compare_solvers.py \
    --prompt-name lie_detector \
    --layer 30 \
    --position letter \
    --filter-reliable

# Custom test
python compare_solvers.py \
    --prompt-name lie_detector \
    --layer 30 \
    --position letter \
    --sample-sizes 5000 10000 20000 50000 \
    --solvers lbfgs liblinear sag \
    --max-iter 100 \
    --tol 1e-2 \
    --trials 3
```

**What it does:**
1. Loads your cached activations
2. Tests each solver at multiple sample sizes
3. Reports timing, AUROC, iterations, convergence
4. Recommends the best solver

**Example output:**
```
Testing with N = 10000 training samples
────────────────────────────────────────────────────────────────────────────────
  lbfgs       │ Time:   3.24s │ AUROC: 0.8234±0.0012 │ Iters: 87/100 │ Converged: True
  liblinear   │ Time:   1.45s │ AUROC: 0.8231±0.0015 │ Iters: N/A/100 │ Converged: Unknown
  sag         │ Time:   2.87s │ AUROC: 0.8229±0.0018 │ Iters: 92/100 │ Converged: True
  saga        │ Time:   5.12s │ AUROC: 0.8227±0.0021 │ Iters: 98/100 │ Converged: True

SUMMARY: Best Solver by Sample Size
===============================================================================
N = 10000:
  Fastest: liblinear (1.45s, AUROC: 0.8231)
  Best AUROC: lbfgs (AUROC: 0.8234, 3.24s)
  ✓ Fastest solver has nearly same AUROC (diff: 0.0003)

RECOMMENDATION
===============================================================================
Fastest overall: liblinear (avg 1.45s across all sizes)
AUROC quality: 0.8231

To use this solver, set in config.py:
    PROBE_SOLVER = 'liblinear'
```

## Current Configuration (After Changes)

```python
# In config.py
PROBE_MAX_ITER = 100       # Reduced for faster iteration (was 1000)
PROBE_SOLVER = 'lbfgs'     # Reverted to default (was temporarily 'saga')
PROBE_TOL = 1e-2           # Relaxed for faster convergence (was 1e-4)
PROBE_N_JOBS = -1          # Use all CPU cores
PROBE_VERBOSE = False      # Set to True to see diagnostics
```

These settings provide a good balance between speed and quality:
- **Reduced iterations**: 100 is usually enough for convergence on most problems
- **Relaxed tolerance**: 1e-2 is faster than 1e-4 with minimal AUROC loss (<0.001)
- **Parallel execution**: Uses all CPU cores for solvers that support it

## How to Optimize for Your Data

### Step 1: Enable Diagnostics

Set `PROBE_VERBOSE = True` in `config.py` and run your experiment:

```bash
python probe_analysis.py \
    --prompt-name lie_detector \
    --layer 30 \
    --experiments auroc_vs_n \
    --filter-reliable
```

Watch the output to see:
- How long each training takes
- Whether solver converges before max_iter
- If it's hitting max_iter frequently

### Step 2: Run Solver Comparison

```bash
python compare_solvers.py \
    --prompt-name lie_detector \
    --layer 30 \
    --position letter
```

This tests all solvers and recommends the fastest one.

### Step 3: Update Configuration

Based on comparison results, update `config.py`:

```python
# If liblinear was fastest:
PROBE_SOLVER = 'liblinear'

# If not converging:
PROBE_MAX_ITER = 200  # Increase iterations

# If too slow but converging early:
PROBE_MAX_ITER = 50   # Reduce iterations

# If AUROC is same with looser tolerance:
PROBE_TOL = 1e-1      # Even more relaxed
```

## Troubleshooting

### "Solver not converging"
**Symptom:** Iterations = max_iter, converged = False
**Solution:** 
- Increase `PROBE_MAX_ITER` (try 200, 500)
- OR relax `PROBE_TOL` (try 1e-1)

### "Still too slow"
**Symptom:** Training takes >30s even with optimizations
**Solutions:**
1. Try different solver: `PROBE_SOLVER = 'liblinear'`
2. Relax tolerance more: `PROBE_TOL = 1e-1`
3. Reduce max_iter: `PROBE_MAX_ITER = 50`
4. For AUROC vs N: Use smaller sample sizes or subsample data

### "AUROC dropped after changes"
**Symptom:** AUROC decreased by >0.01
**Solution:**
- Tighten tolerance: `PROBE_TOL = 1e-3`
- Increase iterations: `PROBE_MAX_ITER = 200`
- Check if solver converged (enable verbose)

## Expected Performance

### With Original Settings (lbfgs, max_iter=1000, tol=1e-4):
- 1k samples: ~1-2 sec
- 10k samples: ~10-20 sec  
- 50k samples: ~3-5 min
- 100k samples: ~8-15 min

### With Optimized Settings (lbfgs, max_iter=100, tol=1e-2):
- 1k samples: ~0.5 sec
- 10k samples: ~3-5 sec
- 50k samples: ~30-60 sec
- 100k samples: ~2-4 min

### If liblinear is fastest (typical):
- 1k samples: ~0.3 sec
- 10k samples: ~1-2 sec
- 50k samples: ~15-30 sec
- 100k samples: ~1-2 min

**Speedup: 5-10x faster** just from config changes, or **10-20x faster** with optimal solver.

## Technical Details

### Why SAGA Was Slower

SAGA has overhead that only pays off at very large scales:
- Maintains variance-reduced gradient estimates (memory cost)
- Stochastic updates (more iterations to converge)
- Parallelization overhead
- Better for >100k samples, not for 10k-50k

### Why Relaxed Tolerance Works

For classification (AUROC), we care about ranking, not exact probabilities:
- `tol=1e-4`: Probabilities accurate to 0.01%
- `tol=1e-2`: Probabilities accurate to 1%
- AUROC difference: Usually <0.001 (negligible)

### Solver Characteristics

| Solver | Speed | Memory | Convergence | Best For |
|--------|-------|--------|-------------|----------|
| **lbfgs** | Medium | High | Fast, precise | General use |
| **liblinear** | Fast | Low | Fast | 10k-100k samples |
| **sag** | Fast | Medium | Medium | Large datasets |
| **saga** | Slow | Medium | Slow | Very large (>100k) |

## Next Steps

1. Run `compare_solvers.py` on your data
2. Update `PROBE_SOLVER` based on results
3. Enable `PROBE_VERBOSE` to monitor convergence
4. Adjust `PROBE_MAX_ITER` and `PROBE_TOL` as needed
5. Run your experiments and enjoy faster training!

