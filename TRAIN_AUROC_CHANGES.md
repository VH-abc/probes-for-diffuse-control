# Train AUROC Added to Linear Probe AUROC_vs_N Plot

## Summary

Added training set AUROC to the linear probe AUROC vs training set size plot. This allows you to visualize both training and test performance to detect overfitting/underfitting.

## Changes Made

### 1. `lib/probes.py` - Modified `measure_auroc_vs_training_size()`

**Before:**
- Returned single dictionary of test error rates

**After:**
- Returns tuple: `(test_results, train_results)`
- Both dictionaries map training sizes → lists of error rates (1 - AUROC)
- Computes train AUROC by evaluating the trained probe on its training set

```python
# Old behavior
results = measure_auroc_vs_training_size(X, y, ...)
for n, errors in results.items():
    print(f"N={n}: Error={np.mean(errors)}")

# New behavior  
test_results, train_results = measure_auroc_vs_training_size(X, y, ...)
for n in test_results.keys():
    test_err = np.mean(test_results[n])
    train_err = np.mean(train_results[n])
    print(f"N={n}: Test Error={test_err}, Train Error={train_err}")
```

### 2. `lib/visualization.py` - Updated `plot_auroc_vs_training_size()`

**Before:**
- Plotted only test error curve

**After:**
- Added optional `train_results` parameter (default: None)
- When `train_results` is provided, plots both train and test curves
- Train curve shown with dashed line and square markers
- Legend added to distinguish curves
- Backward compatible: works without `train_results` parameter

```python
# With train AUROC
plot_auroc_vs_training_size(
    test_results,
    output_path,
    title="Linear Probe Error vs Training Set Size",
    train_results=train_results  # NEW: optional parameter
)

# Without train AUROC (backward compatible)
plot_auroc_vs_training_size(
    test_results,
    output_path,
    title="Linear Probe Error vs Training Set Size"
)
```

### 3. `probe_analysis.py` - Updated Experiment 5

**Before:**
```python
results_linear = measure_auroc_vs_training_size(...)
plot_auroc_vs_training_size(results_linear, ...)
```

**After:**
```python
results_linear, train_results_linear = measure_auroc_vs_training_size(...)

print("\n  Test Set Performance:")
for n, errors in results_linear.items():
    print(f"    N={n}: Error = {np.mean(errors):.4f} ± {np.std(errors):.4f}")

print("\n  Train Set Performance:")
for n, errors in train_results_linear.items():
    print(f"    N={n}: Error = {np.mean(errors):.4f} ± {np.std(errors):.4f}")

plot_auroc_vs_training_size(
    results_linear, 
    output_path,
    title="Linear Probe Error vs Training Set Size",
    train_results=train_results_linear
)
```

## Usage

Run probe analysis as usual:

```bash
python probe_analysis.py \
    --prompt-name lie_detector \
    --layer 30 \
    --token-position letter \
    --num-examples 100000 \
    --experiments auroc_vs_n \
    --filter-reliable
```

The generated plot `auroc_vs_n_layer30_pos-letter_n100000_filtered.png` will now show:
- **Solid line with circles**: Test Error (1 - AUROC)
- **Dashed line with squares**: Train Error (1 - AUROC)

## Interpreting the Plot

### Normal Behavior
- **Train error < Test error**: Expected - probe generalizes reasonably
- **Gap increases with N**: More capacity to fit training data
- **Both decrease with N**: Model gets better with more data

### Overfitting
- **Large gap** between train and test error
- Train error near 0, test error high
- May indicate need for regularization

### Underfitting
- **Both errors high** even with large N
- Little gap between train and test
- May indicate need for more model capacity

## Example Output

```
Experiment 5: AUROC vs Training Set Size (Linear Probe)
================================================================

Training linear probes with 16 examples
Training linear probes with 32 examples
Training linear probes with 64 examples
...

  Test Set Performance:
    N=16: Error = 0.3245 ± 0.0123
    N=32: Error = 0.2876 ± 0.0098
    N=64: Error = 0.2134 ± 0.0076
    ...

  Train Set Performance:
    N=16: Error = 0.2456 ± 0.0089
    N=32: Error = 0.1987 ± 0.0067
    N=64: Error = 0.1234 ± 0.0045
    ...
```

## Backward Compatibility

✓ Old code that doesn't unpack the tuple will break (intentional - forces update)
✓ Plotting function remains backward compatible (train_results is optional)
✓ MLP probe AUROC vs N unchanged (only linear probe updated)

## Technical Details

Train AUROC computation:
```python
# After training probe on X_train, y_train
y_train_pred_proba = clf.predict_proba(X_train)[:, 1]
auroc_train = roc_auc_score(y_train, y_train_pred_proba)
train_error = 1 - auroc_train
```

This measures how well the probe fits its training data, independent of test performance.


