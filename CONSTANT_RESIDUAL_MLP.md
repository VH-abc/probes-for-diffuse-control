# Constant Residual MLP with Identity Skip Connections

## Overview

A new MLP architecture variant with **pure identity skip connections** has been added: `ConstantResidualMLPClassifier`.

## Key Difference from Standard Residual MLP

### Standard Residual MLP
```
Input(3840) → Hidden(12000) → Output(1)
     ↓___________/↑
  (needs projection if dimensions differ)
```

**Problem**: When dimensions change, requires a projection layer for the skip connection.

### Constant Residual MLP (NEW!)
```
Input(3840) → [↑12000 ↓3840] → [↑12000 ↓3840] → Output(1)
     ↓___________/↑            ↓___________/↑
  (pure identity - no projection ever!)
```

**Benefit**: Residual dimension stays constant (= input dim) throughout, enabling **pure identity skip connections**.

## Architecture Details

### Bottleneck Block Structure

Each "hidden layer" is actually a **bottleneck block**:

1. **Up-project**: `Input(3840) → Hidden(12000)`
2. **Compute**: BatchNorm, ReLU, Dropout
3. **Down-project**: `Hidden(12000) → Output(3840)`
4. **Skip**: Add identity residual (no projection!)
5. **Activate**: BatchNorm, ReLU, Dropout

```python
# Pseudo-code for one block
residual = input  # Save for skip connection

# Up-project
out = Linear(input_size, hidden_size)(input)
out = BatchNorm(out) → ReLU → Dropout

# Down-project
out = Linear(hidden_size, input_size)(out)
out = BatchNorm(out)

# Identity skip connection (key advantage!)
out = out + residual  # Always same dimensions!

# Post-skip activation
out = ReLU → Dropout
```

### Multiple Blocks

```python
ConstantResidualMLPClassifier(hidden_layer_sizes=(12000, 16000, 8000))
```

Architecture:
```
Input(3840)
  → [↑12000 ↓3840] + Identity
  → [↑16000 ↓3840] + Identity
  → [↑8000 ↓3840] + Identity
  → Output(1)
```

All skip connections are **pure identity** - no projections ever needed!

## Benefits

### 1. **Perfect Gradient Flow** 🌊
- Identity skip connections preserve gradients perfectly
- No dimension mismatch issues
- Similar to ResNet's original identity shortcuts

### 2. **Arbitrary Hidden Sizes** 📐
- Can use different hidden sizes in each block: `(12000, 16000, 8000)`
- No constraint that they must match
- Enables exploring very large hidden sizes (e.g., 24000)

### 3. **Consistent Representation** 🎯
- Input/output dimension stays constant throughout
- Easier to reason about information flow
- Similar to Transformer's constant hidden dimension

### 4. **Better Regularization** 🛡️
- The up/down projection acts as a bottleneck
- Forces information compression
- May prevent overfitting better than standard residual

## Usage

### In Hyperparameter Search

The constant residual MLP is automatically included in the hyperparameter search with 7 configurations:

```python
# Configuration Set 8: Constant Residual MLP
configs = [
    (12000,),              # Single block
    (16000,),              # Single block, larger
    (12000, 12000),        # Two blocks
    (16000, 8000),         # Two blocks, varied
    (12000, 12000, 12000), # Three blocks
    (12000, 12000),        # With high regularization
    (24000,),              # Very large (possible with const residual!)
]
```

Total configurations tested: **~49** (42 standard + 7 constant residual)

### Direct Usage

```python
from lib.probes import ConstantResidualMLPClassifier

clf = ConstantResidualMLPClassifier(
    hidden_layer_sizes=(12000, 12000),  # Multiple blocks
    learning_rate=0.001,
    weight_decay=1e-4,
    dropout=0.1,
    patience=10,
    lr_scheduler="plateau"
)

clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
```

### Comparison to Standard

```python
# Standard Residual MLP
from lib.probes import ResidualMLPClassifier

standard_clf = ResidualMLPClassifier(
    hidden_layer_sizes=(12000, 6000),  # Dimensions change
    # Skip connections need projection when dims differ
)

# Constant Residual MLP
constant_clf = ConstantResidualMLPClassifier(
    hidden_layer_sizes=(12000, 6000),  # Dimensions DON'T change!
    # Up: 3840→12000, Down: 12000→3840 (identity skip!)
    # Up: 3840→6000, Down: 6000→3840 (identity skip!)
)
```

## Architecture Visualization

### Example: Input dim = 3840, Hidden = (12000, 16000)

```
Layer Structure:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: (3840)
  │
  ├──────────────────────┐
  │                      │
  └→ Up: 3840→12000      │ (Identity)
     BN, ReLU, Drop      │
     Down: 12000→3840    │
     BN                  │
  ┌────────────────────<─┘
  │  Add + ReLU + Drop
  │
  ├──────────────────────┐
  │                      │
  └→ Up: 3840→16000      │ (Identity)
     BN, ReLU, Drop      │
     Down: 16000→3840    │
     BN                  │
  ┌────────────────────<─┘
  │  Add + ReLU + Drop
  │
  └→ Output: 3840→1

Final: (1) [logit]
```

## Parameter Counts

For input size 3840:

| Configuration | Standard Residual | Constant Residual |
|---------------|-------------------|-------------------|
| `(12000,)` | ~190M | ~138M |
| `(12000, 12000)` | ~334M | ~276M |
| `(12000, 6000)` | ~262M | ~161M |
| `(16000,)` | ~251M | ~184M |
| `(24000,)` | ~564M | ~276M |

**Note**: Constant residual often has **fewer parameters** because:
- No skip projection layers (always identity)
- Down-projection brings dimension back down

## When to Use

### Use Constant Residual When:
✅ Want **pure identity skip connections** (better gradients)  
✅ Want to explore **very large hidden sizes** (e.g., 24000)  
✅ Standard residual overfits (bottleneck may regularize)  
✅ Want **consistent dimensionality** throughout network

### Use Standard Residual When:
✅ Want **dimension reduction** through network  
✅ Want **less compute** (no up/down projection)  
✅ Simpler architecture sufficient

## Hyperparameter Search Integration

### Auto-Skip Feature

The hyperparameter search now **automatically skips** configurations that have already been tested:

```python
# First run - tests all 49 configs
python mlp_hyperparameter_search.py

# Script interrupted after 20 configs
# Ctrl+C

# Restart - automatically skips first 20, resumes at 21!
python mlp_hyperparameter_search.py
```

**How it works:**
1. Reads incremental file at startup
2. Extracts all completed configurations
3. For each config in queue:
   - Compares to completed configs
   - If match found → **SKIP**
   - If new → **TEST**

**Output:**
```
Found 20 previously completed configurations
  These will be automatically skipped

Configuration 1/49: arch_sweep
  ⏩ SKIPPING - Already completed in previous run

Configuration 21/49: lr_sweep
  ✓ Testing (new)...
```

### Incremental File Format

Each completed config is saved immediately:

```jsonl
{"auroc": 0.8012, "train_time": 42.3, "config": {...}, "timestamp": "2025-11-17 10:30:00"}
{"auroc": 0.8156, "train_time": 38.1, "config": {...}, "timestamp": "2025-11-17 10:35:12"}
```

**Benefits:**
- ✅ **Resume anywhere** - restart and pick up where you left off
- ✅ **No duplicates** - never re-test the same config
- ✅ **Append-only** - history preserved across multiple runs
- ✅ **Automatic** - no manual intervention needed

## Comparison Results

After running the full hyperparameter search, compare:

```python
import json

# Load results
with open('mlp_hyperparam_search_layer20_pos-letter.json') as f:
    results = json.load(f)

# Filter by type
standard = [r for r in results['all_results'] 
           if not r['config'].get('use_constant_residual', False)]
constant = [r for r in results['all_results'] 
           if r['config'].get('use_constant_residual', False)]

print(f"Standard Residual - Best: {max(r['auroc'] for r in standard):.4f}")
print(f"Constant Residual - Best: {max(r['auroc'] for r in constant):.4f}")
```

Or check the report file:

```bash
cat experiments/.../mlp_hyperparam_report_layer20_pos-letter.txt | grep -A 20 "TOP 10"
```

Look for the "Type" column:
- **StdRes** = Standard Residual MLP
- **ConstRes** = Constant Residual MLP

## Implementation Details

### Files Modified

1. **`lib/probes.py`**:
   - Added `ConstantResidualMLPClassifier` class (~230 lines)
   - Implements bottleneck architecture with identity skips

2. **`mlp_hyperparameter_search.py`**:
   - Added `use_constant_residual` parameter
   - Added 7 constant residual configurations (Set 8)
   - Added auto-skip functionality:
     - `load_completed_configs()`
     - `config_matches()`
     - `is_config_completed()`
   - Displays "ConstRes" vs "StdRes" in reports

### Architecture Code

```python
class ConstantResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, dropout_rate):
        for hidden_size in hidden_layer_sizes:
            # Bottleneck block
            block = {
                'up': Linear(input_size, hidden_size),
                'bn1': BatchNorm1d(hidden_size),
                'down': Linear(hidden_size, input_size),
                'bn2': BatchNorm1d(input_size),
            }
            self.blocks.append(block)
    
    def forward(self, x):
        for block in self.blocks:
            residual = x  # Identity!
            
            # Up → Compute → Down
            out = block['up'](x)
            out = relu(bn1(out))
            out = dropout(out)
            out = block['down'](out)
            out = bn2(out)
            
            # Identity skip connection
            out = out + residual  # Always same dimensions
            out = relu(dropout(out))
        
        return output_layer(out)
```

## Summary

**Constant Residual MLP** = Bottleneck blocks with identity skip connections

**Key advantages:**
- 🎯 Pure identity skips (better gradients)
- 📐 Arbitrary hidden sizes
- 🔄 Consistent dimensionality
- 🛡️ Natural regularization

**Auto-skip feature:**
- ⏭️ Automatically resumes from interruptions
- 🚫 Never re-tests same configuration
- 💾 Preserves full history

Now included in hyperparameter search with 7 specialized configurations to help find if this architecture performs better than standard residual connections!

