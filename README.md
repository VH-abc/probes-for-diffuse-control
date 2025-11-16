# Probes for Diffuse Control (Rewritten)

A clean, well-structured toolkit for probing whether language models internally know when they are correct or incorrect.

## Overview

This codebase tests whether neural language models have internal representations that encode their own correctness. The key hypothesis: **If we can train a linear probe to predict whether the model answered correctly from its internal activations, then the model "knows" when it's right or wrong.**

### Methodology

1. **Generate** completions for MMLU multiple-choice questions
2. **Extract** neural activations at specific layers and token positions
3. **Train** linear probes to predict correctness from activations
4. **Evaluate** probe performance using AUROC and robustness experiments

## Key Features

✨ **Clean Architecture**: Modular library design with clear separation of concerns

🔧 **Token Position Support**: Extract activations from first, last, middle, or all tokens

🐛 **Bug Fixes**: 
- Fixed anomaly detection (now correctly treats incorrect as anomalies)
- Fixed hardcoded model names
- Unified configuration across all modules

🧪 **Unit Tests**: Comprehensive test coverage with sanity checks

⚡ **Performance**: Concurrent API requests and multi-GPU parallelism

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download MMLU Data

```bash
python3 -c "from datasets import load_dataset; dataset = load_dataset('cais/mmlu', 'all'); dataset['auxiliary_train'].to_json('mmlu_data/train.json')"
```

### 3. Configure

Edit `config.py` to set your model and VLLM server parameters:

```python
MODEL_NAME = "google/gemma-3-12b-it"
MODEL_SHORT_NAME = "gemma-3-12b"
VLLM_BASE_PORT = 8100
VLLM_NUM_SERVERS = 8
```

### 4. Start VLLM Servers

```bash
bash vllm_launcher.sh
```

This launches multiple VLLM servers (one per GPU) for fast parallel inference.

### 5. Cache Activations

```bash
# Cache activations using unified API (multiple layers/positions at once)
python3 cache_activations_unified.py --prompt benign

# Or specify parameters
python3 cache_activations_unified.py --prompt benign --layers 0 10 20 30 --positions last first --num-examples 500
```

### 6. Run Probe Analysis

```bash
# Analyze cached activations
python3 probe_analysis.py

# Or specify parameters
python3 probe_analysis.py --layer 15 --position first --num-examples 500
```

### 7. Sweep Across Layers/Positions

```bash
# Sweep arbitrary (layer, position) pairs
python3 sweep.py --pairs 10,last 12,first 13,middle 14,last

# Sweep across multiple layers at one position
python3 sweep.py --layers 10 12 14 16 --position last

# Sweep across token positions at one layer
python3 sweep.py --positions last first middle --layer 13

# Quick sweep (layers 10-16 at last token)
python3 sweep.py --quick
```

## Project Structure

```
rewritten/
├── config.py                      # Central configuration
├── cache_activations_unified.py   # Cache activations from VLLM (unified)
├── probe_analysis.py               # Probe training and evaluation
├── sweep.py                        # Sweep layers/positions
├── vllm_launcher.sh                # Launch VLLM servers
├── requirements.txt                # Python dependencies
├── lib/                            # Core library
│   ├── __init__.py
│   ├── data.py                    # MMLU data loading
│   ├── generation.py              # VLLM generation
│   ├── activations.py             # Activation extraction
│   ├── probes.py                  # Probe training
│   └── visualization.py           # Plotting utilities
└── tests/                          # Unit tests
    ├── test_data.py
    ├── test_activations.py
    └── test_probes.py
```

## Configuration

All settings are in `config.py`:

### Model Configuration
- `MODEL_NAME`: Full HuggingFace model identifier
- `MODEL_SHORT_NAME`: Short name for directories/files

### VLLM Configuration
- `VLLM_BASE_PORT`: Starting port for VLLM servers
- `VLLM_NUM_SERVERS`: Number of servers/GPUs
- `VLLM_MAX_MODEL_LEN`: Maximum sequence length
- `VLLM_GPU_MEMORY_UTILIZATION`: GPU memory fraction to use

### Experiment Configuration
- `DEFAULT_LAYER`: Default layer to probe
- `DEFAULT_NUM_EXAMPLES`: Default number of MMLU examples
- `DEFAULT_TOKEN_POSITION`: Token position (e.g. "last", "first", "middle", "all")
- `TEMPERATURE`: Sampling temperature (1.0 = default)

## Token Position Support

One of the key improvements in this rewrite is full support for different token positions:

- **`last`**: Extract from the last non-special token (default)
- **`first`**: Extract from the first non-special token
- **`middle`**: Extract from the middle non-special token
- **`all`**: Average activations over all non-special tokens

This allows you to test which token position best captures the model's "knowledge" of correctness.

## Experiments

The probe analysis runs several experiments:

### 1. Linear Probe
Trains a logistic regression probe to predict correctness from activations.
- **Metric**: AUROC (Area Under ROC Curve)
- **Interpretation**: AUROC > 0.5 indicates the model has some internal signal about correctness

### 2. PCA Visualization
Projects activations to 2D using PCA to visualize separation between correct/incorrect.

### 3. Anomaly Detection
Fits a Gaussian to correct answers, detects incorrect as anomalies.
- **Fixed Bug**: Now correctly treats incorrect (class 0) as anomalies, not correct (class 1)

### 4. AUROC vs Training Set Size
Measures how probe performance scales with training data size.

### 5. Label Corruption Robustness
Tests probe robustness when training labels are corrupted.

## Running Tests

```bash
# Run all tests
python3 -m unittest discover tests/

# Run specific test module
python3 -m unittest tests.test_data
python3 -m unittest tests.test_activations
python3 -m unittest tests.test_probes
```

Tests include:
- Data loading and formatting
- Answer extraction
- Token position selection
- Probe training sanity checks
- Anomaly detection correctness

## Example Workflow

```bash
# 1. Start VLLM servers
bash vllm_launcher.sh

# Wait for servers to start (check with curl)
curl http://localhost:8100/v1/models

# 2. Quick layer sweep
python3 sweep.py --quick

# 3. Token position comparison at best layer
python3 sweep.py --positions last first middle --layer 13

# 4. Detailed analysis of best configuration
python3 probe_analysis.py --layer 13 --position last --num-examples 500
```

## Output Files

### Cached Activations
Files saved to `experiments/{MODEL_SHORT_NAME}/cached_activations/`:
- `mmlu_layer{XX}_pos-{position}_n{N}_activations.npy`: Activations
- `mmlu_layer{XX}_pos-{position}_n{N}_labels.npy`: Binary labels (1=correct, 0=incorrect)
- `mmlu_layer{XX}_pos-{position}_n{N}_metadata.json`: Run metadata

### Analysis Results
Files saved to `experiments/{MODEL_SHORT_NAME}/results/`:
- `roc_*.png`: ROC curves
- `scores_*.png`: Score distributions
- `pca_*.png`: PCA visualizations
- `auroc_*.json`: AUROC scores
- `*_summary.txt`: Comparison summaries

## Bug Fixes from Original

This rewrite fixes several issues in the original codebase:

1. **Anomaly Detection Bug**: Now correctly fits Gaussian to correct answers and detects incorrect as anomalies (was backwards)

2. **Hardcoded Model Names**: Removed hardcoded `"gemma"` in API calls, now uses `config.MODEL_SHORT_NAME`

3. **Config Inconsistencies**: Unified configuration - all modules now use `config.py`

4. **Temperature Inconsistency**: Now consistently uses `config.TEMPERATURE` (1.0) everywhere

5. **Missing Token Position Support**: Added full support for first/last/middle/all token positions

## Architecture Improvements

- **Modular Library Design**: Core functionality in `lib/` for easy reuse
- **Clear Separation**: Data, generation, activations, probes, visualization are separate modules
- **Type Hints**: Added type hints throughout for better code clarity
- **Unit Tests**: Comprehensive test coverage with sanity checks
- **Documentation**: Detailed docstrings and README

## Advanced Usage

### Custom Token Positions

```python
from lib.activations import get_probe_positions

# Get token positions programmatically
positions = get_probe_positions(
    input_ids, 
    special_token_ids, 
    token_position="middle"
)
```

### Custom Probe Experiments

```python
from lib.probes import train_linear_probe

# Train custom probe
auroc, clf, probs, fpr, tpr = train_linear_probe(
    X_train, y_train, X_test, y_test,
    max_iter=1000,
    random_state=42
)
```

### Batch Processing

```python
from lib.generation import generate_with_vllm_multi_server

# Generate completions in parallel
full_texts, completions = generate_with_vllm_multi_server(
    prompts=prompts,
    max_new_tokens=100,
    temperature=1.0,
    model_name="gemma-3-12b",
    num_servers=8,
    base_port=8100
)
```

## Troubleshooting

### VLLM Servers Won't Start
- Check GPU availability: `nvidia-smi`
- Verify model is downloaded: `huggingface-cli download {MODEL_NAME}`
- Check port conflicts: `netstat -tulpn | grep {PORT}`

### Out of Memory Errors
- Reduce `VLLM_GPU_MEMORY_UTILIZATION` in config.py
- Reduce `ACTIVATION_BATCH_SIZE` in config.py
- Use fewer servers: reduce `VLLM_NUM_SERVERS`

### Activation Extraction Fails
- Ensure model is cached: Run once with `use_model_cache=False`
- Check layer index is valid for your model
- Verify token position is valid

## Citation

If you use this code, please cite the original research.

## License

[Specify license]

## Contributing

Contributions welcome! Please:
1. Add unit tests for new features
2. Update documentation
3. Follow existing code style
4. Run tests before submitting

