# Probes for Diffuse Control

Experiments to probe whether language models internally know when they are correct or incorrect.

## Configuration

All model-specific settings are centralized in `config.py`. To switch models:

1. Edit `config.py` and update `MODEL_NAME` and `MODEL_SHORT_NAME`
2. Run scripts normally - they will automatically use the new configuration

```python
# Example config.py
MODEL_NAME = "google/gemma-3-1b-it"  # Full model name
MODEL_SHORT_NAME = "gemma-3-1b"      # Short name for directories
```

## Directory Structure

The system organizes experiments by model:

```
experiments/
├── gemma-3-1b/
│   ├── cached_activations/   # Cached neural activations
│   └── results/               # Analysis plots and outputs
├── gemma-3-12b/
│   ├── cached_activations/
│   └── results/
└── ...
```

## Quick Start

### 1. Start VLLM Servers

The VLLM servers read configuration from `config.py`:

```bash
bash vllm.bash
```

This will start multiple VLLM servers (one per GPU) on consecutive ports starting from `VLLM_BASE_PORT`.

### 2. Cache Activations

Generate completions and cache activations:

```bash
python cache_activations.py
```

All parameters default to `config.py` settings. Activations are saved to `experiments/{MODEL_SHORT_NAME}/cached_activations/`.

### 3. Run Analysis

Analyze the cached activations:

```bash
python probe_slack.py
```

Results are saved to `experiments/{MODEL_SHORT_NAME}/results/`.

## Configuration Reference

See `config.py` for all available settings:

- `MODEL_NAME`: Full HuggingFace model identifier
- `MODEL_SHORT_NAME`: Short name used for directory structure
- `VLLM_BASE_PORT`: Starting port for VLLM servers (default: 8000)
- `VLLM_NUM_SERVERS`: Number of VLLM servers/GPUs to use (default: 8)
- `VLLM_MAX_MODEL_LEN`: Maximum sequence length for VLLM
- `VLLM_GPU_MEMORY_UTILIZATION`: GPU memory to use (default: 0.93)
- `MAX_NEW_TOKENS`: Maximum tokens to generate per prompt
- `TEMPERATURE`: Sampling temperature (0.0 = greedy)
- `DEFAULT_LAYER`: Default layer to probe (default: 13)
- `DEFAULT_NUM_EXAMPLES`: Default number of MMLU examples (default: 200)

## Switching Models

To switch to a different model:

1. Update `config.py`:
```python
MODEL_NAME = "google/gemma-3-12b-it"
MODEL_SHORT_NAME = "gemma-3-12b"
```

2. Restart VLLM servers:
```bash
# Stop existing servers (Ctrl+C)
bash vllm.bash  # Starts with new model
```

3. Run experiments:
```bash
python cache_activations.py  # Uses new model automatically
python probe_slack.py         # Analyzes new model's activations
```

All outputs will be stored in `experiments/gemma-3-12b/` automatically.

## Files

- `config.py`: Central configuration file
- `cache_activations.py`: Generate and cache activations
- `probe_slack.py`: Analyze activations with probes
- `vllm.bash`: Start VLLM API servers
- `download_mmlu.py`: Download MMLU dataset
- `evaluate_gemma_mmlu.py`: Evaluate model on MMLU

## Experiment Details

See the main [evaluation README](README_evaluation.md) for details on the MMLU evaluation experiments.

