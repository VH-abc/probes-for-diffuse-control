"""
Global configuration for probes-for-diffuse-control experiments.

This file contains all model-specific settings and paths.
To switch models, simply update MODEL_NAME and MODEL_SHORT_NAME.
"""

# Model configuration
MODEL_NAME = "google/gemma-3-12b-it"
MODEL_SHORT_NAME = "gemma-3-12b"  # Used for directory/file names

# VLLM server configuration
VLLM_BASE_PORT = 8100  # Changed from 8000 to avoid nginx on port 8001
VLLM_NUM_SERVERS = 8
VLLM_MAX_MODEL_LEN = 2000
VLLM_GPU_MEMORY_UTILIZATION = 0.5 #0.93

# Directory structure (model-specific)
BASE_DIR = "experiments"
CACHED_ACTIVATIONS_DIR = f"{BASE_DIR}/{MODEL_SHORT_NAME}/cached_activations"
RESULTS_DIR = f"{BASE_DIR}/{MODEL_SHORT_NAME}/results"

# Generation parameters
MAX_NEW_TOKENS = 100
TEMPERATURE = 1.0

# Experiment parameters
DEFAULT_LAYER = 13
DEFAULT_NUM_EXAMPLES = 200

def get_model_config():
    """Return a dictionary with all configuration settings."""
    return {
        "model_name": MODEL_NAME,
        "model_short_name": MODEL_SHORT_NAME,
        "vllm_base_port": VLLM_BASE_PORT,
        "vllm_num_servers": VLLM_NUM_SERVERS,
        "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
        "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "cached_activations_dir": CACHED_ACTIVATIONS_DIR,
        "results_dir": RESULTS_DIR,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "default_layer": DEFAULT_LAYER,
        "default_num_examples": DEFAULT_NUM_EXAMPLES,
    }

def print_config():
    """Print current configuration."""
    print("="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    config = get_model_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80)

if __name__ == "__main__":
    print_config()

