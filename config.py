"""
Central configuration for probes-for-diffuse-control.

All model-specific settings and paths are defined here.
To switch models, update MODEL_NAME and MODEL_SHORT_NAME.
"""

# Model configuration
MODEL_NAME = "google/gemma-3-12b-it"
MODEL_SHORT_NAME = "gemma-3-12b"  # Used for directory/file names

# VLLM server configuration
VLLM_BASE_PORT = 8100
VLLM_MAX_MODEL_LEN = 2000
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_NUM_SERVERS = 1
VLLM_GPUS = [0]  # GPUs for VLLM servers (e.g., [0,1,2,3] or list(range(4)))
ACTIVATION_GPUS = list(range(1, 4))  # GPUs for activation extraction (should not overlap with VLLM_GPUS)

# Directory structure (model-specific)
BASE_DIR = "experiments"
CACHED_ACTIVATIONS_DIR = f"{BASE_DIR}/{MODEL_SHORT_NAME}/cached_activations"
RESULTS_DIR = f"{BASE_DIR}/{MODEL_SHORT_NAME}/results"  # Base results directory
MMLU_FILE = "mmlu_data/test.json"

def get_results_dir(num_examples: int = None, filter_reliable: bool = False, prompt_name: str = None) -> str:
    """
    Get results directory path organized by prompt, number of examples and filter status.
    
    Args:
        num_examples: Number of examples (default: DEFAULT_NUM_EXAMPLES)
        filter_reliable: Whether using filtered data (default: False)
        prompt_name: Prompt name (e.g., "benign", "semimalign", "lie_detector"). 
                     If None or standard prompts, uses default structure.
                     Special prompts like "lie_detector" get separate directories.
        
    Returns:
        Path to results subdirectory
    """
    num_examples = num_examples or DEFAULT_NUM_EXAMPLES
    filter_suffix = "filtered" if filter_reliable else "unfiltered"
    
    # Special handling for lie_detector and potentially other specialized experiments
    if prompt_name == "lie_detector":
        return f"{RESULTS_DIR}/lie_detector/n{num_examples}_{filter_suffix}"
    else:
        # Default structure for benign, semimalign, etc.
        return f"{RESULTS_DIR}/n{num_examples}_{filter_suffix}"

# Generation parameters
MAX_NEW_TOKENS = 6 #100 #900 #100
TEMPERATURE = 1.0  # Sampling temperature

# Experiment parameters
DEFAULT_LAYER = 30
DEFAULT_NUM_EXAMPLES = 20000
DEFAULT_TOKEN_POSITION = "letter"
# Available experiments: ["linear_probe", "mlp_probe", "pca", "anomaly_detection", "auroc_vs_n", "corruption_sweep", "auroc_vs_n_mlp"]
DEFAULT_EXPERIMENTS = "all" #["linear_probe", "mlp_probe"] #["linear_probe", "mlp_probe", "pca", "anomaly_detection", "auroc_vs_n", "auroc_vs_n_mlp", "corruption_sweep"]

# Positions to cache in unified cache (excludes "all" and "all_appended" which can be computed from others)
# CACHED_POSITIONS = [DEFAULT_TOKEN_POSITION] #["last", "first", "middle", "letter"]
CACHED_POSITIONS = ["last", "first", "middle", "letter", "letter+1", "yes-no"]

# SUPPORTED_LAYERS = [DEFAULT_LAYER] #[0, 2, 5, 10, 15, 20, 25, 30, 36, 42, 45, 47]
SUPPORTED_LAYERS = [0, 2, 5, 10, 15, 20, 25, 30, 36, 42, 45, 47]

DEFAULT_LAYER_SWEEP = [0, 2, 5, 10, 20, 30, 36, 42, 45, 47]
# for layer in DEFAULT_LAYER_SWEEP:
#     assert layer in SUPPORTED_LAYERS, f"Layer {layer} not in supported layers"

SUPPORTED_POSITIONS = ["last", "first", "middle", "all", "all_appended", "letter", "letter+1", "yes-no"]
DEFAULT_POSITION_SWEEP = ["letter", "all_appended", "letter+1", "yes-no", "last"]

# Probe training parameters
PROBE_MAX_ITER = 1000
PROBE_RANDOM_STATE = 42

# MLP Probe Configuration (optimal from hyperparameter search)
# Best AUROC: 0.8548 on Layer 30, Position: letter, n=2000
MLP_HIDDEN_LAYER_SIZES = (12000, 6000)  # Two-layer pyramidal architecture
MLP_LEARNING_RATE = 0.001
MLP_WEIGHT_DECAY = 0.08
MLP_DROPOUT = 0.1
MLP_PATIENCE = 10
MLP_LR_SCHEDULER = None
MLP_USE_CONSTANT_RESIDUAL = False

# Multiprocessing parameters
MAX_CONCURRENT_REQUESTS_PER_SERVER = 10  # For VLLM API calls
ACTIVATION_BATCH_SIZE = 1000 # Batch size for activation extraction (reduce if OOM)

# Visualization parameters
N_ANNOTATE = 30
SAVE_PLOTS = True  # Set to False to suppress creation of .png files

# Verbosity settings
# 0 = Silent (only errors and final results)
# 1 = Minimal (key milestones only)
# 2 = Normal (default - all status updates)
# 3 = Verbose (includes debug information)
VERBOSITY = 0

def get_config():
    """Return a dictionary with all configuration settings."""
    return {
        "model_name": MODEL_NAME,
        "model_short_name": MODEL_SHORT_NAME,
        "vllm_base_port": VLLM_BASE_PORT,
        "vllm_num_servers": VLLM_NUM_SERVERS,
        "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
        "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "vllm_gpus": VLLM_GPUS,
        "activation_gpus": ACTIVATION_GPUS,
        "cached_activations_dir": CACHED_ACTIVATIONS_DIR,
        "results_dir": RESULTS_DIR,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "default_layer": DEFAULT_LAYER,
        "default_num_examples": DEFAULT_NUM_EXAMPLES,
        "default_token_position": DEFAULT_TOKEN_POSITION,
        "probe_max_iter": PROBE_MAX_ITER,
        "probe_random_state": PROBE_RANDOM_STATE,
        "max_concurrent_requests_per_server": MAX_CONCURRENT_REQUESTS_PER_SERVER,
        "activation_batch_size": ACTIVATION_BATCH_SIZE,
        "save_plots": SAVE_PLOTS,
    }


def print_config():
    """Print current configuration."""
    print("=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    config = get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()

