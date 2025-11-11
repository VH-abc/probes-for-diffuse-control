"""
Neural activation extraction utilities.
"""

import time
import multiprocessing as mp
from multiprocessing import Manager
from typing import List, Literal
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


TokenPosition = Literal["last", "first", "middle", "all"]


def ensure_model_cached(model_name: str):
    """
    Pre-download model to cache to avoid multiple processes downloading simultaneously.

    Args:
        model_name: Model name to cache
    """
    try:
        print(f"Ensuring {model_name} is cached locally...")
        # This will download/cache the model config and weights if not already present
        AutoConfig.from_pretrained(model_name)
        print(f"  ✓ Model cached")
    except Exception as e:
        print(f"  ⚠ Warning: Could not pre-cache model: {e}")
        print(f"  Processes will download individually (slower)")


def get_probe_positions(
    input_ids: torch.Tensor,
    special_token_ids: set,
    token_position: TokenPosition = "last"
) -> List[int]:
    """
    Get token positions to extract activations from.

    Args:
        input_ids: Token IDs (1D tensor)
        special_token_ids: Set of special token IDs to exclude
        token_position: Which position(s) to extract
            - "last": Last non-special token
            - "first": First non-special token
            - "middle": Middle non-special token
            - "all": All non-special tokens

    Returns:
        List of token positions to extract
    """
    # Find all non-special token positions
    non_special_positions = [
        idx for idx, token_id in enumerate(input_ids.tolist())
        if int(token_id) not in special_token_ids
    ]

    if len(non_special_positions) == 0:
        # Fallback to last position if all are special tokens
        return [len(input_ids) - 1 if len(input_ids) > 0 else 0]

    if token_position == "last":
        return [non_special_positions[-1]]
    elif token_position == "first":
        return [non_special_positions[0]]
    elif token_position == "middle":
        mid_idx = len(non_special_positions) // 2
        return [non_special_positions[mid_idx]]
    elif token_position == "all":
        return non_special_positions
    else:
        raise ValueError(f"Unknown token_position: {token_position}")


def extract_activations_single_gpu(
    gpu_id: int,
    model_name: str,
    full_texts: List[str],
    layer_idx: int,
    token_position: TokenPosition,
    return_dict: dict,
    index: int,
    batch_size: int = 4,
    use_cache: bool = True
):
    """
    Worker function to extract activations on a single GPU with batching and caching.

    Args:
        gpu_id: GPU device ID
        model_name: Model name/path
        full_texts: List of texts to process
        layer_idx: Layer to extract from
        token_position: Token position to extract ("last", "first", "middle", "all")
        return_dict: Shared dict for results
        index: Index in return dict
        batch_size: Process this many texts at once (reduces overhead)
        use_cache: Use cached model files (faster loading)
    """
    try:
        start_time = time.time()
        print(f"[GPU {gpu_id}] Loading model for activation extraction ({len(full_texts)} texts)...")
        print(f"[GPU {gpu_id}] Token position: {token_position}")

        # Set device for this process
        device = f"cuda:{gpu_id}"

        # Load tokenizer (fast, no optimization needed)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with optimizations
        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
            low_cpu_mem_usage=True,  # Reduces CPU memory during loading
            local_files_only=use_cache  # Don't re-download if cached
        )
        load_time = time.time() - load_start
        print(f"[GPU {gpu_id}] Model loaded in {load_time:.2f}s")

        if hasattr(model, "config"):
            model.config.use_cache = False

        model.eval()
        activations_list = []

        print(f"[GPU {gpu_id}] Extracting activations from layer {layer_idx} (batch_size={batch_size})...")

        with torch.no_grad():
            # Process in batches to reduce per-sample overhead
            for batch_start in range(0, len(full_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(full_texts))
                batch_texts = full_texts[batch_start:batch_end]

                if (batch_end) % 20 == 0 or batch_end == len(full_texts):
                    elapsed = time.time() - start_time
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    print(f"  [GPU {gpu_id}] Processing {batch_end}/{len(full_texts)} ({rate:.1f} texts/sec)")

                # Process each text in batch (can't truly batch due to variable lengths)
                for full_text in batch_texts:
                    # Tokenize the full generated text
                    inputs = tokenizer(full_text, return_tensors="pt").to(device)
                    input_ids = inputs.input_ids[0]

                    # Find the position(s) of tokens to probe
                    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))
                    probe_positions = get_probe_positions(
                        input_ids, special_token_ids, token_position
                    )

                    # Run forward pass to get hidden states
                    forward_outputs = model(**inputs, output_hidden_states=True, use_cache=False)

                    # Get the hidden states from the specified layer
                    # Note: hidden_states[0] is embeddings, hidden_states[1] is layer 0, etc.
                    layer_output = forward_outputs.hidden_states[layer_idx + 1]

                    # Extract activation(s) at the chosen position(s)
                    if token_position == "all":
                        # Average over all non-special token positions
                        activations_at_positions = [
                            layer_output[0, pos, :].float().cpu().numpy()
                            for pos in probe_positions
                        ]
                        # Average pooling
                        final_activation = np.mean(activations_at_positions, axis=0)
                    else:
                        # Single position
                        pos = probe_positions[0]
                        final_activation = layer_output[0, pos, :].float().cpu().numpy()

                    activations_list.append(final_activation)

        total_time = time.time() - start_time
        rate = len(full_texts) / total_time if total_time > 0 else 0
        print(f"[GPU {gpu_id}] Complete! {len(full_texts)} texts in {total_time:.2f}s ({rate:.1f} texts/sec)")
        return_dict[index] = np.array(activations_list)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[index] = np.array([])


def extract_activations_multi_gpu(
    model_name: str,
    full_texts: List[str],
    layer_idx: int,
    token_position: TokenPosition = "last",
    num_gpus: int = 8,
    batch_size: int = 4,
    use_model_cache: bool = True
) -> np.ndarray:
    """
    Extract activations from a specific layer using multiple GPUs in parallel.

    Args:
        model_name: Name of the model to use
        full_texts: List of full generated texts (prompt + completion)
        layer_idx: Which layer to extract activations from
        token_position: Which token position to extract ("last", "first", "middle", "all")
        num_gpus: Number of GPUs to use
        batch_size: Batch size for processing (reduces overhead)
        use_model_cache: Use cached model files (faster loading)

    Returns:
        activations: Array of activations (num_prompts, hidden_dim)
    """
    print(f"\n{'='*60}")
    print(f"Distributing {len(full_texts)} texts across {num_gpus} GPUs for activation extraction")
    print(f"Layer: {layer_idx}, Token position: {token_position}")
    print(f"Batch size: {batch_size}, Model cache: {'enabled' if use_model_cache else 'disabled'}")
    print(f"{'='*60}")

    # Pre-cache model to avoid race conditions during parallel loading
    if use_model_cache:
        ensure_model_cached(model_name)

    # Split texts into chunks for each GPU
    chunk_size = (len(full_texts) + num_gpus - 1) // num_gpus
    text_chunks = [full_texts[i:i + chunk_size] for i in range(0, len(full_texts), chunk_size)]

    print(f"Chunk sizes: {[len(chunk) for chunk in text_chunks]}")

    # Use Manager to share results between processes
    manager = Manager()
    return_dict = manager.dict()

    # Create and start processes
    processes = []
    for gpu_id, chunk in enumerate(text_chunks):
        p = mp.Process(
            target=extract_activations_single_gpu,
            args=(gpu_id, model_name, chunk, layer_idx, token_position,
                  return_dict, gpu_id, batch_size, use_model_cache)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Concatenate all activations (maintaining order)
    results = []
    for i in range(len(text_chunks)):
        if i in return_dict:
            results.append(return_dict[i])
        else:
            print(f"Warning: GPU {i} did not return results")

    activations = np.vstack(results) if results else np.array([])

    print(f"\nAll GPUs completed activation extraction! Shape: {activations.shape}")
    return activations

