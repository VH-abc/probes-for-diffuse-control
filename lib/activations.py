"""
Neural activation extraction utilities.
"""

import re
import time
import multiprocessing as mp
from multiprocessing import Manager
from typing import List, Literal, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


TokenPosition = Literal["last", "first", "middle", "all", "all_appended", "letter"]


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


def find_letter_token_in_completion(completion: str) -> Optional[str]:
    """
    Find the answer letter (A, B, C, or D) in the completion text.
    
    Args:
        completion: The generated completion text
        
    Returns:
        The answer letter if found, otherwise None
        
    Common patterns:
        - "(A)", "(B)", etc.
        - "A)", "B)", etc.
        - "A.", "B.", etc.
        - Just "A", "B", "C", "D" (last resort)
    """
    # Try to find answer in common formats (in order of specificity)
    patterns = [
        r'\(([A-D])\)',  # (A)
        r'\b([A-D])\)',  # A)
        r'\b([A-D])\.',  # A.
        r'\b([A-D])\b',  # Just the letter
    ]
    
    for pattern in patterns:
        match = re.search(pattern, completion)
        if match:
            return match.group(1)
    
    return None


def find_letter_token_position(
    full_text: str,
    completion: str,
    tokenizer,
    input_ids: torch.Tensor
) -> Tuple[Optional[int], Optional[str]]:
    """
    Find the token position where the answer letter appears in the completion.
    
    Args:
        full_text: The full text (prompt + completion)
        completion: Just the completion part
        tokenizer: The tokenizer
        input_ids: Token IDs for the full text
        
    Returns:
        Tuple of (token_position, literal_token_string) or (None, None) if not found
    """
    # Find the answer letter in the completion
    answer_letter = find_letter_token_in_completion(completion)
    if answer_letter is None:
        # Fallback to last token if we can't find the letter
        return None, None
    
    # Find where the completion starts in the full text
    prompt_length = len(full_text) - len(completion)
    
    # Look for the answer letter in the completion
    # We'll search for common patterns
    patterns = [
        f'({answer_letter})',  # (A)
        f'{answer_letter})',   # A)
        f'{answer_letter}.',   # A.
        answer_letter,         # Just A
    ]
    
    letter_char_pos = None
    for pattern in patterns:
        pos = completion.find(pattern)
        if pos != -1:
            # Found it! Calculate position in full text
            letter_char_pos = prompt_length + pos
            # Find the exact position where just the letter appears (not the parenthesis)
            if pattern.startswith('('):
                letter_char_pos += 1  # Skip the opening paren
            break
    
    if letter_char_pos is None:
        return None, None
    
    # Now we need to find which token contains this character position
    # We'll do this by decoding tokens progressively until we pass the character position
    cumulative_text = ""
    for token_idx in range(len(input_ids)):
        # Decode up to this token
        decoded = tokenizer.decode(input_ids[:token_idx + 1], skip_special_tokens=False)
        
        if len(decoded) > letter_char_pos:
            # This token contains or passes our target position
            # Get the literal token string
            token_str = tokenizer.decode([input_ids[token_idx]], skip_special_tokens=False)
            return token_idx, token_str
    
    # Fallback
    return None, None


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
            - "all_appended": All non-special tokens appended together

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
    elif token_position == "all_appended":
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
    use_cache: bool = True,
    completions: Optional[List[str]] = None
):
    """
    Worker function to extract activations on a single GPU with batching and caching.

    Args:
        gpu_id: GPU device ID
        model_name: Model name/path
        full_texts: List of texts to process
        layer_idx: Layer to extract from
        token_position: Token position to extract (e.g. "last", "first", "middle", "all", "all_appended", "letter")
        return_dict: Shared dict for results
        index: Index in return dict
        batch_size: Process this many texts at once (reduces overhead)
        use_cache: Use cached model files (faster loading)
        completions: List of completion texts (required for "letter" position)
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
        print(f"[GPU {gpu_id}] Model loaded on device: {device} == {model.device}")
        load_time = time.time() - load_start
        print(f"[GPU {gpu_id}] Model loaded in {load_time:.2f}s")

        if hasattr(model, "config"):
            model.config.use_cache = False

        model.eval()
        activations_list = []
        probed_tokens_list = []  # Track which literal tokens were probed

        print(f"[GPU {gpu_id}] Extracting activations from layer {layer_idx} (batch_size={batch_size})...")
        if token_position == "letter" and completions is None:
            raise ValueError("completions must be provided when token_position='letter'")

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
                for idx, full_text in enumerate(batch_texts):
                    text_idx = batch_start + idx
                    
                    # Tokenize the full generated text
                    inputs = tokenizer(full_text, return_tensors="pt").to(device)
                    input_ids = inputs.input_ids[0]

                    # Find the position(s) of tokens to probe
                    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))
                    probed_token_str = None
                    
                    if token_position == "letter":
                        # Special handling for letter token
                        completion = completions[text_idx] if completions else ""
                        letter_pos, probed_token_str = find_letter_token_position(
                            full_text, completion, tokenizer, input_ids
                        )
                        if letter_pos is not None:
                            probe_positions = [letter_pos]
                        else:
                            # Fallback to last token if we can't find the letter
                            probe_positions = get_probe_positions(
                                input_ids, special_token_ids, "last"
                            )
                            probed_token_str = tokenizer.decode([input_ids[probe_positions[0]]], skip_special_tokens=False)
                    else:
                        probe_positions = get_probe_positions(
                            input_ids, special_token_ids, token_position
                        )
                        # Record the token string for the first position
                        if probe_positions and token_position not in ["all", "all_appended"]:
                            probed_token_str = tokenizer.decode([input_ids[probe_positions[0]]], skip_special_tokens=False)

                    # Run forward pass to get hidden states
                    forward_outputs = model(**inputs, output_hidden_states=True, use_cache=False)

                    # Get the hidden states from the specified layer
                    # Note: hidden_states[0] is embeddings, hidden_states[1] is layer 0, etc.
                    layer_output = forward_outputs.hidden_states[layer_idx + 1]

                    # Extract activation(s) at the chosen position(s)
                    if token_position == "all" or token_position == "all_appended":
                        activations_at_positions = [
                            layer_output[0, pos, :].float().cpu().numpy()
                            for pos in probe_positions
                        ]
                        if token_position == "all_appended":
                            # Append together all activations
                            final_activation = np.concatenate(activations_at_positions, axis=0)
                        else:
                            # Average pooling
                            final_activation = np.mean(activations_at_positions, axis=0)
                    else:
                        # Single position
                        pos = probe_positions[0]
                        final_activation = layer_output[0, pos, :].float().cpu().numpy()

                    activations_list.append(final_activation)
                    probed_tokens_list.append(probed_token_str if probed_token_str else "")
                
                # Clear CUDA cache periodically to prevent OOM
                if batch_end % 50 == 0:
                    torch.cuda.empty_cache()

        total_time = time.time() - start_time
        rate = len(full_texts) / total_time if total_time > 0 else 0
        print(f"[GPU {gpu_id}] Complete! {len(full_texts)} texts in {total_time:.2f}s ({rate:.1f} texts/sec)")
        return_dict[index] = (np.array(activations_list), probed_tokens_list)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[index] = (np.array([]), [])


def extract_activations_multi_gpu(
    model_name: str,
    full_texts: List[str],
    layer_idx: int,
    token_position: TokenPosition = "last",
    num_gpus: int = 8,
    batch_size: int = 4,
    use_model_cache: bool = True,
    gpu_ids: List[int] = None,
    completions: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract activations from a specific layer using multiple GPUs in parallel.

    Args:
        model_name: Name of the model to use
        full_texts: List of full generated texts (prompt + completion)
        layer_idx: Which layer to extract activations from
        token_position: Which token position to extract (e.g. "last", "first", "middle", "all", "all_appended", "letter")
        num_gpus: Number of GPUs to use (if gpu_ids not specified)
        batch_size: Batch size for processing (reduces overhead)
        use_model_cache: Use cached model files (faster loading)
        gpu_ids: Specific GPU IDs to use (e.g., [4,5,6,7]). If None, uses 0..num_gpus-1
        completions: List of completion texts (required for "letter" position)

    Returns:
        Tuple of (activations, probed_tokens):
            - activations: Array of activations (num_prompts, hidden_dim)
            - probed_tokens: List of literal token strings that were probed
    """
    # Determine which GPUs to use
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    else:
        num_gpus = len(gpu_ids)
    
    print(f"\n{'='*60}")
    print(f"Distributing {len(full_texts)} texts across {num_gpus} GPUs for activation extraction")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Layer: {layer_idx}, Token position: {token_position}")
    print(f"Batch size: {batch_size}, Model cache: {'enabled' if use_model_cache else 'disabled'}")
    print(f"{'='*60}")

    # Pre-cache model to avoid race conditions during parallel loading
    if use_model_cache:
        ensure_model_cached(model_name)

    # Split texts into chunks for each GPU
    chunk_size = (len(full_texts) + num_gpus - 1) // num_gpus
    text_chunks = [full_texts[i:i + chunk_size] for i in range(0, len(full_texts), chunk_size)]
    
    # Split completions if provided
    completion_chunks = None
    if completions is not None:
        completion_chunks = [completions[i:i + chunk_size] for i in range(0, len(completions), chunk_size)]

    print(f"Chunk sizes: {[len(chunk) for chunk in text_chunks]}")

    # Use Manager to share results between processes
    manager = Manager()
    return_dict = manager.dict()

    # Create and start processes
    processes = []
    for i, chunk in enumerate(text_chunks):
        gpu_id = gpu_ids[i]
        completion_chunk = completion_chunks[i] if completion_chunks else None
        p = mp.Process(
            target=extract_activations_single_gpu,
            args=(gpu_id, model_name, chunk, layer_idx, token_position,
                  return_dict, i, batch_size, use_model_cache, completion_chunk)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Concatenate all activations and probed_tokens (maintaining order)
    activation_results = []
    probed_token_results = []
    for i in range(len(text_chunks)):
        if i in return_dict:
            activations_chunk, tokens_chunk = return_dict[i]
            activation_results.append(activations_chunk)
            probed_token_results.extend(tokens_chunk)
        else:
            print(f"Warning: GPU {i} did not return results")

    activations = np.vstack(activation_results) if activation_results else np.array([])

    print(f"\nAll GPUs completed activation extraction! Shape: {activations.shape}")
    return activations, probed_token_results

