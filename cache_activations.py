#!/usr/bin/env python3
"""
Cache neural activations for MMLU questions.

This script:
1. Loads MMLU questions
2. Generates completions via VLLM API servers
3. Extracts neural activations at specified layer and token position
4. Saves activations and metadata to disk

Requires VLLM servers to be running (see vllm_launcher.sh).
"""

import os
import json
import argparse
import multiprocessing as mp
import numpy as np

import config
from lib.data import load_mmlu_data, compute_correctness_labels
from lib.generation import generate_with_vllm_multi_server
from lib.activations import extract_activations_multi_gpu


def cache_mmlu_activations(
    prompt_name: str,
    model_name: str = None,
    layer_idx: int = None,
    token_position: str = None,
    num_examples: int = None,
    max_new_tokens: int = None,
    temperature: float = None,
    num_gpus: int = None,
    mmlu_file: str = "mmlu_data/train.json"
):
    """
    Cache layer activations for MMLU questions with autoregressive generation.

    Args:
        prompt_name: Name of the prompt to use (e.g., "benign", "50/50") - REQUIRED
        model_name: Model name (defaults to config.MODEL_NAME)
        layer_idx: Layer index to extract (defaults to config.DEFAULT_LAYER)
        token_position: Token position to extract (defaults to config.DEFAULT_TOKEN_POSITION)
        num_examples: Number of examples (defaults to config.DEFAULT_NUM_EXAMPLES)
        max_new_tokens: Max tokens to generate (defaults to config.MAX_NEW_TOKENS)
        temperature: Sampling temperature (defaults to config.TEMPERATURE)
        num_gpus: Number of GPUs (defaults to config.VLLM_NUM_SERVERS)
        mmlu_file: Path to MMLU data file
    """
    # Use config defaults if not specified
    model_name = model_name or config.MODEL_NAME
    layer_idx = layer_idx if layer_idx is not None else config.DEFAULT_LAYER
    token_position = token_position or config.DEFAULT_TOKEN_POSITION
    num_examples = num_examples or config.DEFAULT_NUM_EXAMPLES
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    temperature = temperature if temperature is not None else config.TEMPERATURE
    num_gpus = num_gpus or config.VLLM_NUM_SERVERS

    # Create output directory based on prompt name
    output_dir = os.path.join(config.CACHED_ACTIVATIONS_DIR, prompt_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("ACTIVATION CACHING")
    print("=" * 80)
    config.print_config()
    print(f"\nRun Parameters:")
    print(f"  Prompt: {prompt_name}")
    print(f"  Layer: {layer_idx}")
    print(f"  Token Position: {token_position}")
    print(f"  Examples: {num_examples}")
    print(f"  Max New Tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    # Load MMLU data
    print(f"\nLoading {num_examples} MMLU examples from {mmlu_file}...")
    questions, prompts = load_mmlu_data(mmlu_file, num_examples, prompt_name=prompt_name)
    print(f"Loaded {len(prompts)} samples")

    # Step 1: Generate completions via VLLM
    print(f"\n{'=' * 80}")
    print(f"STEP 1: Generating completions with VLLM servers")
    print(f"{'=' * 80}")
    generated_texts, completions_only = generate_with_vllm_multi_server(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        model_name=model_name,  # Use full model name for API
        num_servers=num_gpus,
        base_port=config.VLLM_BASE_PORT,
        max_concurrent_requests=config.MAX_CONCURRENT_REQUESTS_PER_SERVER
    )

    # Step 2: Extract activations
    print(f"\n{'=' * 80}")
    print(f"STEP 2: Extracting activations")
    print(f"{'=' * 80}")
    activations = extract_activations_multi_gpu(
        model_name=model_name,
        full_texts=generated_texts,
        layer_idx=layer_idx,
        token_position=token_position,
        num_gpus=num_gpus,
        batch_size=config.ACTIVATION_BATCH_SIZE,
        use_model_cache=True
    )

    print(f"\n{'=' * 80}")
    print(f"Activations shape: {activations.shape}")
    print(f"  Samples: {activations.shape[0]}")
    print(f"  Features: {activations.shape[1]}")
    print(f"{'=' * 80}")

    # Compute correctness labels
    print(f"\nComputing correctness labels...")
    labels, predicted_answers, correct_answer_indices = compute_correctness_labels(
        questions, generated_texts
    )
    labels = np.array(labels)

    num_correct = np.sum(labels)
    num_incorrect = len(labels) - num_correct
    accuracy = num_correct / len(labels) if len(labels) > 0 else 0

    print(f"  Correct: {num_correct}/{len(labels)} ({100 * accuracy:.1f}%)")
    print(f"  Incorrect: {num_incorrect}/{len(labels)}")

    # Extract subjects
    subjects = np.array([q['subject'] for q in questions])

    # Save everything
    output_prefix = f"mmlu_layer{layer_idx:02d}_pos-{token_position}_n{num_examples}"

    files_to_save = {
        "activations": (f"{output_prefix}_activations.npy", activations),
        "labels": (f"{output_prefix}_labels.npy", labels),
        "subjects": (f"{output_prefix}_subjects.npy", subjects),
        "prompts": (f"{output_prefix}_prompts.npy", np.array(prompts)),
        "generated": (f"{output_prefix}_generated.npy", np.array(generated_texts)),
        "completions": (f"{output_prefix}_completions.npy", np.array(completions_only)),
        "predicted_answers": (f"{output_prefix}_predicted_answers.npy", np.array(predicted_answers)),
        "correct_answer_indices": (f"{output_prefix}_correct_answer_indices.npy", np.array(correct_answer_indices)),
    }

    print(f"\nSaving cached data...")
    saved_files = {}
    for key, (filename, data) in files_to_save.items():
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, data)
        saved_files[key] = filepath
        print(f"  ✓ {key}: {filepath}")

    # Save questions as JSON
    questions_file = os.path.join(output_dir, f"{output_prefix}_questions.json")
    with open(questions_file, 'w') as f:
        json.dump(questions, f, indent=2)
    saved_files['questions'] = questions_file
    print(f"  ✓ questions: {questions_file}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_short_name": config.MODEL_SHORT_NAME,
        "prompt_name": prompt_name,
        "layer_idx": layer_idx,
        "token_position": token_position,
        "num_examples": num_examples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "activation_shape": list(activations.shape),
        "num_correct": int(num_correct),
        "num_incorrect": int(num_incorrect),
        "accuracy": float(accuracy),
        "description": f"MMLU activations at layer {layer_idx}, token position '{token_position}', prompt '{prompt_name}'. Labels: 1=correct, 0=incorrect.",
        "files": saved_files
    }
    metadata_file = os.path.join(output_dir, f"{output_prefix}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata: {metadata_file}")

    print(f"\n{'=' * 80}")
    print(f"✓ Caching complete!")
    print(f"  Cached {len(prompts)} samples from layer {layer_idx}, position '{token_position}'")
    print(f"  Accuracy: {100 * accuracy:.1f}%")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(description="Cache MMLU activations")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt name to use (e.g., 'benign', '50/50') - REQUIRED")
    parser.add_argument("--model", type=str, help=f"Model name (default: {config.MODEL_NAME})")
    parser.add_argument("--layer", type=int, help=f"Layer index (default: {config.DEFAULT_LAYER})")
    parser.add_argument("--position", type=str, choices=["last", "first", "middle", "all"],
                        help=f"Token position (default: {config.DEFAULT_TOKEN_POSITION})")
    parser.add_argument("--num-examples", type=int, help=f"Number of examples (default: {config.DEFAULT_NUM_EXAMPLES})")
    parser.add_argument("--max-tokens", type=int, help=f"Max new tokens (default: {config.MAX_NEW_TOKENS})")
    parser.add_argument("--temperature", type=float, help=f"Temperature (default: {config.TEMPERATURE})")
    parser.add_argument("--num-gpus", type=int, help=f"Number of GPUs (default: {config.VLLM_NUM_SERVERS})")
    parser.add_argument("--mmlu-file", type=str, default="mmlu_data/train.json",
                        help="Path to MMLU data file")

    args = parser.parse_args()

    cache_mmlu_activations(
        prompt_name=args.prompt,
        model_name=args.model,
        layer_idx=args.layer,
        token_position=args.position,
        num_examples=args.num_examples,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        num_gpus=args.num_gpus,
        mmlu_file=args.mmlu_file
    )

