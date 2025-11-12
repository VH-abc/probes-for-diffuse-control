#!/usr/bin/env python3
"""
Identify questions that the model has a 100% pass rate on.

This script:
1. Loads MMLU questions
2. Generates completions multiple times with benign prompts
3. Identifies questions answered correctly every time
4. Saves list of reliable question indices
"""

import os
import json
import argparse
import multiprocessing as mp
import numpy as np
from collections import defaultdict

import config
from lib.data import load_mmlu_data, compute_correctness_labels
from lib.generation import generate_with_vllm_multi_server


def identify_reliable_questions(
    num_trials: int = 5,
    num_examples: int = None,
    temperature: float = None,
    max_new_tokens: int = None,
    num_gpus: int = None,
    mmlu_file: str = "mmlu_data/train.json",
    output_file: str = None
):
    """
    Identify questions with 100% pass rate across multiple trials.
    
    Args:
        num_trials: Number of times to generate completions for each question
        num_examples: Number of MMLU examples to test
        temperature: Sampling temperature (defaults to config.TEMPERATURE)
        max_new_tokens: Max tokens to generate
        num_gpus: Number of GPUs
        mmlu_file: Path to MMLU data file
        output_file: Path to save reliable question indices
        
    Returns:
        reliable_indices: List of question indices with 100% pass rate
    """
    # Use config defaults
    num_examples = num_examples or config.DEFAULT_NUM_EXAMPLES
    temperature = temperature if temperature is not None else config.TEMPERATURE
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    num_gpus = num_gpus or config.VLLM_NUM_SERVERS
    
    print("\n" + "=" * 80)
    print("IDENTIFYING RELIABLE QUESTIONS (100% PASS RATE)")
    print("=" * 80)
    config.print_config()
    print(f"\nParameters:")
    print(f"  Trials per question: {num_trials}")
    print(f"  Total examples: {num_examples}")
    print(f"  Temperature: {temperature} (low for consistency)")
    print(f"  Max New Tokens: {max_new_tokens}")
    print("=" * 80)
    
    # Load MMLU data
    print(f"\nLoading {num_examples} MMLU examples from {mmlu_file}...")
    questions, prompts = load_mmlu_data(mmlu_file, num_examples)
    print(f"Loaded {len(prompts)} questions")
    
    # Generate completions for all trials in one call
    # Repeat each prompt num_trials times for efficient batching
    print(f"\nGenerating {num_trials} completions per question ({len(prompts) * num_trials} total)...")
    repeated_prompts = prompts * num_trials
    repeated_questions = questions * num_trials
    
    generated_texts, completions_only = generate_with_vllm_multi_server(
        prompts=repeated_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        model_name=config.MODEL_NAME,
        num_servers=num_gpus,
        base_port=config.VLLM_BASE_PORT,
        max_concurrent_requests=config.MAX_CONCURRENT_REQUESTS_PER_SERVER
    )
    
    # Compute correctness for all completions
    labels, predicted_answers, correct_answer_indices = compute_correctness_labels(
        repeated_questions, generated_texts
    )
    
    # Organize results by question and trial
    print(f"\nOrganizing results by question and trial...")
    correctness_tracker = defaultdict(list)
    num_questions = len(prompts)
    
    for i in range(len(labels)):
        question_idx = i % num_questions
        correctness_tracker[question_idx].append(labels[i])
    
    # Print per-trial statistics
    print(f"\n{'=' * 80}")
    print(f"TRIAL STATISTICS")
    print(f"{'=' * 80}")
    for trial in range(num_trials):
        trial_labels = [labels[trial * num_questions + i] for i in range(num_questions)]
        num_correct = sum(trial_labels)
        accuracy = num_correct / num_questions if num_questions > 0 else 0
        print(f"Trial {trial + 1}: {num_correct}/{num_questions} correct ({100 * accuracy:.1f}%)")
    
    # Identify questions with 100% pass rate
    reliable_indices = []
    for idx in range(len(questions)):
        if all(correctness_tracker[idx]):  # All trials correct
            reliable_indices.append(idx)
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS")
    print(f"{'=' * 80}")
    print(f"  Total questions tested: {len(questions)}")
    print(f"  Questions with 100% pass rate: {len(reliable_indices)}")
    print(f"  Percentage: {100 * len(reliable_indices) / len(questions):.1f}%")
    print(f"{'=' * 80}")
    
    # Analyze by subject
    subject_stats = defaultdict(lambda: {"total": 0, "reliable": 0})
    for idx in range(len(questions)):
        subject = questions[idx]['subject']
        subject_stats[subject]["total"] += 1
        if idx in reliable_indices:
            subject_stats[subject]["reliable"] += 1
    
    print(f"\nReliability by subject:")
    print(f"{'Subject':<40} {'Reliable/Total':<20} {'Rate':<10}")
    print(f"{'-' * 75}")
    for subject in sorted(subject_stats.keys()):
        stats = subject_stats[subject]
        rate = stats["reliable"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{subject:<40} {stats['reliable']}/{stats['total']:<15} {100*rate:>5.1f}%")
    
    # Save results
    os.makedirs(config.CACHED_ACTIVATIONS_DIR, exist_ok=True)
    
    if output_file is None:
        output_file = os.path.join(
            config.CACHED_ACTIVATIONS_DIR,
            f"reliable_questions_n{num_examples}_trials{num_trials}.json"
        )
    
    # Extract reliable questions
    reliable_questions = [questions[i] for i in reliable_indices]
    
    output_data = {
        "num_trials": num_trials,
        "total_questions": len(questions),
        "num_reliable": len(reliable_indices),
        "reliability_rate": len(reliable_indices) / len(questions),
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "model_name": config.MODEL_NAME,
        "model_short_name": config.MODEL_SHORT_NAME,
        "reliable_indices": reliable_indices,
        "reliable_questions": reliable_questions,
        "subject_stats": dict(subject_stats)
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved reliable question data to: {output_file}")
    print(f"\nTo use these questions in experiments:")
    print(f"  --reliable-questions-file {output_file}")
    print()
    
    return reliable_indices, output_file


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(
        description="Identify questions with 100% pass rate"
    )
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Number of trials per question (default: 5)")
    parser.add_argument("--num-examples", type=int, help=f"Number of examples (default: {config.DEFAULT_NUM_EXAMPLES})")
    parser.add_argument("--temperature", type=float,
                        help=f"Temperature for generation (default: {config.TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, help=f"Max new tokens (default: {config.MAX_NEW_TOKENS})")
    parser.add_argument("--num-gpus", type=int, help=f"Number of GPUs (default: {config.VLLM_NUM_SERVERS})")
    parser.add_argument("--mmlu-file", type=str, default="mmlu_data/test.json",
                        help="Path to MMLU data file")
    parser.add_argument("--output-file", type=str, help="Output file path", default=f"experiments/{config.MODEL_SHORT_NAME}/reliable_questions.json")
    
    args = parser.parse_args()
    
    identify_reliable_questions(
        num_trials=args.num_trials,
        num_examples=args.num_examples,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        num_gpus=args.num_gpus,
        mmlu_file=args.mmlu_file,
        output_file=args.output_file
    )

