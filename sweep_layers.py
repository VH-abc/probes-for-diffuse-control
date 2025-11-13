#!/usr/bin/env python3
"""
Sweep across layers and token positions to find optimal probe configurations.

This script systematically:
1. Caches activations for multiple layers/positions
2. Runs probe analysis for each
3. Compares results to find best configuration
"""

import os
import json
import argparse
from typing import List, Optional

import config
from cache_activations import cache_mmlu_activations
from probe_analysis import run_probe_analysis
from download_mmlu import download_mmlu

if not os.path.exists("mmlu_data"):
    download_mmlu()


def check_cache_exists(prompt_name: str, layer: int, token_position: str, num_examples: int, filter_reliable: bool = False) -> bool:
    """
    Check if cached activations already exist for the given parameters.
    
    Args:
        prompt_name: Prompt name
        layer: Layer index
        token_position: Token position
        num_examples: Number of examples
        filter_reliable: Whether using filtered cache
        
    Returns:
        True if cache exists and is complete, False otherwise
    """
    cache_prompt_name = f"{prompt_name}_filtered" if filter_reliable else prompt_name
    filter_suffix = "filtered" if filter_reliable else "unfiltered"
    prefix = f"mmlu_layer{layer:02d}_pos-{token_position}_n{num_examples}_{filter_suffix}"
    cache_dir = os.path.join(config.CACHED_ACTIVATIONS_DIR, cache_prompt_name)
    
    # Check for essential files
    essential_files = [
        f"{prefix}_activations.npy",
        f"{prefix}_labels.npy",
        f"{prefix}_metadata.json"
    ]
    
    for filename in essential_files:
        filepath = os.path.join(cache_dir, filename)
        if not os.path.exists(filepath):
            return False
    
    return True

def sweep_layers(
    prompt_name: str = "50-50",
    layers: List[int] = None,
    token_position: str = "last",
    num_examples: int = None,
    skip_cache: bool = False,
    skip_analysis: bool = False,
    filter_reliable: bool = False,
    reliable_questions_file: str = None
):
    """
    Sweep across multiple layers.

    Args:
        prompt_name: Name of the prompt to use (e.g., "benign", "50-50") - defaults to "50-50"
        layers: List of layer indices to sweep
        token_position: Token position to use
        num_examples: Number of examples
        skip_cache: Skip activation caching (use existing)
        skip_analysis: Skip probe analysis
        filter_reliable: Whether to filter to only reliable questions
        reliable_questions_file: Path to reliable questions JSON file
    """
    num_examples = num_examples or config.DEFAULT_NUM_EXAMPLES
    
    # Set default reliable questions file if filtering
    if filter_reliable and reliable_questions_file is None:
        reliable_questions_file = os.path.join(
            config.BASE_DIR, config.MODEL_SHORT_NAME, "reliable_questions.json"
        )

    print("\n" + "#" * 80)
    print("# LAYER SWEEP")
    print(f"# Model: {config.MODEL_SHORT_NAME}")
    print(f"# Prompt: {prompt_name}")
    print(f"# Filtered: {filter_reliable}")
    if filter_reliable:
        print(f"# Reliable Questions File: {reliable_questions_file}")
    print(f"# Layers: {layers}")
    print(f"# Token Position: {token_position}")
    print(f"# Examples: {num_examples}")
    print("#" * 80)

    successful_layers = []

    for layer in layers:
        print(f"\n{'█' * 80}")
        print(f"█ PROCESSING LAYER {layer}")
        print(f"{'█' * 80}\n")

        try:
            # Step 1: Cache activations
            if not skip_cache:
                # Check if cache already exists
                cache_exists = check_cache_exists(prompt_name, layer, token_position, num_examples, filter_reliable)
                
                if cache_exists:
                    print(f"✓ Cache already exists for layer {layer}, skipping activation caching")
                else:
                    print(f"Caching activations for layer {layer}...")
                    cache_mmlu_activations(
                        prompt_name=prompt_name,
                        layer_idx=layer,
                        token_position=token_position,
                        num_examples=num_examples,
                        filter_reliable=filter_reliable,
                        reliable_questions_file=reliable_questions_file
                    )
            else:
                print(f"⏩ Skipping cache (--skip-cache flag)")

            # Step 2: Run analysis
            if not skip_analysis:
                print(f"Running probe analysis for layer {layer}...")
                run_probe_analysis(
                    prompt_name=prompt_name,
                    layer=layer,
                    token_position=token_position,
                    num_examples=num_examples,
                    filter_reliable=filter_reliable
                )
            else:
                print(f"⏩ Skipping analysis")

            successful_layers.append(layer)
            print(f"✓ Layer {layer} complete")

        except Exception as e:
            print(f"⚠️  Error processing layer {layer}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison report
    if successful_layers and not skip_analysis:
        generate_layer_comparison(prompt_name, successful_layers, token_position, num_examples, filter_reliable)

    results_dir = config.get_results_dir(num_examples, filter_reliable)
    print(f"\n{'#' * 80}")
    print(f"# LAYER SWEEP COMPLETE")
    print(f"# Successful: {len(successful_layers)}/{len(layers)} layers")
    print(f"# Results in: {results_dir}")
    print(f"{'#' * 80}\n")


def sweep_positions(
    prompt_name: str = "50-50",
    positions: List[str] = None,
    layer: int = None,
    num_examples: int = None,
    skip_cache: bool = False,
    skip_analysis: bool = False,
    filter_reliable: bool = False,
    reliable_questions_file: str = None
):
    """
    Sweep across multiple token positions.

    Args:
        prompt_name: Name of the prompt to use (e.g., "benign", "50-50") - defaults to "50-50"
        positions: List of token positions to sweep
        layer: Layer to use
        num_examples: Number of examples
        skip_cache: Skip activation caching (use existing)
        skip_analysis: Skip probe analysis
        filter_reliable: Whether to filter to only reliable questions
        reliable_questions_file: Path to reliable questions JSON file
    """
    layer = layer if layer is not None else config.DEFAULT_LAYER
    num_examples = num_examples or config.DEFAULT_NUM_EXAMPLES
    
    # Set default reliable questions file if filtering
    if filter_reliable and reliable_questions_file is None:
        reliable_questions_file = os.path.join(
            config.BASE_DIR, config.MODEL_SHORT_NAME, "reliable_questions.json"
        )

    print("\n" + "#" * 80)
    print("# TOKEN POSITION SWEEP")
    print(f"# Model: {config.MODEL_SHORT_NAME}")
    print(f"# Prompt: {prompt_name}")
    print(f"# Filtered: {filter_reliable}")
    if filter_reliable:
        print(f"# Reliable Questions File: {reliable_questions_file}")
    print(f"# Layer: {layer}")
    print(f"# Positions: {positions}")
    print(f"# Examples: {num_examples}")
    print("#" * 80)

    successful_positions = []

    for position in positions:
        print(f"\n{'█' * 80}")
        print(f"█ PROCESSING POSITION: {position.upper()}")
        print(f"{'█' * 80}\n")

        try:
            # Step 1: Cache activations
            if not skip_cache:
                # Check if cache already exists
                cache_exists = check_cache_exists(prompt_name, layer, position, num_examples, filter_reliable)
                
                if cache_exists:
                    print(f"✓ Cache already exists for position '{position}', skipping activation caching")
                else:
                    print(f"Caching activations for position '{position}'...")
                    cache_mmlu_activations(
                        prompt_name=prompt_name,
                        layer_idx=layer,
                        token_position=position,
                        num_examples=num_examples,
                        filter_reliable=filter_reliable,
                        reliable_questions_file=reliable_questions_file
                    )
            else:
                print(f"⏩ Skipping cache (--skip-cache flag)")

            # Step 2: Run analysis
            if not skip_analysis:
                print(f"Running probe analysis for position '{position}'...")
                run_probe_analysis(
                    prompt_name=prompt_name,
                    layer=layer,
                    token_position=position,
                    num_examples=num_examples,
                    filter_reliable=filter_reliable
                )
            else:
                print(f"⏩ Skipping analysis")

            successful_positions.append(position)
            print(f"✓ Position '{position}' complete")

        except Exception as e:
            print(f"⚠️  Error processing position '{position}': {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison report
    if successful_positions and not skip_analysis:
        generate_position_comparison(prompt_name, successful_positions, layer, num_examples, filter_reliable)

    results_dir = config.get_results_dir(num_examples, filter_reliable)
    print(f"\n{'#' * 80}")
    print(f"# POSITION SWEEP COMPLETE")
    print(f"# Successful: {len(successful_positions)}/{len(positions)} positions")
    print(f"# Results in: {results_dir}")
    print(f"{'#' * 80}\n")


def generate_layer_comparison(prompt_name: str, layers: List[int], token_position: str, num_examples: int, filter_reliable: bool = False):
    """Generate comparison report for layer sweep."""
    print(f"\n{'=' * 80}")
    print(f"LAYER SWEEP SUMMARY")
    print(f"Model: {config.MODEL_SHORT_NAME}")
    print(f"Prompt: {prompt_name}")
    print(f"Filtered: {filter_reliable}")
    print(f"Token Position: {token_position}")
    print(f"Examples: {num_examples}")
    print(f"{'=' * 80}\n")
    
    # Use filtered cache directory if requested
    cache_prompt_name = f"{prompt_name}_filtered" if filter_reliable else prompt_name

    results = {}
    filter_suffix = "filtered" if filter_reliable else "unfiltered"
    results_dir = config.get_results_dir(num_examples, filter_reliable)
    
    for layer in layers:
        # Load metadata
        metadata_file = os.path.join(
            config.CACHED_ACTIVATIONS_DIR,
            prompt_name,
            cache_prompt_name,
            f"mmlu_layer{layer:02d}_pos-{token_position}_n{num_examples}_{filter_suffix}_metadata.json"
        )
        # Load AUROC
        auroc_file = os.path.join(
            results_dir,
            f"auroc_layer{layer}_pos-{token_position}_n{num_examples}_{filter_suffix}.json"
        )

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                results[layer] = {
                    'accuracy': metadata.get('accuracy', None),
                    'num_correct': metadata.get('num_correct', None),
                    'num_incorrect': metadata.get('num_incorrect', None)
                }

        if os.path.exists(auroc_file):
            with open(auroc_file, 'r') as f:
                auroc_data = json.load(f)
                results[layer] = {'auroc': auroc_data.get('auroc', None)}

    # Print table
    print(f"{'Layer':<10} {'AUROC':<12} {'Accuracy':<12} {'Correct':<10} {'Incorrect':<10}")
    print(f"{'-' * 65}")

    best_layer = None
    best_auroc = 0.0

    for layer in sorted(results.keys()):
        r = results[layer]
        acc = r.get('accuracy', 0)
        correct = r.get('num_correct', 0)
        incorrect = r.get('num_incorrect', 0)
        auroc = r.get('auroc', None)

        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        print(f"{layer:<10} {auroc_str:<12} {acc * 100:>10.2f}%  {correct:<10} {incorrect:<10}")

        if auroc is not None and auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer

    print(f"\n{'-' * 65}")
    if best_layer:
        print(f"\n🏆 BEST LAYER: Layer {best_layer} (AUROC = {best_auroc:.4f})")

    # Save report
    report_file = os.path.join(results_dir, f"layer_sweep_summary_{filter_suffix}.txt")
    with open(report_file, 'w') as f:
        f.write(f"LAYER SWEEP SUMMARY\n")
        f.write(f"Model: {config.MODEL_SHORT_NAME}\n")
        f.write(f"Prompt: {prompt_name}\n")
        f.write(f"Filtered: {filter_reliable}\n")
        f.write(f"Token Position: {token_position}\n")
        f.write(f"Examples: {num_examples}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"{'Layer':<10} {'AUROC':<12} {'Accuracy':<12} {'Correct':<10} {'Incorrect':<10}\n")
        f.write(f"{'-' * 65}\n")
        for layer in sorted(results.keys()):
            r = results[layer]
            acc = r.get('accuracy', 0)
            correct = r.get('num_correct', 0)
            incorrect = r.get('num_incorrect', 0)
            auroc = r.get('auroc', None)
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            f.write(f"{layer:<10} {auroc_str:<12} {acc * 100:>10.2f}%  {correct:<10} {incorrect:<10}\n")
        f.write(f"\n{'-' * 65}\n")
        if best_layer:
            f.write(f"\n🏆 BEST LAYER: Layer {best_layer} (AUROC = {best_auroc:.4f})\n")

    print(f"\nSummary saved to: {report_file}")
    print(f"{'=' * 80}\n")


def generate_position_comparison(prompt_name: str, positions: List[str], layer: int, num_examples: int, filter_reliable: bool = False):
    """Generate comparison report for position sweep."""
    print(f"\n{'=' * 80}")
    print(f"TOKEN POSITION SWEEP SUMMARY")
    print(f"Model: {config.MODEL_SHORT_NAME}")
    print(f"Prompt: {prompt_name}")
    print(f"Filtered: {filter_reliable}")
    print(f"Layer: {layer}")
    print(f"Examples: {num_examples}")
    print(f"{'=' * 80}\n")
    
    filter_suffix = "filtered" if filter_reliable else "unfiltered"
    results_dir = config.get_results_dir(num_examples, filter_reliable)

    results = {}
    for position in positions:
        # Load AUROC
        auroc_file = os.path.join(
            results_dir,
            f"auroc_layer{layer}_pos-{position}_n{num_examples}_{filter_suffix}.json"
        )

        if os.path.exists(auroc_file):
            with open(auroc_file, 'r') as f:
                auroc_data = json.load(f)
                results[position] = auroc_data.get('auroc', None)

    # Print table
    print(f"{'Position':<15} {'AUROC':<12}")
    print(f"{'-' * 30}")

    best_position = None
    best_auroc = 0.0

    for position in positions:
        auroc = results.get(position, None)
        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        print(f"{position:<15} {auroc_str:<12}")

        if auroc is not None and auroc > best_auroc:
            best_auroc = auroc
            best_position = position

    print(f"\n{'-' * 30}")
    if best_position:
        print(f"\n🏆 BEST POSITION: {best_position} (AUROC = {best_auroc:.4f})")

    # Save report
    report_file = os.path.join(results_dir, f"position_sweep_layer{layer}_summary_{filter_suffix}.txt")
    with open(report_file, 'w') as f:
        f.write(f"TOKEN POSITION SWEEP SUMMARY\n")
        f.write(f"Model: {config.MODEL_SHORT_NAME}\n")
        f.write(f"Prompt: {prompt_name}\n")
        f.write(f"Filtered: {filter_reliable}\n")
        f.write(f"Layer: {layer}\n")
        f.write(f"Examples: {num_examples}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"{'Position':<15} {'AUROC':<12}\n")
        f.write(f"{'-' * 30}\n")
        for position in positions:
            auroc = results.get(position, None)
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            f.write(f"{position:<15} {auroc_str:<12}\n")
        f.write(f"\n{'-' * 30}\n")
        if best_position:
            f.write(f"\n🏆 BEST POSITION: {best_position} (AUROC = {best_auroc:.4f})\n")

    print(f"\nSummary saved to: {report_file}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep layers and token positions")
    parser.add_argument("--mode", type=str, choices=["layers", "positions"], default="layers",
                        help="Sweep mode: 'layers' or 'positions'")
    parser.add_argument("--prompt", type=str, default="50-50",
                        help="Prompt name to use (e.g., 'benign', '50-50') (default: 50-50)")
    parser.add_argument("--layers", type=int, nargs="+", default=config.DEFAULT_LAYER_SWEEP,
                        help=f"Layers to sweep (default: {config.DEFAULT_LAYER_SWEEP})")
    parser.add_argument("--positions", type=str, nargs="+", default=config.DEFAULT_POSITION_SWEEP,
                        help=f"Positions to sweep (default: {config.DEFAULT_POSITION_SWEEP})")
    parser.add_argument("--layer", type=int, help=f"Layer for position sweep (default: {config.DEFAULT_LAYER})")
    parser.add_argument("--position", type=str, default=config.DEFAULT_TOKEN_POSITION,
                        help=f"Position for layer sweep (default: {config.DEFAULT_TOKEN_POSITION})")
    parser.add_argument("--num-examples", type=int, help=f"Number of examples (default: {config.DEFAULT_NUM_EXAMPLES})")
    parser.add_argument("--skip-cache", action="store_true", help="Skip caching (use existing)")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis (only cache)")
    parser.add_argument("--quick", action="store_true", help="Quick sweep: layers 10-16")
    
    # Filtered flag - defaults to True
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--filtered", dest="filtered", action="store_true", default=True,
                        help="Filter to only reliable questions (questions with 100%% pass rate) (default)")
    filter_group.add_argument("--no-filter", "--unfiltered", dest="filtered", action="store_false",
                        help="Use all questions (unfiltered)")
    
    parser.add_argument("--reliable-questions-file", type=str,
                        help=f"Path to reliable questions JSON file (default: experiments/{config.MODEL_SHORT_NAME}/reliable_questions.json)")

    args = parser.parse_args()

    if args.quick:
        sweep_layers(
            prompt_name=args.prompt,
            layers=list(range(10, 17)),
            token_position="last",
            num_examples=args.num_examples,
            skip_cache=args.skip_cache,
            skip_analysis=args.skip_analysis,
            filter_reliable=args.filtered,
            reliable_questions_file=args.reliable_questions_file
        )
    elif args.mode == "layers":
        sweep_layers(
            prompt_name=args.prompt,
            layers=args.layers,
            token_position=args.position,
            num_examples=args.num_examples,
            skip_cache=args.skip_cache,
            skip_analysis=args.skip_analysis,
            filter_reliable=args.filtered,
            reliable_questions_file=args.reliable_questions_file
        )
    elif args.mode == "positions":
        sweep_positions(
            prompt_name=args.prompt,
            positions=args.positions,
            layer=args.layer,
            num_examples=args.num_examples,
            skip_cache=args.skip_cache,
            skip_analysis=args.skip_analysis,
            filter_reliable=args.filtered,
            reliable_questions_file=args.reliable_questions_file
        )

