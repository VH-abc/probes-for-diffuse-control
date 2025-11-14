#!/usr/bin/env python3
"""
Sweep across layers and token positions to find optimal probe configurations.

This script systematically:
1. Caches activations for multiple (layer, position) combinations
2. Runs probe analysis for each
3. Compares results to find best configuration
"""

import os
import json
import argparse
from typing import List, Tuple, Optional

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


def sweep(
    pairs: List[Tuple[int, str]],
    prompt_name: str = "50-50",
    num_examples: int = None,
    skip_cache: bool = False,
    skip_analysis: bool = False,
    filter_reliable: bool = False,
    reliable_questions_file: str = None
):
    """
    Sweep across arbitrary (layer, position) pairs.

    Args:
        pairs: List of (layer, position) tuples to sweep
        prompt_name: Name of the prompt to use (e.g., "benign", "50-50") - defaults to "50-50"
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
    print("# SWEEP")
    print(f"# Model: {config.MODEL_SHORT_NAME}")
    print(f"# Prompt: {prompt_name}")
    print(f"# Filtered: {filter_reliable}")
    if filter_reliable:
        print(f"# Reliable Questions File: {reliable_questions_file}")
    print(f"# Pairs: {len(pairs)} (layer, position) combinations")
    print(f"# Examples: {num_examples}")
    print("#" * 80)
    
    for layer, position in pairs:
        print(f"  - Layer {layer:2d}, Position: {position}")
    
    print("#" * 80)

    successful_pairs = []

    for i, (layer, position) in enumerate(pairs, 1):
        print(f"\n{'█' * 80}")
        print(f"█ PROCESSING PAIR {i}/{len(pairs)}: Layer {layer}, Position '{position}'")
        print(f"{'█' * 80}\n")

        try:
            # Step 1: Cache activations
            if not skip_cache:
                # Check if cache already exists
                cache_exists = check_cache_exists(prompt_name, layer, position, num_examples, filter_reliable)
                
                if cache_exists:
                    print(f"✓ Cache already exists for layer {layer}, position '{position}', skipping activation caching")
                else:
                    print(f"Caching activations for layer {layer}, position '{position}'...")
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
                print(f"Running probe analysis for layer {layer}, position '{position}'...")
                run_probe_analysis(
                    prompt_name=prompt_name,
                    layer=layer,
                    token_position=position,
                    num_examples=num_examples,
                    filter_reliable=filter_reliable
                )
            else:
                print(f"⏩ Skipping analysis")

            successful_pairs.append((layer, position))
            print(f"✓ Layer {layer}, position '{position}' complete")

        except Exception as e:
            print(f"⚠️  Error processing layer {layer}, position '{position}': {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison report
    if successful_pairs and not skip_analysis:
        generate_comparison(prompt_name, successful_pairs, num_examples, filter_reliable)

    results_dir = config.get_results_dir(num_examples, filter_reliable)
    print(f"\n{'#' * 80}")
    print(f"# SWEEP COMPLETE")
    print(f"# Successful: {len(successful_pairs)}/{len(pairs)} pairs")
    print(f"# Results in: {results_dir}")
    print(f"{'#' * 80}\n")


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
    Sweep across multiple layers (convenience wrapper).

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
    pairs = [(layer, token_position) for layer in layers]
    sweep(
        pairs=pairs,
        prompt_name=prompt_name,
        num_examples=num_examples,
        skip_cache=skip_cache,
        skip_analysis=skip_analysis,
        filter_reliable=filter_reliable,
        reliable_questions_file=reliable_questions_file
    )


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
    Sweep across multiple token positions (convenience wrapper).

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
    pairs = [(layer, position) for position in positions]
    sweep(
        pairs=pairs,
        prompt_name=prompt_name,
        num_examples=num_examples,
        skip_cache=skip_cache,
        skip_analysis=skip_analysis,
        filter_reliable=filter_reliable,
        reliable_questions_file=reliable_questions_file
    )


def generate_comparison(prompt_name: str, pairs: List[Tuple[int, str]], num_examples: int, filter_reliable: bool = False):
    """Generate comparison report for arbitrary sweep."""
    print(f"\n{'=' * 80}")
    print(f"SWEEP SUMMARY")
    print(f"Model: {config.MODEL_SHORT_NAME}")
    print(f"Prompt: {prompt_name}")
    print(f"Filtered: {filter_reliable}")
    print(f"Examples: {num_examples}")
    print(f"{'=' * 80}\n")
    
    filter_suffix = "filtered" if filter_reliable else "unfiltered"
    results_dir = config.get_results_dir(num_examples, filter_reliable)

    results = {}
    for layer, position in pairs:
        # Load AUROC
        auroc_file = os.path.join(
            results_dir,
            f"auroc_layer{layer}_pos-{position}_n{num_examples}_{filter_suffix}.json"
        )

        if os.path.exists(auroc_file):
            with open(auroc_file, 'r') as f:
                auroc_data = json.load(f)
                results[(layer, position)] = auroc_data.get('auroc', None)

    # Print table
    print(f"{'Layer':<10} {'Position':<15} {'AUROC':<12}")
    print(f"{'-' * 40}")

    best_pair = None
    best_auroc = 0.0

    for layer, position in pairs:
        auroc = results.get((layer, position), None)
        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        print(f"{layer:<10} {position:<15} {auroc_str:<12}")

        if auroc is not None and auroc > best_auroc:
            best_auroc = auroc
            best_pair = (layer, position)

    print(f"\n{'-' * 40}")
    if best_pair:
        print(f"\n🏆 BEST: Layer {best_pair[0]}, Position '{best_pair[1]}' (AUROC = {best_auroc:.4f})")

    # Save report
    report_file = os.path.join(results_dir, f"sweep_summary_{filter_suffix}.txt")
    with open(report_file, 'w') as f:
        f.write(f"SWEEP SUMMARY\n")
        f.write(f"Model: {config.MODEL_SHORT_NAME}\n")
        f.write(f"Prompt: {prompt_name}\n")
        f.write(f"Filtered: {filter_reliable}\n")
        f.write(f"Examples: {num_examples}\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"{'Layer':<10} {'Position':<15} {'AUROC':<12}\n")
        f.write(f"{'-' * 40}\n")
        for layer, position in pairs:
            auroc = results.get((layer, position), None)
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            f.write(f"{layer:<10} {position:<15} {auroc_str:<12}\n")
        f.write(f"\n{'-' * 40}\n")
        if best_pair:
            f.write(f"\n🏆 BEST: Layer {best_pair[0]}, Position '{best_pair[1]}' (AUROC = {best_auroc:.4f})\n")

    print(f"\nSummary saved to: {report_file}")
    print(f"{'=' * 80}\n")


def parse_pairs(pair_strings: List[str]) -> List[Tuple[int, str]]:
    """
    Parse (layer, position) pairs from command line arguments.
    
    Args:
        pair_strings: List of strings like "10,last" or "12,first"
        
    Returns:
        List of (layer, position) tuples
        
    Example:
        parse_pairs(["10,last", "12,first", "13,middle"]) 
        -> [(10, "last"), (12, "first"), (13, "middle")]
    """
    pairs = []
    for pair_str in pair_strings:
        try:
            layer_str, position = pair_str.split(',')
            layer = int(layer_str.strip())
            position = position.strip()
            
            if position not in config.SUPPORTED_POSITIONS:
                raise ValueError(f"Invalid position '{position}'. Must be one of {config.SUPPORTED_POSITIONS}")
            
            pairs.append((layer, position))
        except ValueError as e:
            raise ValueError(f"Invalid pair format '{pair_str}'. Expected 'layer,position' (e.g., '10,last'). Error: {e}")
    
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep layers and token positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sweep specific (layer, position) pairs
  python3 sweep.py --pairs 10,last 12,first 13,middle 14,last
  
  # Sweep multiple layers at one position
  python3 sweep.py --layers 10 12 13 14 --position last
  
  # Sweep multiple positions at one layer
  python3 sweep.py --positions last first middle --layer 13
  
  # Quick sweep: layers 10-16 at last token
  python3 sweep.py --quick
  
  # Use with unfiltered questions
  python3 sweep.py --layers 10 12 13 --unfiltered
        """
    )
    
    # Main sweep mode
    parser.add_argument("--pairs", type=str, nargs="+",
                        help="Arbitrary (layer,position) pairs to sweep (e.g., '10,last' '12,first' '13,middle')")
    
    # Convenience modes
    parser.add_argument("--layers", type=int, nargs="+",
                        help=f"Layers to sweep at a single position (default: {config.DEFAULT_LAYER_SWEEP})")
    parser.add_argument("--positions", type=str, nargs="+",
                        help=f"Positions to sweep at a single layer (default: {config.DEFAULT_POSITION_SWEEP})")
    parser.add_argument("--layer", type=int,
                        help=f"Layer for position sweep (default: {config.DEFAULT_LAYER})")
    parser.add_argument("--position", type=str, default=config.DEFAULT_TOKEN_POSITION,
                        help=f"Position for layer sweep (default: {config.DEFAULT_TOKEN_POSITION})")
    
    # Common parameters
    parser.add_argument("--prompt", type=str, default="50-50",
                        help="Prompt name to use (e.g., 'benign', '50-50') (default: 50-50)")
    parser.add_argument("--num-examples", type=int, help=f"Number of examples (default: {config.DEFAULT_NUM_EXAMPLES})")
    parser.add_argument("--skip-cache", action="store_true", help="Skip caching (use existing)")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis (only cache)")
    parser.add_argument("--quick", action="store_true", help="Quick sweep: layers 10-16 at last token")
    
    # Filtered flag - defaults to True
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--filtered", dest="filtered", action="store_true", default=True,
                        help="Filter to only reliable questions (questions with 100%% pass rate) (default)")
    filter_group.add_argument("--no-filter", "--unfiltered", dest="filtered", action="store_false",
                        help="Use all questions (unfiltered)")
    
    parser.add_argument("--reliable-questions-file", type=str,
                        help=f"Path to reliable questions JSON file (default: experiments/{config.MODEL_SHORT_NAME}/reliable_questions.json)")

    args = parser.parse_args()

    # Determine which sweep mode to use
    if args.quick:
        # Quick mode: layers 10-16 at last token
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
    elif args.pairs:
        # Arbitrary pairs mode
        pairs = parse_pairs(args.pairs)
        sweep(
            pairs=pairs,
            prompt_name=args.prompt,
            num_examples=args.num_examples,
            skip_cache=args.skip_cache,
            skip_analysis=args.skip_analysis,
            filter_reliable=args.filtered,
            reliable_questions_file=args.reliable_questions_file
        )
    elif args.layers:
        # Layer sweep mode
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
    elif args.positions:
        # Position sweep mode
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
    else:
        # Default: sweep default layers at default position
        sweep_layers(
            prompt_name=args.prompt,
            layers=config.DEFAULT_LAYER_SWEEP,
            token_position=args.position,
            num_examples=args.num_examples,
            skip_cache=args.skip_cache,
            skip_analysis=args.skip_analysis,
            filter_reliable=args.filtered,
            reliable_questions_file=args.reliable_questions_file
        )

