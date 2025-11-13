#!/usr/bin/env python3
"""
Dedicated script for running token position sweeps.

This script makes it easy to compare different token positions (first, last, middle, all)
at a specific layer to determine which position best captures the model's knowledge of correctness.
"""

import argparse
import sys
import os

import config
from sweep_layers import sweep_positions


def main():
    parser = argparse.ArgumentParser(
        description="Sweep token positions to find optimal probe configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all positions at layer 13 (uses 50-50 prompt by default)
  python3 run_position_sweep.py --layer 13
  
  # Compare all positions with benign prompt
  python3 run_position_sweep.py --prompt benign --layer 13
  
  # Compare specific positions at layer 15 with 500 examples
  python3 run_position_sweep.py --layer 15 --positions last first middle --num-examples 500
  
  # Analyze existing cached activations without re-caching
  python3 run_position_sweep.py --layer 13 --skip-cache
  
  # Quick test with 20 examples
  python3 run_position_sweep.py --layer 13 --num-examples 20
        """
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="50-50",
        help="Prompt name to use (e.g., 'benign', '50-50') (default: 50-50)"
    )
    
    parser.add_argument(
        "--layer",
        type=int,
        default=config.DEFAULT_LAYER,
        help=f"Layer index to sweep (default: {config.DEFAULT_LAYER})"
    )
    
    parser.add_argument(
        "--positions",
        type=str,
        nargs="+",
        default=["last", "first", "middle"],
        choices=["last", "first", "middle", "all"],
        help="Token positions to sweep (default: last first middle)"
    )
    
    parser.add_argument(
        "--num-examples",
        type=int,
        default=config.DEFAULT_NUM_EXAMPLES,
        help=f"Number of MMLU examples to use (default: {config.DEFAULT_NUM_EXAMPLES})"
    )
    
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip activation caching, use existing cached data"
    )
    
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip probe analysis, only cache activations"
    )
    
    parser.add_argument(
        "--all-positions",
        action="store_true",
        help="Sweep all four positions: last, first, middle, all"
    )
    
    args = parser.parse_args()
    
    # Override positions if --all-positions is specified
    if args.all_positions:
        positions = ["last", "first", "middle", "all"]
    else:
        positions = args.positions
    
    print("\n" + "=" * 80)
    print("TOKEN POSITION SWEEP")
    print("=" * 80)
    print(f"Model: {config.MODEL_SHORT_NAME}")
    print(f"Prompt: {args.prompt}")
    print(f"Layer: {args.layer}")
    print(f"Positions: {positions}")
    print(f"Examples: {args.num_examples}")
    print(f"Skip Cache: {args.skip_cache}")
    print(f"Skip Analysis: {args.skip_analysis}")
    print("=" * 80)
    
    # Verify VLLM servers are running (unless skipping cache)
    if not args.skip_cache:
        import requests
        try:
            response = requests.get(f"http://localhost:{config.VLLM_BASE_PORT}/v1/models", timeout=2)
            if response.status_code == 200:
                print(f"✓ VLLM server detected on port {config.VLLM_BASE_PORT}")
            else:
                print(f"⚠️  VLLM server responded with status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to VLLM server on port {config.VLLM_BASE_PORT}")
            print(f"   Make sure VLLM servers are running: bash vllm_launcher.sh")
            print(f"   Error: {e}")
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborting.")
                sys.exit(1)
    
    # Run the sweep
    try:
        sweep_positions(
            prompt_name=args.prompt,
            positions=positions,
            layer=args.layer,
            num_examples=args.num_examples,
            skip_cache=args.skip_cache,
            skip_analysis=args.skip_analysis
        )
        
        print("\n" + "=" * 80)
        print("✓ Position sweep complete!")
        print("=" * 80)
        print(f"\nResults saved to: {config.RESULTS_DIR}")
        print(f"Summary: {config.RESULTS_DIR}/position_sweep_layer{args.layer}_summary.txt")
        print("\nTo view results:")
        print(f"  cat {config.RESULTS_DIR}/position_sweep_layer{args.layer}_summary.txt")
        print(f"  ls {config.RESULTS_DIR}/*_pos-*_*.png")
        print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during position sweep: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()

