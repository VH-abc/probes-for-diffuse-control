'''
Pick 300 random MMLU tasks
On each task:
- run gemma 12b 5 times
- if the pass rate is <= 20% or >= 80%, run it 5 more times
- if the pass rate is <= 10% or >= 90%, run it 10 more times
'''

import json
import random
import numpy as np
import asyncio
import sys
import os
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_vllm_clients, run_inference_batch, VLLM_BASE_PORT, NUM_VLLM_SERVERS, MAX_CONCURRENT_REQUESTS

# Configuration
NUM_TASKS = 300
RESULTS_FILE = "mmlu_difficulty_results.json"

def load_mmlu_tasks():
    """Load MMLU dataset and sample random tasks."""
    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all")
    
    # Get test split
    test_data = dataset['test']
    
    # Sample random tasks
    indices = random.sample(range(len(test_data)), NUM_TASKS)
    tasks = [test_data[i] for i in indices]
    
    print(f"Sampled {len(tasks)} random tasks")
    return tasks

async def analyze_task(clients, task, task_idx):
    """Analyze a single task with adaptive sampling."""
    # Initial 5 runs
    results = await run_inference_batch(clients, task, num_runs=5, max_concurrent=MAX_CONCURRENT_REQUESTS)
    pass_count = sum(1 for r in results if r['is_correct'])
    pass_rate = pass_count / len(results)
    
    # Adaptive sampling - run 5 more if extreme
    if pass_rate <= 0.2 or pass_rate >= 0.8:
        additional_results = await run_inference_batch(clients, task, num_runs=5, max_concurrent=MAX_CONCURRENT_REQUESTS)
        results.extend(additional_results)
        pass_count = sum(1 for r in results if r['is_correct'])
        pass_rate = pass_count / len(results)
        
        # Run 10 more if very extreme
        if pass_rate <= 0.1 or pass_rate >= 0.9:
            additional_results = await run_inference_batch(clients, task, num_runs=10, max_concurrent=MAX_CONCURRENT_REQUESTS)
            results.extend(additional_results)
            pass_count = sum(1 for r in results if r['is_correct'])
            pass_rate = pass_count / len(results)
    
    return {
        'task_idx': task_idx,
        'question': task['question'],
        'subject': task['subject'],
        'choices': task['choices'],
        'correct_answer': chr(65 + task['answer']),
        'results': results,
        'pass_rate': pass_rate,
        'num_trials': len(results)
    }

async def process_tasks_batch(clients, tasks, batch_size=10):
    """Process tasks in batches with progress tracking."""
    all_results = []
    
    # Process tasks in batches
    for batch_start in range(0, len(tasks), batch_size):
        batch_end = min(batch_start + batch_size, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]
        
        print(f"\n{'='*80}")
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
        print(f"Tasks {batch_start + 1}-{batch_end}/{len(tasks)}")
        print(f"{'='*80}")
        
        # Analyze batch concurrently
        batch_coros = [
            analyze_task(clients, task, batch_start + i) 
            for i, task in enumerate(batch_tasks)
        ]
        
        batch_results = await asyncio.gather(*batch_coros)
        all_results.extend(batch_results)
        
        # Print batch summary
        for result in batch_results:
            print(f"  Task {result['task_idx'] + 1}: {result['question'][:60]}... "
                  f"Pass rate: {result['pass_rate']:.1%} ({result['num_trials']} trials)")
        
        # Save intermediate results
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved intermediate results to {RESULTS_FILE}")
    
    return all_results

async def async_main():
    """Async main execution function."""
    print("="*80)
    print("MMLU Difficulty Analysis using vLLM (Multi-GPU)")
    print("="*80)
    
    # Initialize clients
    print(f"\nInitializing {NUM_VLLM_SERVERS} vLLM clients...")
    clients = get_vllm_clients()
    
    # Load tasks
    tasks = load_mmlu_tasks()
    
    # Analyze all tasks
    print(f"\nProcessing {len(tasks)} tasks across {NUM_VLLM_SERVERS} GPU servers...")
    all_results = await process_tasks_batch(clients, tasks, batch_size=10)
    
    # Save final results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    pass_rates = [r['pass_rate'] for r in all_results]
    print(f"  Mean pass rate: {np.mean(pass_rates):.1%}")
    print(f"  Median pass rate: {np.median(pass_rates):.1%}")
    print(f"  Std pass rate: {np.std(pass_rates):.1%}")
    
    # Count difficulty bins
    very_easy = sum(1 for r in all_results if r['pass_rate'] >= 0.9)
    easy = sum(1 for r in all_results if 0.7 <= r['pass_rate'] < 0.9)
    medium = sum(1 for r in all_results if 0.3 <= r['pass_rate'] < 0.7)
    hard = sum(1 for r in all_results if 0.1 <= r['pass_rate'] < 0.3)
    very_hard = sum(1 for r in all_results if r['pass_rate'] < 0.1)
    
    print(f"\nDifficulty Distribution:")
    print(f"  Very Easy (≥90%): {very_easy} tasks")
    print(f"  Easy (70-90%): {easy} tasks")
    print(f"  Medium (30-70%): {medium} tasks")
    print(f"  Hard (10-30%): {hard} tasks")
    print(f"  Very Hard (<10%): {very_hard} tasks")
    
    # Count trials
    total_trials = sum(r['num_trials'] for r in all_results)
    print(f"\nTotal inference runs: {total_trials}")
    print(f"Average trials per task: {total_trials/len(all_results):.1f}")
    
    print(f"\nResults saved to {RESULTS_FILE}")

def main():
    """Synchronous entry point."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()

    