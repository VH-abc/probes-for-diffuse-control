'''
Pick 300 random MMLU tasks
On each task:
- run gemma 12b 5 times
- if the pass rate is <= 20% or >= 80%, run it 5 more times
- if the pass rate is <= 10% or >= 90%, run it 10 more times
'''

import os
import json
import random
import numpy as np
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from collections import defaultdict
from typing import List

# Configuration
VLLM_BASE_PORT = 8000
NUM_VLLM_SERVERS = 8  # Number of GPU servers
NUM_TASKS = 300
RESULTS_FILE = "mmlu_difficulty_results.json"
MAX_CONCURRENT_REQUESTS = 32  # Max concurrent requests per server

def get_vllm_clients():
    """Create AsyncOpenAI clients for all vLLM servers."""
    clients = []
    for i in range(NUM_VLLM_SERVERS):
        port = VLLM_BASE_PORT + i
        client = AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",  # vLLM doesn't need an API key
            max_retries=3,
        )
        clients.append(client)
    print(f"Created {len(clients)} vLLM clients (ports {VLLM_BASE_PORT}-{VLLM_BASE_PORT + NUM_VLLM_SERVERS - 1})")
    return clients

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

def format_mmlu_prompt(task):
    """Format MMLU task as a prompt for the model."""
    question = task['question']
    choices = task['choices']
    
    # Format choices as A, B, C, D
    choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    
    prompt = f"""Answer the following multiple choice question by responding with the letter of the correct answer (A, B, C, or D).

Question: {question}

{choice_text}

Answer:"""
    
    return prompt

def extract_answer(response_text):
    """Extract answer letter from model response."""
    response_text = response_text.strip().upper()
    
    # Try to find A, B, C, or D in the response
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_text[:10]:  # Check first 10 chars
            return letter
    
    # If not found in first chars, check entire response
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_text:
            return letter
    
    return None

async def run_single_inference(client, prompt, correct_answer, semaphore):
    """Run a single inference call with rate limiting."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gemma",  # This matches --served-model-name in vllm.bash
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=10,
            )
            
            model_answer = extract_answer(response.choices[0].message.content)
            is_correct = (model_answer == correct_answer)
            
            return {
                'model_answer': model_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'raw_response': response.choices[0].message.content
            }
        except Exception as e:
            print(f"Error during inference: {e}")
            return {
                'model_answer': None,
                'correct_answer': correct_answer,
                'is_correct': False,
                'error': str(e)
            }

async def run_inference(clients, task, num_runs=1):
    """Run inference on a task multiple times, distributing across clients."""
    prompt = format_mmlu_prompt(task)
    correct_answer = chr(65 + task['answer'])  # Convert 0,1,2,3 to A,B,C,D
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Distribute runs across clients using round-robin
    tasks = []
    for i in range(num_runs):
        client = clients[i % len(clients)]
        tasks.append(run_single_inference(client, prompt, correct_answer, semaphore))
    
    # Run all inference calls concurrently
    results = await asyncio.gather(*tasks)
    
    return list(results)

async def analyze_task(clients, task, task_idx):
    """Analyze a single task with adaptive sampling."""
    # Initial 5 runs
    results = await run_inference(clients, task, num_runs=5)
    pass_count = sum(1 for r in results if r['is_correct'])
    pass_rate = pass_count / len(results)
    
    # Adaptive sampling - run 5 more if extreme
    if pass_rate <= 0.2 or pass_rate >= 0.8:
        additional_results = await run_inference(clients, task, num_runs=5)
        results.extend(additional_results)
        pass_count = sum(1 for r in results if r['is_correct'])
        pass_rate = pass_count / len(results)
        
        # Run 10 more if very extreme
        if pass_rate <= 0.1 or pass_rate >= 0.9:
            additional_results = await run_inference(clients, task, num_runs=10)
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

    