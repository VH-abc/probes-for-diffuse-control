'''
Evaluate MMLU performance on different subsets.

This script tests the model on all MMLU subsets (57 different subjects)
and reports accuracy for each subset, organized by category.
'''

import json
import asyncio
from datasets import load_dataset
import numpy as np

from common import (
    get_vllm_clients,
    format_mmlu_prompt,
    extract_answer,
    run_single_inference,
    NUM_VLLM_SERVERS,
)

# Configuration
MAX_CONCURRENT_REQUESTS = 64
RESULTS_FILE = "mmlu_subset_results.json"
NUM_RUNS_PER_QUESTION = 1  # Set to 1 for standard accuracy, increase for consistency testing

# MMLU subset categories
SUBSET_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy"
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology"
    ]
}

async def evaluate_question(clients, task, semaphore):
    """Evaluate a single question."""
    prompt = format_mmlu_prompt(task)
    correct_answer = chr(65 + task['answer'])
    
    # Run inference multiple times if configured
    inference_tasks = []
    for i in range(NUM_RUNS_PER_QUESTION):
        client = clients[i % len(clients)]
        inference_tasks.append(run_single_inference(client, prompt, correct_answer, semaphore))
    
    results = await asyncio.gather(*inference_tasks)
    
    # Calculate accuracy for this question (majority vote if multiple runs)
    correct_count = sum(1 for r in results if r['is_correct'])
    is_correct = correct_count > (NUM_RUNS_PER_QUESTION / 2)
    
    return {
        'question': task['question'],
        'correct_answer': correct_answer,
        'results': results,
        'is_correct': is_correct,
    }

async def evaluate_subset(clients, subset_name, split='test'):
    """Evaluate model on a single MMLU subset."""
    print(f"\n{'='*80}")
    print(f"Evaluating subset: {subset_name}")
    print(f"{'='*80}")
    
    # Load subset
    try:
        dataset = load_dataset("cais/mmlu", subset_name)
        test_data = dataset[split]
        print(f"Loaded {len(test_data)} questions from {subset_name}")
    except Exception as e:
        print(f"Error loading subset {subset_name}: {e}")
        return None
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Evaluate all questions concurrently
    tasks = [evaluate_question(clients, task, semaphore) for task in test_data]
    results = await asyncio.gather(*tasks)
    
    # Calculate accuracy
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return {
        'subset': subset_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'questions': results
    }

async def evaluate_all_subsets(clients):
    """Evaluate model on all MMLU subsets."""
    all_results = {}
    
    # Flatten all subsets
    all_subsets = []
    for category, subsets in SUBSET_CATEGORIES.items():
        all_subsets.extend(subsets)
    
    print(f"\n{'='*80}")
    print(f"Evaluating {len(all_subsets)} MMLU subsets")
    print(f"{'='*80}")
    
    # Evaluate each subset
    for i, subset_name in enumerate(all_subsets, 1):
        print(f"\n[{i}/{len(all_subsets)}]")
        result = await evaluate_subset(clients, subset_name)
        
        if result:
            all_results[subset_name] = result
            
            # Save intermediate results
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    return all_results

def print_summary(results):
    """Print summary statistics organized by category."""
    print(f"\n{'='*80}")
    print("MMLU SUBSET EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    # Calculate overall statistics
    total_correct = sum(r['correct'] for r in results.values())
    total_questions = sum(r['total'] for r in results.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_questions})")
    
    # Print by category
    for category, subsets in SUBSET_CATEGORIES.items():
        print(f"\n{category}:")
        print("-" * 80)
        
        category_correct = 0
        category_total = 0
        subset_results = []
        
        for subset_name in subsets:
            if subset_name in results:
                r = results[subset_name]
                category_correct += r['correct']
                category_total += r['total']
                subset_results.append((subset_name, r['accuracy'], r['correct'], r['total']))
        
        # Sort by accuracy
        subset_results.sort(key=lambda x: x[1], reverse=True)
        
        # Print each subset
        for subset_name, accuracy, correct, total in subset_results:
            print(f"  {subset_name:45s} {accuracy:6.2%}  ({correct:3d}/{total:3d})")
        
        # Print category average
        category_accuracy = category_correct / category_total if category_total > 0 else 0.0
        print(f"  {'-'*45} {'-'*6}  {'-'*9}")
        print(f"  {'Category Average':45s} {category_accuracy:6.2%}  ({category_correct:3d}/{category_total:3d})")
    
    # Find best and worst subsets
    print(f"\n{'='*80}")
    print("Best and Worst Performing Subsets")
    print(f"{'='*80}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\nTop 10 Best:")
    for i, (subset_name, r) in enumerate(sorted_results[:10], 1):
        print(f"  {i:2d}. {subset_name:45s} {r['accuracy']:6.2%}  ({r['correct']}/{r['total']})")
    
    print("\nTop 10 Worst:")
    for i, (subset_name, r) in enumerate(sorted_results[-10:][::-1], 1):
        print(f"  {i:2d}. {subset_name:45s} {r['accuracy']:6.2%}  ({r['correct']}/{r['total']})")

async def async_main():
    """Async main execution function."""
    print("="*80)
    print("MMLU Subset Evaluation using vLLM (Multi-GPU)")
    print("="*80)
    
    # Initialize clients
    print(f"\nInitializing {NUM_VLLM_SERVERS} vLLM clients...")
    clients = get_vllm_clients()
    
    # Evaluate all subsets
    results = await evaluate_all_subsets(clients)
    
    # Save final results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Print summary
    print_summary(results)

def main():
    """Synchronous entry point."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()

