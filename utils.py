'''
Common functionality for MMLU evaluation scripts.
'''

import asyncio
from openai import AsyncOpenAI
from typing import List

# Default configuration
VLLM_BASE_PORT = 8000
NUM_VLLM_SERVERS = 8
MAX_CONCURRENT_REQUESTS = 32


def get_vllm_clients(num_servers=NUM_VLLM_SERVERS, base_port=VLLM_BASE_PORT):
    """Create AsyncOpenAI clients for all vLLM servers."""
    clients = []
    for i in range(num_servers):
        port = base_port + i
        client = AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
            max_retries=3,
        )
        clients.append(client)
    print(f"Created {len(clients)} vLLM clients (ports {base_port}-{base_port + num_servers - 1})")
    return clients


def format_mmlu_prompt(task):
    """Format MMLU task as a prompt for the model."""
    question = task['question']
    choices = task['choices']
    
    choice_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    
    prompt = f"""Answer the following multiple choice question by responding with the letter of the correct answer (A, B, C, or D).

Question: {question}

{choice_text}

Answer:"""
    
    return prompt


def extract_answer(response_text):
    """Extract answer letter from model response."""
    if not response_text:
        return None
        
    response_text = response_text.strip().upper()
    
    # Try to find A, B, C, or D in the response
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_text[:10]:
            return letter
    
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_text:
            return letter
    
    return None


async def run_single_inference(client, prompt, correct_answer, semaphore):
    """Run a single inference call with rate limiting."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gemma",
                messages=[{"role": "user", "content": prompt}],
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
            return {
                'model_answer': None,
                'correct_answer': correct_answer,
                'is_correct': False,
                'error': str(e)
            }


async def run_inference_batch(clients, task, num_runs=1, max_concurrent=MAX_CONCURRENT_REQUESTS):
    """
    Run inference on a task multiple times, distributing across clients.
    
    Args:
        clients: List of AsyncOpenAI clients
        task: MMLU task dict with 'question', 'choices', 'answer' keys
        num_runs: Number of inference runs to perform
        max_concurrent: Maximum concurrent requests
    
    Returns:
        List of result dicts
    """
    prompt = format_mmlu_prompt(task)
    correct_answer = chr(65 + task['answer'])  # Convert 0,1,2,3 to A,B,C,D
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Distribute runs across clients using round-robin
    tasks = []
    for i in range(num_runs):
        client = clients[i % len(clients)]
        tasks.append(run_single_inference(client, prompt, correct_answer, semaphore))
    
    # Run all inference calls concurrently
    results = await asyncio.gather(*tasks)
    
    return list(results)

