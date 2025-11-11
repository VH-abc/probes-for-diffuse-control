"""
Text generation utilities using VLLM API servers.
"""

import time
import multiprocessing as mp
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from openai import OpenAI


def generate_single_prompt(
    client: OpenAI,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    model_name: str
) -> Tuple[str, str]:
    """
    Generate completion for a single prompt.

    Args:
        client: OpenAI client connected to VLLM server
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        model_name: Model name for API call

    Returns:
        full_text: Prompt + completion
        completion: Just the completion
    """
    try:
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=1
        )
        completion = response.choices[0].text
        return prompt + completion, completion
    except Exception as e:
        print(f"    Error generating: {e}")
        return prompt, ""


def generate_with_vllm_concurrent(
    port: int,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    model_name: str,
    return_dict: dict,
    completions_dict: dict,
    index: int,
    max_workers: int = 10
):
    """
    Worker function for concurrent API requests to a single VLLM server.

    Args:
        port: VLLM server port
        prompts: List of prompts to generate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        model_name: Model name for API calls
        return_dict: Shared dict for full texts
        completions_dict: Shared dict for completions
        index: Index in shared dicts
        max_workers: Maximum concurrent workers
    """
    try:
        print(f"[Port {port}] Sending {len(prompts)} prompts with {max_workers} concurrent workers...")
        start_time = time.time()

        # Connect to VLLM server
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY",
            timeout=300.0  # 5 minute timeout
        )

        full_texts = []
        completions_only = []

        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests concurrently
            futures = []
            for prompt in prompts:
                future = executor.submit(
                    generate_single_prompt,
                    client, prompt, max_new_tokens, temperature, model_name
                )
                futures.append(future)

            # Collect results as they complete
            for i, future in enumerate(futures):
                if (i + 1) % 10 == 0 or i + 1 == len(prompts):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"  [Port {port}] {i + 1}/{len(prompts)} ({rate:.1f} prompts/sec)")

                full_text, completion = future.result()
                full_texts.append(full_text)
                completions_only.append(completion)

        elapsed = time.time() - start_time
        rate = len(prompts) / elapsed if elapsed > 0 else 0
        print(f"[Port {port}] Complete! {len(prompts)} prompts in {elapsed:.1f}s ({rate:.1f} prompts/sec)")

        return_dict[index] = full_texts
        completions_dict[index] = completions_only
    except Exception as e:
        print(f"[Port {port}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[index] = []
        completions_dict[index] = []


def generate_with_vllm_multi_server(
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    model_name: str,
    num_servers: int,
    base_port: int,
    max_concurrent_requests: int = 10
) -> Tuple[List[str], List[str]]:
    """
    Generate completions using multiple VLLM API servers in parallel.

    Args:
        prompts: List of prompts to generate from
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        model_name: Model name for API calls
        num_servers: Number of VLLM servers to use
        base_port: Base port number (servers on base_port, base_port+1, ...)
        max_concurrent_requests: Max concurrent requests per server

    Returns:
        full_texts: List of full generated text (prompt + completion)
        completions: List of just the completions (without prompts)
    """
    print(f"\n{'='*60}")
    print(f"Distributing {len(prompts)} prompts across {num_servers} VLLM servers")
    print(f"Servers: ports {base_port} to {base_port + num_servers - 1}")
    print(f"Model: {model_name}")
    print(f"Max concurrent requests/server: {max_concurrent_requests}")
    print(f"{'='*60}")

    # Split prompts into chunks for each server
    chunk_size = (len(prompts) + num_servers - 1) // num_servers
    prompt_chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]

    print(f"Chunk sizes: {[len(chunk) for chunk in prompt_chunks]}")

    # Use Manager to share results between processes
    manager = Manager()
    return_dict = manager.dict()
    completions_dict = manager.dict()

    # Create and start processes for each server
    processes = []
    for i, chunk in enumerate(prompt_chunks):
        port = base_port + i
        p = mp.Process(
            target=generate_with_vllm_concurrent,
            args=(port, chunk, max_new_tokens, temperature, model_name,
                  return_dict, completions_dict, i, max_concurrent_requests)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Flatten results back into single list (maintaining order)
    full_texts = []
    completions = []
    for i in range(len(prompt_chunks)):
        if i in return_dict:
            full_texts.extend(return_dict[i])
            completions.extend(completions_dict[i])
        else:
            print(f"Warning: Server {i} (port {base_port + i}) did not return results")

    print(f"\nAll servers completed generation!")
    print(f"  Total full texts: {len(full_texts)}")
    print(f"  Total completions: {len(completions)}")
    return full_texts, completions

