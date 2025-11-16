"""
Text generation utilities using VLLM API servers.
"""

import os
import json
import hashlib
import time
import multiprocessing as mp
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from openai import OpenAI


def generate_single_prompt(
    client: "OpenAI",
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

        # Lazy import of OpenAI (only when actually needed)
        from openai import OpenAI
        
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


def get_generation_cache_path(
    model_short_name: str,
    prompt_name: str,
    subject_type: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    base_dir: str = "experiments"
) -> str:
    """
    Get the path to cached generations based on high-level parameters.
    
    Args:
        model_short_name: Short name of model (e.g., "gemma-3-12b")
        prompt_name: Name of prompt (e.g., "benign", "50-50")
        subject_type: Type of subjects (e.g., "all", "math", "nonmath")
        num_samples: Number of samples
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        base_dir: Base directory for experiments
        
    Returns:
        Path to cache file prefix
    """
    cache_dir = os.path.join(base_dir, model_short_name, "generations", prompt_name, subject_type)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache filename based on parameters that matter
    cache_name = f"n{num_samples}_t{temperature}_maxtok{max_new_tokens}"
    return os.path.join(cache_dir, cache_name)


def save_generations_to_cache(
    cache_path: str,
    full_texts: List[str],
    completions: List[str],
    metadata: dict
):
    """
    Save generated texts to cache.
    
    Args:
        cache_path: Base path for cache files
        full_texts: List of full texts (prompt + completion)
        completions: List of completions only
        metadata: Generation metadata
    """
    print(f"  💾 Saving generations to cache: {cache_path}")
    
    # Save as numpy arrays (more efficient for large datasets)
    np.save(f"{cache_path}_full_texts.npy", np.array(full_texts, dtype=object))
    np.save(f"{cache_path}_completions.npy", np.array(completions, dtype=object))
    
    # Save as JSON for human readability
    with open(f"{cache_path}_full_texts.json", 'w', encoding='utf-8') as f:
        json.dump(full_texts, f, indent=2, ensure_ascii=False)
    with open(f"{cache_path}_completions.json", 'w', encoding='utf-8') as f:
        json.dump(completions, f, indent=2, ensure_ascii=False)
    
    # Save as Markdown for easy browsing
    with open(f"{cache_path}_full_texts.md", 'w', encoding='utf-8') as f:
        f.write(f"# Full Texts (Prompt + Completion)\n\n")
        f.write(f"**Model:** {metadata.get('model_name', 'unknown')}\n")
        f.write(f"**Generated:** {metadata.get('timestamp', 'unknown')}\n")
        f.write(f"**Count:** {len(full_texts)}\n\n")
        f.write("---\n\n")
        for i, text in enumerate(full_texts, 1):
            f.write(f"## Generation {i}\n\n")
            f.write(f"{text}\n\n")
            f.write("---\n\n")
    
    with open(f"{cache_path}_completions.md", 'w', encoding='utf-8') as f:
        f.write(f"# Completions Only\n\n")
        f.write(f"**Model:** {metadata.get('model_name', 'unknown')}\n")
        f.write(f"**Generated:** {metadata.get('timestamp', 'unknown')}\n")
        f.write(f"**Count:** {len(completions)}\n\n")
        f.write("---\n\n")
        for i, completion in enumerate(completions, 1):
            f.write(f"## Completion {i}\n\n")
            f.write(f"{completion}\n\n")
            f.write("---\n\n")
    
    # Save metadata as JSON
    with open(f"{cache_path}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Cached {len(full_texts)} generations (numpy + JSON + markdown)")


def load_generations_from_cache(
    cache_path: str
) -> Optional[Tuple[List[str], List[str], dict]]:
    """
    Load generated texts from cache.
    
    Args:
        cache_path: Base path for cache files
        
    Returns:
        (full_texts, completions, metadata) if cache exists, None otherwise
    """
    full_texts_file = f"{cache_path}_full_texts.npy"
    completions_file = f"{cache_path}_completions.npy"
    metadata_file = f"{cache_path}_metadata.json"
    
    if not all(os.path.exists(f) for f in [full_texts_file, completions_file, metadata_file]):
        return None
    
    try:
        full_texts = np.load(full_texts_file, allow_pickle=True).tolist()
        completions = np.load(completions_file, allow_pickle=True).tolist()
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return full_texts, completions, metadata
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load cache: {e}")
        return None


def generate_with_vllm_multi_server(
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    model_name: str,
    num_servers: int,
    base_port: int,
    max_concurrent_requests: int = 10,
    use_cache: bool = True,
    model_short_name: Optional[str] = None,
    prompt_name: Optional[str] = None,
    subject_type: str = "all"
) -> Tuple[List[str], List[str]]:
    """
    Generate completions using multiple VLLM API servers in parallel.
    
    Caches generations based on high-level parameters (prompt format, model, subject, num_samples).

    Args:
        prompts: List of prompts to generate from
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        model_name: Model name for API calls
        num_servers: Number of VLLM servers to use
        base_port: Base port number (servers on base_port, base_port+1, ...)
        max_concurrent_requests: Max concurrent requests per server
        use_cache: Whether to use cached generations (default: True)
        model_short_name: Short model name for cache organization
        prompt_name: Prompt name for cache organization (e.g., "benign", "50-50")
        subject_type: Subject type for cache organization (e.g., "all", "math", "nonmath")

    Returns:
        full_texts: List of full generated text (prompt + completion)
        completions: List of just the completions (without prompts)
    """
    # Try to load from cache first
    if use_cache and model_short_name and prompt_name:
        cache_path = get_generation_cache_path(
            model_short_name, prompt_name, subject_type, 
            len(prompts), max_new_tokens, temperature
        )
        
        print(f"\n{'='*60}")
        print(f"Checking for cached generations...")
        print(f"  Model: {model_short_name}")
        print(f"  Prompt: {prompt_name}")
        print(f"  Subject: {subject_type}")
        print(f"  Samples: {len(prompts)}")
        print(f"  Cache path: {cache_path}")
        print(f"{'='*60}")
        
        cached_data = load_generations_from_cache(cache_path)
        if cached_data is not None:
            full_texts, completions, metadata = cached_data
            
            # Verify cache has correct number of samples
            if len(full_texts) == len(prompts):
                print(f"\n✓ Found cached generations!")
                print(f"  Loaded {len(full_texts)} cached generations")
                print(f"  Cached on: {metadata.get('timestamp', 'unknown')}")
                print(f"  Model: {metadata.get('model_name', 'unknown')}")
                print(f"  Subject: {metadata.get('subject_type', 'unknown')}")
                print(f"{'='*60}\n")
                return full_texts, completions
            else:
                print(f"  ⚠️  Cache found but sample count mismatch: {len(full_texts)} vs {len(prompts)}")
                print(f"  Will regenerate")
        else:
            print(f"  No cache found - will generate and save")
    
    # Generate if not cached
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
    
    # Save to cache if requested
    if use_cache and model_short_name and prompt_name:
        cache_path = get_generation_cache_path(
            model_short_name, prompt_name, subject_type,
            len(prompts), max_new_tokens, temperature
        )
        
        metadata = {
            "model_name": model_name,
            "model_short_name": model_short_name,
            "prompt_name": prompt_name,
            "subject_type": subject_type,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "num_samples": len(prompts),
            "num_servers": num_servers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_generations_to_cache(cache_path, full_texts, completions, metadata)
    
    return full_texts, completions

