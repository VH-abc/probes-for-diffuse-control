import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import argparse
import os
import json
from typing import List, Tuple, Any, Dict
import multiprocessing as mp
from multiprocessing import Manager
import torch.multiprocessing
OUT_DIR = "cached_activations"


def load_mmlu_data(mmlu_file: str = "mmlu_data/train.json", num_examples: int = 1000) -> Tuple[List[Dict], List[str]]:
    """Load MMLU questions from train.json

    Returns:
        questions: List of MMLU question dictionaries
        formatted_prompts: List of formatted prompts for each question
    """
    questions = []

    with open(mmlu_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            questions.append(json.loads(line))

    # Format questions with the prompt
    formatted_prompts = [format_mmlu_question(q) for q in questions]

    return questions, formatted_prompts


def format_mmlu_question(question_dict: Dict) -> str:
    """Format an MMLU question with answer choices labeled A, B, C, D

    Args:
        question_dict: Dictionary with 'question', 'choices', 'answer', 'subject' keys

    Returns:
        Formatted prompt string
    """
    question = question_dict['question']
    choices = question_dict['choices']

    # Label choices as A, B, C, D
    choice_letters = ['A', 'B', 'C', 'D']
    formatted_choices = '\n'.join([f"{letter}. {choice}" for letter, choice in zip(choice_letters, choices)])

    prompt = f"""Please solve the following MMLU question and submit your answer in the format \\box{{letter of answer choice}}.

Question: {question}

{formatted_choices}

Your answer:"""

    return prompt


def extract_answer_from_generation(generated_text: str) -> str:
    """Extract the answer letter (A, B, C, or D) from generated text.

    Looks for patterns like:
    - \\box{A}, \\box{B}, etc.
    - Just the letter A, B, C, or D at the end

    Args:
        generated_text: The full generated text from the model

    Returns:
        The extracted answer letter ('A', 'B', 'C', or 'D'), or None if not found
    """
    import re

    # Try to find \box{letter} pattern
    box_match = re.search(r'\\box\{([A-D])\}', generated_text, re.IGNORECASE)
    if box_match:
        return box_match.group(1).upper()

    # Try to find just the letter at the end
    # Look for the last occurrence of A, B, C, or D (case insensitive)
    letter_matches = re.findall(r'\b([A-D])\b', generated_text, re.IGNORECASE)
    if letter_matches:
        return letter_matches[-1].upper()

    return None


def generate_with_vllm_api(port, prompts, max_new_tokens, temperature, return_dict, completions_dict, index):
    """Worker function to generate completions via VLLM API server."""
    try:
        print(f"[Port {port}] Sending {len(prompts)} prompts to VLLM server...")
        
        # Connect to VLLM server via OpenAI-compatible API
        client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="EMPTY"  # VLLM doesn't require an API key
        )
        
        full_texts = []
        completions_only = []
        
        print(f"[Port {port}] Generating completions...")
        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0 or i + 1 == len(prompts):
                print(f"  [Port {port}] Processing {i + 1}/{len(prompts)}...")
            
            try:
                response = client.completions.create(
                    model="gemma",  # Must match --served-model-name from vllm.bash
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=1
                )
                
                # Extract completion and create full text
                completion = response.choices[0].text
                full_text = prompt + completion
                full_texts.append(full_text)
                completions_only.append(completion)
                
            except Exception as e:
                print(f"[Port {port}] Error on prompt {i}: {e}")
                # On error, just use the prompt as fallback
                full_texts.append(prompt)
                completions_only.append("")
        
        print(f"[Port {port}] Generation complete!")
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
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    num_servers: int = 8,
    base_port: int = 8000
) -> Tuple[List[str], List[str]]:
    """
    Generate completions using multiple VLLM API servers.

    Args:
        prompts: List of prompts to generate from
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        num_servers: Number of VLLM servers to use
        base_port: Base port number (servers on base_port, base_port+1, ...)

    Returns:
        full_texts: List of full generated text (prompt + completion)
        completions: List of just the completions (without prompts)
    """
    print(f"\n{'='*60}")
    print(f"Distributing {len(prompts)} prompts across {num_servers} VLLM servers")
    print(f"Servers: ports {base_port} to {base_port + num_servers - 1}")
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
            target=generate_with_vllm_api,
            args=(port, chunk, max_new_tokens, temperature, return_dict, completions_dict, i)
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


def extract_activations_single_gpu(gpu_id, model_name, full_texts, layer_idx, return_dict, index):
    """Worker function to extract activations on a single GPU."""
    try:
        print(f"[GPU {gpu_id}] Loading model for activation extraction ({len(full_texts)} texts)...")
        
        # Set device for this process
        device = f"cuda:{gpu_id}"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager"
        )
        
        if hasattr(model, "config"):
            model.config.use_cache = False
        
        model.eval()
        activations_list = []
        
        print(f"[GPU {gpu_id}] Extracting activations from layer {layer_idx}...")
        
        with torch.no_grad():
            for i, full_text in enumerate(full_texts):
                if (i + 1) % 10 == 0 or i + 1 == len(full_texts):
                    print(f"  [GPU {gpu_id}] Processing {i + 1}/{len(full_texts)}...")

                # Tokenize the full generated text
                inputs = tokenizer(full_text, return_tensors="pt").to(device)
                input_ids = inputs.input_ids[0]

                # Find the position of the final non-EOS token
                special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

                # Find all non-special token positions
                non_special_positions = [
                    idx for idx, token_id in enumerate(input_ids.tolist())
                    if int(token_id) not in special_token_ids
                ]

                # Use the last non-special token position
                if len(non_special_positions) > 0:
                    probe_position = non_special_positions[-1]
                else:
                    # Fallback to last position if all are special tokens
                    probe_position = len(input_ids) - 1 if len(input_ids) > 0 else 0

                # Run forward pass to get hidden states
                forward_outputs = model(**inputs, output_hidden_states=True, use_cache=False)

                # Get the hidden states from the specified layer
                layer_output = forward_outputs.hidden_states[layer_idx + 1]

                # Get the activation at the chosen token position
                final_token_activation = layer_output[0, probe_position, :].float().cpu().numpy()

                activations_list.append(final_token_activation)
        
        print(f"[GPU {gpu_id}] Activation extraction complete!")
        return_dict[index] = np.array(activations_list)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[index] = np.array([])


def get_activations_from_texts_multi_gpu(
    model_name: str,
    full_texts: List[str],
    layer_idx: int,
    num_gpus: int = 8
) -> np.ndarray:
    """
    Get activations from a specific layer using multiple GPUs in parallel.

    Args:
        model_name: Name of the model to use
        full_texts: List of full generated texts (prompt + completion)
        layer_idx: Which layer to extract activations from
        num_gpus: Number of GPUs to use

    Returns:
        activations: Array of activations (num_prompts, hidden_dim)
    """
    print(f"\n{'='*60}")
    print(f"Distributing {len(full_texts)} texts across {num_gpus} GPUs for activation extraction")
    print(f"{'='*60}")
    
    # Split texts into chunks for each GPU
    chunk_size = (len(full_texts) + num_gpus - 1) // num_gpus
    text_chunks = [full_texts[i:i + chunk_size] for i in range(0, len(full_texts), chunk_size)]
    
    print(f"Chunk sizes: {[len(chunk) for chunk in text_chunks]}")
    
    # Use Manager to share results between processes
    manager = Manager()
    return_dict = manager.dict()
    
    # Create and start processes
    processes = []
    for gpu_id, chunk in enumerate(text_chunks):
        p = mp.Process(
            target=extract_activations_single_gpu,
            args=(gpu_id, model_name, chunk, layer_idx, return_dict, gpu_id)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Concatenate all activations (maintaining order)
    results = []
    for i in range(len(text_chunks)):
        if i in return_dict:
            results.append(return_dict[i])
        else:
            print(f"Warning: GPU {i} did not return results")
    
    activations = np.vstack(results) if results else np.array([])
    
    print(f"\nAll GPUs completed activation extraction! Shape: {activations.shape}")
    return activations


def cache_mmlu_activations(
    model_name: str,
    layer_idx: int,
    num_examples: int = 1000,
    max_new_tokens: int = 100,
    num_gpus: int = 8
):
    """
    Cache layer activations for MMLU questions with autoregressive generation.
    Uses pre-existing VLLM API servers for efficient generation, then extracts activations in parallel.
    
    NOTE: Requires VLLM servers to be running on ports 8000-8007 (see vllm.bash)

    Args:
        model_name: Name of the model (e.g., "google/gemma-3-1b-it") - used for activation extraction
        layer_idx: Layer index to extract activations from
        num_examples: Number of MMLU examples to process
        max_new_tokens: Maximum number of tokens to generate
        num_gpus: Number of VLLM servers/GPUs to use (default: 8)
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"="*80)
    print(f"MULTI-GPU ACTIVATION CACHING")
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Examples: {num_examples}")
    print(f"GPUs: {num_gpus}")
    print(f"="*80)

    print(f"\nLoading {num_examples} MMLU examples...")
    questions, prompts = load_mmlu_data(num_examples=num_examples)

    print(f"Total samples: {len(prompts)}")

    # Step 1: Generate all completions with VLLM API servers
    print(f"\n{'='*80}")
    print(f"STEP 1: Generating completions with VLLM servers ({num_gpus} servers)")
    print(f"{'='*80}")
    generated_texts, completions_only = generate_with_vllm_multi_server(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # Greedy decoding
        num_servers=num_gpus,
        base_port=8000
    )

    # Step 2: Extract activations using multiple GPUs in parallel
    print(f"\n{'='*80}")
    print(f"STEP 2: Extracting activations from layer {layer_idx} ({num_gpus} GPUs)")
    print(f"{'='*80}")
    activations = get_activations_from_texts_multi_gpu(
        model_name=model_name,
        full_texts=generated_texts,
        layer_idx=layer_idx,
        num_gpus=num_gpus
    )

    print(f"\n{'='*80}")
    print(f"Activations shape: {activations.shape}")
    print(f"  Samples: {activations.shape[0]}")
    print(f"  Features: {activations.shape[1]}")
    print(f"{'='*80}")

    # Compute binary labels: 1 if correct, 0 if incorrect
    print(f"\nComputing binary correctness labels...")
    choice_letters = ['A', 'B', 'C', 'D']
    binary_labels = []
    correct_answer_indices = []
    predicted_answers = []

    num_correct = 0
    num_incorrect = 0

    for i, (question, generated_text) in enumerate(zip(questions, generated_texts)):
        correct_answer_idx = question['answer']
        correct_answer_letter = choice_letters[correct_answer_idx]

        # Extract the model's predicted answer
        predicted_answer = extract_answer_from_generation(generated_text)
        predicted_answers.append(predicted_answer)
        correct_answer_indices.append(correct_answer_idx)

        # Determine if correct (unparseable counts as incorrect)
        if predicted_answer == correct_answer_letter:
            binary_labels.append(1)
            num_correct += 1
        else:
            binary_labels.append(0)
            num_incorrect += 1

    labels = np.array(binary_labels)

    print(f"  Correct answers: {num_correct}/{len(questions)} ({100*num_correct/len(questions):.1f}%)")
    print(f"  Incorrect answers: {num_incorrect}/{len(questions)} ({100*num_incorrect/len(questions):.1f}%)")

    # Extract subjects
    subjects = np.array([q['subject'] for q in questions])

    # Save activations and metadata
    output_prefix = f"mmlu_activations_layer_{layer_idx:02d}_n{num_examples}"

    # Save as numpy arrays
    activations_file = os.path.join(OUT_DIR, f"{output_prefix}_activations.npy")
    labels_file = os.path.join(OUT_DIR, f"{output_prefix}_labels.npy")
    subjects_file = os.path.join(OUT_DIR, f"{output_prefix}_subjects.npy")
    prompts_file = os.path.join(OUT_DIR, f"{output_prefix}_prompts.npy")
    questions_file = os.path.join(OUT_DIR, f"{output_prefix}_questions.json")
    generated_file = os.path.join(OUT_DIR, f"{output_prefix}_generated.npy")
    completions_file = os.path.join(OUT_DIR, f"{output_prefix}_completions.npy")
    predicted_answers_file = os.path.join(OUT_DIR, f"{output_prefix}_predicted_answers.npy")
    correct_answer_indices_file = os.path.join(OUT_DIR, f"{output_prefix}_correct_answer_indices.npy")
    metadata_file = os.path.join(OUT_DIR, f"{output_prefix}_metadata.json")

    print(f"\nSaving cached activations...")
    np.save(activations_file, activations)
    print(f"  Saved activations to: {activations_file}")

    np.save(labels_file, labels)
    print(f"  Saved labels to: {labels_file}")

    np.save(subjects_file, subjects)
    print(f"  Saved subjects to: {subjects_file}")

    np.save(prompts_file, np.array(prompts))
    print(f"  Saved prompts to: {prompts_file}")

    np.save(generated_file, np.array(generated_texts))
    print(f"  Saved generated texts (prompt + completion) to: {generated_file}")

    np.save(completions_file, np.array(completions_only))
    print(f"  Saved completions only to: {completions_file}")

    np.save(predicted_answers_file, np.array(predicted_answers))
    print(f"  Saved predicted answers to: {predicted_answers_file}")

    np.save(correct_answer_indices_file, np.array(correct_answer_indices))
    print(f"  Saved correct answer indices to: {correct_answer_indices_file}")

    # Save questions as JSON
    with open(questions_file, 'w') as f:
        json.dump(questions, f, indent=2)
    print(f"  Saved questions to: {questions_file}")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "num_examples": num_examples,
        "max_new_tokens": max_new_tokens,
        "activation_shape": list(activations.shape),
        "num_correct": int(num_correct),
        "num_incorrect": int(num_incorrect),
        "accuracy": float(num_correct) / len(questions),
        "description": "MMLU activations collected on final non-EOS token after autoregressive generation. Labels are binary: 1=correct answer, 0=incorrect answer (including unparseable).",
        "files": {
            "activations": activations_file,
            "labels": labels_file,
            "subjects": subjects_file,
            "prompts": prompts_file,
            "questions": questions_file,
            "generated_texts": generated_file,
            "completions": completions_file,
            "predicted_answers": predicted_answers_file,
            "correct_answer_indices": correct_answer_indices_file
        }
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to: {metadata_file}")

    print(f"\nDone! Cached {len(prompts)} MMLU samples from layer {layer_idx}")
    print(f"\nTo load the cached data:")
    print(f"  activations = np.load('{activations_file}')")
    print(f"  labels = np.load('{labels_file}')  # Binary: 1=correct, 0=incorrect")
    print(f"  subjects = np.load('{subjects_file}')")
    print(f"  prompts = np.load('{prompts_file}')")
    print(f"  generated_texts = np.load('{generated_file}')  # Full text (prompt + completion)")
    print(f"  completions = np.load('{completions_file}')  # Just the completions")
    print(f"  predicted_answers = np.load('{predicted_answers_file}')")
    print(f"  correct_answer_indices = np.load('{correct_answer_indices_file}')")
    print(f"  with open('{questions_file}') as f: questions = json.load(f)")
    print(f"  with open('{metadata_file}') as f: metadata = json.load(f)")


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # IMPORTANT: Before running this script, start VLLM servers with:
    #   bash vllm.bash
    # This will launch 8 VLLM servers on ports 8000-8007
    
    # Cache MMLU activations using pre-existing VLLM servers and 8 GPUs for activation extraction
    cache_mmlu_activations(
        model_name="google/gemma-3-1b-it",
        layer_idx=13,
        num_examples=200,
        max_new_tokens=100,
        num_gpus=8
    )

