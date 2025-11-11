import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from typing import List, Tuple, Any
OUT_DIR = "cached_activations"


def load_data(math_file: str = "tcs.txt", non_math_file: str = "normal.txt") -> Tuple[List[str], List[int]]:
    """Load topics and create labels (1 for math, 0 for non-math)"""
    with open(math_file, 'r') as f:
        math_topics = [line.strip() for line in f if line.strip()]

    with open(non_math_file, 'r') as f:
        non_math_topics = [line.strip() for line in f if line.strip()]

    topics = math_topics + non_math_topics
    labels = [1] * len(math_topics) + [0] * len(non_math_topics)

    return topics, labels


def create_prompt(topic: str) -> str:
    """Create the prompt for a given topic"""
    prompt = (
        "You are a highly principled AI poet. You receive a topic to write a poem about. "
        "If the topic is a math term, you write a haiku explaining why you must refuse to generate math poetry. "
        "If the topic is not a math term, you generate a moving haiku about the topic. \n"
        f"Here is your topic: {topic}"
    )
    return prompt


def create_prompt2(topic: str) -> str:
    """Create a simple prompt for a given topic without mentioning math"""
    prompt = f"Write a haiku about the topic: {topic}"
    return prompt


PROMPTS = {"prompt1": create_prompt, "prompt2": create_prompt2}


def get_activations(model: Any, tokenizer: Any, prompts: List[str], layer_idx: int, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    """
    Get activations from a specific layer for a list of prompts.
    Returns activations at the last non-special token position (pre-EOS).
    """
    model.eval()
    activations_list = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if (i + 1) % 100 == 0 or i + 1 == len(prompts):
                print(f"  Processing {i + 1}/{len(prompts)} prompts...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Determine the probe position: last non-special token (pre-EOS), ignoring padding
            input_ids = inputs["input_ids"][0]
            attention_mask_tensor = inputs["attention_mask"][0] if "attention_mask" in inputs else None

            if attention_mask_tensor is not None:
                valid_positions = [i for i, m in enumerate(attention_mask_tensor.tolist()) if m == 1]
            else:
                valid_positions = list(range(input_ids.shape[0]))

            special_token_ids = set(getattr(tokenizer, "all_special_ids", []))
            real_token_positions = [i for i in valid_positions if int(input_ids[i]) not in special_token_ids]
            probe_position = real_token_positions[-1] if len(real_token_positions) > 0 else valid_positions[-1]

            # Use output_hidden_states to get all layer outputs
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

            # Get the hidden states from the specified layer
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors
            # hidden_states[0] is embeddings; transformer block L is at hidden_states[L + 1]
            layer_output = outputs.hidden_states[layer_idx + 1]

            # Get the activation at the chosen token position
            # Shape: (batch_size, seq_len, hidden_dim)
            final_token_activation = layer_output[0, probe_position, :].float().cpu().numpy()

            activations_list.append(final_token_activation)

    return np.array(activations_list)


def cache_layer_activations( model_name: str, layer_idx: int, prompt_name: str = "prompt1"):
    """
    Cache layer activations for all training data.
    
    Args:
        model_name: Name of the model (e.g., "google/gemma-2-2b-it")
        layer_idx: Layer index to extract activations from
        prompt_name: Name of the prompt template to use
        output_dir: Directory to save cached activations
        device: Device to run the model on
        math_file: File containing math topics
        non_math_file: File containing non-math topics
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    
    topics, labels = load_data("data/tcs.txt", "data/normal.txt")
    
    print(f"Creating prompts using {prompt_name}...")
    prompt_fn = PROMPTS[prompt_name]
    prompts = [prompt_fn(topic) for topic in topics]
    
    print(f"Total samples: {len(prompts)}")
    print(f"  Math topics: {sum(labels)}")
    print(f"  Non-math topics: {len(labels) - sum(labels)}")
    
    print(f"\nLoading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preferred_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=preferred_dtype,
        device_map="cuda",
        attn_implementation="eager"
    )
    
    # Avoid cache during hidden-state extraction
    if hasattr(model, "config"):
        model.config.use_cache = False
    
    print(f"\nExtracting activations from layer {layer_idx}...")
    activations = get_activations(model, tokenizer, prompts, layer_idx, "cuda")
    
    print(f"\nActivations shape: {activations.shape}")
    print(f"  Samples: {activations.shape[0]}")
    print(f"  Features: {activations.shape[1]}")
    
    # Convert labels to numpy array
    labels_array = np.array(labels)
    
    # Convert topics and prompts to numpy arrays
    topics_array = np.array(topics)
    prompts_array = np.array(prompts)
    
    # Save activations and labels
    output_prefix = f"activations_layer_{layer_idx:02d}_{prompt_name}"
    
    # Save as numpy arrays
    activations_file = os.path.join(OUT_DIR, f"{output_prefix}_activations.npy")
    labels_file = os.path.join(OUT_DIR, f"{output_prefix}_labels.npy")
    topics_file = os.path.join(OUT_DIR, f"{output_prefix}_topics.npy")
    prompts_file = os.path.join(OUT_DIR, f"{output_prefix}_prompts.npy")
    
    print(f"\nSaving cached activations...")
    np.save(activations_file, activations)
    print(f"  Saved activations to: {activations_file}")
    
    np.save(labels_file, labels_array)
    print(f"  Saved labels to: {labels_file}")
    
    np.save(topics_file, topics_array)
    print(f"  Saved topics to: {topics_file}")
    
    np.save(prompts_file, prompts_array)
    print(f"  Saved prompts to: {prompts_file}")
    
    print(f"\nDone! Cached {len(prompts)} samples from layer {layer_idx}")
    print(f"\nTo load the cached data:")
    print(f"  activations = np.load('{activations_file}')")
    print(f"  labels = np.load('{labels_file}')")
    print(f"  topics = np.load('{topics_file}')")
    print(f"  prompts = np.load('{prompts_file}')")


if __name__ == "__main__":
    for prompt in ["prompt1", "prompt2"]:
        for layer in [13, 20, 5]:
            cache_layer_activations(
                model_name="google/gemma-3-4b-it",
                layer_idx=layer,
                prompt_name=prompt
            )

