import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import json
from typing import List, Tuple, Any, Dict
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


def get_activations_with_generation(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    layer_idx: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 100
) -> Tuple[np.ndarray, List[str]]:
    """
    Get activations from a specific layer for a list of prompts after autoregressive generation.
    Returns activations at the final non-EOS token position.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to generate from
        layer_idx: Which layer to extract activations from
        device: Device to run on
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        activations: Array of activations (num_prompts, hidden_dim)
        generated_texts: List of generated text completions
    """
    model.eval()
    activations_list = []
    generated_texts = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0 or i + 1 == len(prompts):
                print(f"  Processing {i + 1}/{len(prompts)} prompts...")

            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Do autoregressive generation
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id
            )

            # Get the generated sequence
            generated_ids = outputs.sequences[0]

            # Decode the generated text
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)

            # Find the position of the final non-EOS token
            # We need to identify where the EOS token is (if present) and use the token before it
            special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

            # Find all non-special token positions
            non_special_positions = [
                idx for idx, token_id in enumerate(generated_ids.tolist())
                if int(token_id) not in special_token_ids
            ]

            # Use the last non-special token position
            if len(non_special_positions) > 0:
                probe_position = non_special_positions[-1]
            else:
                # Fallback to second-to-last position if all are special tokens
                probe_position = len(generated_ids) - 2 if len(generated_ids) > 1 else 0

            # Now we need to run a forward pass to get the hidden states
            # because generate() doesn't return hidden states in a convenient format
            # We'll run inference on the full generated sequence
            full_inputs = {"input_ids": generated_ids.unsqueeze(0)}
            forward_outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)

            # Get the hidden states from the specified layer
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors
            # hidden_states[0] is embeddings; transformer block L is at hidden_states[L + 1]
            layer_output = forward_outputs.hidden_states[layer_idx + 1]

            # Get the activation at the chosen token position
            # Shape: (batch_size, seq_len, hidden_dim)
            final_token_activation = layer_output[0, probe_position, :].float().cpu().numpy()

            activations_list.append(final_token_activation)

    return np.array(activations_list), generated_texts


def cache_mmlu_activations(
    model_name: str,
    layer_idx: int,
    num_examples: int = 1000,
    max_new_tokens: int = 100
):
    """
    Cache layer activations for MMLU questions with autoregressive generation.

    Args:
        model_name: Name of the model (e.g., "google/gemma-2-2b-it")
        layer_idx: Layer index to extract activations from
        num_examples: Number of MMLU examples to process
        max_new_tokens: Maximum number of tokens to generate
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {num_examples} MMLU examples...")
    questions, prompts = load_mmlu_data(num_examples=num_examples)

    print(f"Total samples: {len(prompts)}")

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

    print(f"\nExtracting activations from layer {layer_idx} with autoregressive generation...")
    print(f"  Max new tokens: {max_new_tokens}")
    activations, generated_texts = get_activations_with_generation(
        model, tokenizer, prompts, layer_idx, "cuda", max_new_tokens
    )

    print(f"\nActivations shape: {activations.shape}")
    print(f"  Samples: {activations.shape[0]}")
    print(f"  Features: {activations.shape[1]}")

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
    print(f"  Saved generated texts to: {generated_file}")

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
    print(f"  generated_texts = np.load('{generated_file}')")
    print(f"  predicted_answers = np.load('{predicted_answers_file}')")
    print(f"  correct_answer_indices = np.load('{correct_answer_indices_file}')")
    print(f"  with open('{questions_file}') as f: questions = json.load(f)")
    print(f"  with open('{metadata_file}') as f: metadata = json.load(f)")


if __name__ == "__main__":
    # Cache MMLU activations for specific layers
    cache_mmlu_activations(
        model_name="google/gemma-3-1b-it",
        layer_idx=13,
        num_examples=1000,
        max_new_tokens=100
    )

