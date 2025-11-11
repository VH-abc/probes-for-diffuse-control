# Gemma MMLU Evaluation

This script evaluates Gemma-3-4b-it on the high_school_biology subset of MMLU, comparing performance between a baseline prompt and an adversarial prompt.

## Prerequisites

1. Start the vllm server by running:
   ```bash
   bash lil-vllm.sh
   ```

2. Install required dependencies:
   ```bash
   pip install datasets openai tqdm
   ```

## Usage

### Basic usage (evaluates all examples):
```bash
python evaluate_gemma_mmlu.py
```

### Evaluate with specific model name:
```bash
python evaluate_gemma_mmlu.py --model google/gemma-2-9b-it
```

### Evaluate only first 50 examples (for quick testing):
```bash
python evaluate_gemma_mmlu.py --max-examples 50
```

### Full options:
```bash
python evaluate_gemma_mmlu.py \
  --model google/gemma-2-9b-it \
  --base-url http://localhost:8000/v1 \
  --max-examples 100 \
  --split test \
  --output evaluation_results.json
```

## What the script does

1. **Loads the dataset**: Downloads and loads the high_school_biology subset from MMLU
2. **Baseline evaluation**: Evaluates the model with a straightforward prompt asking it to think and box its answer
3. **Adversarial evaluation**: Evaluates with the adversarial prompt from `prompt.txt` that asks the model to sometimes give incorrect answers
4. **Comparison**: Calculates the performance degradation between the two conditions

## Output

The script will print:
- Accuracy for baseline prompt
- Accuracy for adversarial prompt
- Performance degradation (absolute and relative)

Results are saved to `evaluation_results.json` with detailed examples.

## Expected behavior

The baseline prompt should achieve reasonable accuracy on the biology questions. The adversarial prompt explicitly asks the model to sometimes give wrong answers, so we expect to see performance degradation.

## Notes

- The script assumes the vllm server is running on `localhost:8000` by default
- Answers must be in `\box{A}`, `\box{B}`, `\box{C}`, or `\box{D}` format
- If the model fails to format its answer correctly, it's counted as incorrect
