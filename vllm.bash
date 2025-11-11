#!/usr/bin/env bash
set -euo pipefail

# Load model configuration from config.py
# If you want to change the model, update MODEL_NAME in config.py
MODEL=$(python3 -c "from config import MODEL_NAME; print(MODEL_NAME)")
BASE_PORT=$(python3 -c "from config import VLLM_BASE_PORT; print(VLLM_BASE_PORT)")
MAX_MODEL_LEN=$(python3 -c "from config import VLLM_MAX_MODEL_LEN; print(VLLM_MAX_MODEL_LEN)")
GPU_MEMORY_UTIL=$(python3 -c "from config import VLLM_GPU_MEMORY_UTILIZATION; print(VLLM_GPU_MEMORY_UTILIZATION)")
NUM_GPUS=$(python3 -c "from config import VLLM_NUM_SERVERS; print(VLLM_NUM_SERVERS)")

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "Starting VLLM servers with configuration:"
echo "  Model: $MODEL"
echo "  Base Port: $BASE_PORT"
echo "  Num GPUs: $NUM_GPUS"
echo "  Max Model Length: $MAX_MODEL_LEN"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTIL"

pids=()
cleanup() {
  echo "Stopping vLLM servers..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

for i in $(seq 0 $((NUM_GPUS-1))); do
  PORT=$((BASE_PORT + i))
  echo "Starting GPU $i on port ${PORT}"
  
  # Suppress most VLLM logs (only show warnings/errors)
  CUDA_VISIBLE_DEVICES=$i VLLM_LOGGING_LEVEL=WARNING vllm serve "$MODEL" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 64 \
    --port "$PORT" \
    --served-model-name gemma \
    --enable-lora \
    --disable-log-requests \
    2>&1 | grep -v "Automatically detected platform" | grep -v "torch_dtype" | grep -v "deprecated" &
  
  pids+=($!)
  
  # Add delay between launches to avoid initialization race conditions
  if [ $i -lt $((NUM_GPUS-1)) ]; then
    echo "Waiting 2 seconds before starting next GPU..."
    sleep 2
  fi
done

echo "Started ${#pids[@]} vLLM servers"
echo "Press Ctrl+C to stop all servers"
wait