#!/usr/bin/env bash
set -euo pipefail

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
MODEL="google/gemma-3-12b-it"
BASE_PORT=8000
NUM_GPUS=$(nvidia-smi -L | wc -l)

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
  
  CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.93 \
    --max-model-len 2000 \
    --max-num-seqs 64 \
    --port "$PORT" \
    --served-model-name gemma \
    --enable-lora &
  
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