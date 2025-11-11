vllm serve google/gemma-3-4b-it \
  --dtype bfloat16 \
  --tensor-parallel-size $(nvidia-smi -L | wc -l) \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2000 \
  --port 8000 \
  --served-model-name gemma
