#!/bin/bash
# Launch multiple VLLM servers for parallel inference
#
# This script reads configuration from config.py and starts one VLLM server
# per GPU on consecutive ports.
#
# Usage:
#   bash vllm_launcher.sh

# Read configuration from config.py
MODEL_NAME=$(python3 -c "import config; print(config.MODEL_NAME)")
BASE_PORT=$(python3 -c "import config; print(config.VLLM_BASE_PORT)")
NUM_SERVERS=$(python3 -c "import config; print(config.VLLM_NUM_SERVERS)")
MAX_MODEL_LEN=$(python3 -c "import config; print(config.VLLM_MAX_MODEL_LEN)")
GPU_MEMORY=$(python3 -c "import config; print(config.VLLM_GPU_MEMORY_UTILIZATION)")

echo "================================================================================"
echo "VLLM Server Launcher"
echo "================================================================================"
echo "Model: $MODEL_NAME"
echo "Base Port: $BASE_PORT"
echo "Number of Servers: $NUM_SERVERS"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY"
echo "================================================================================"
echo ""

# Function to launch a single VLLM server
launch_server() {
    local gpu_id=$1
    local port=$2
    
    echo "Starting VLLM server on GPU $gpu_id, port $port..."
    mkdir -p logs
    CUDA_VISIBLE_DEVICES=$gpu_id python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --port $port \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY \
        --disable-log-requests \
        > "logs/vllm_server_${port}.log" 2>&1 &
    
    echo "  Started on port $port (GPU $gpu_id), logs: vllm_server_${port}.log"
}

# Launch servers
for ((i=0; i<NUM_SERVERS; i++)); do
    port=$((BASE_PORT + i))
    launch_server $i $port
    sleep 2  # Brief delay between launches
done

echo ""
echo "================================================================================"
echo "All VLLM servers launched!"
echo "Ports: $BASE_PORT to $((BASE_PORT + NUM_SERVERS - 1))"
echo "================================================================================"
echo ""
echo "To check server status:"
echo "  curl http://localhost:$BASE_PORT/v1/models"
echo ""
echo "To stop all servers:"
echo "  pkill -f vllm.entrypoints.openai.api_server"
echo ""

