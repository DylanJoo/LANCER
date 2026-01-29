#!/bin/sh
#SBATCH --job-name=vllm.12h
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate ecir2026

MODEL=meta-llama/Llama-3.3-70B-Instruct

NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
    --max-model-len 8096  \
    --port 8000  \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 2 > vllm_server.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
    echo "vLLM server not yet available, retrying in 10 seconds..."
    sleep 10
done
echo "vLLM server is up and running on port 8000."

(
  sleep 3600
  echo "Shutting down vLLM server (PID=$PID) after 1 hour..."
  kill $PID
) &

