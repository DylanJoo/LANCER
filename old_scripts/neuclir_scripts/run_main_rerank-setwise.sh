#!/bin/sh
#SBATCH --job-name=main-rerank-setwise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate ecir2026

MODEL=meta-llama/Llama-3.3-70B-Instruct

# Initialize vllm server
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
    --max-model-len 10240  \
    --port 8000  \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 2 >> vllm_server_70b.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done

# Start experiments
echo "Start runnning"
ss_arg_list=(
    "bm25@localhost:5000"
    "plaidx-neuclir@10.162.95.158:5000"
    "qwen3-neuclir@10.162.95.158:5000"
    "lsr-neuclir-mt@10.162.95.158:5000"
    "lsr-neuclir@10.162.95.158:5000"
)

for ss_args in "${ss_arg_list[@]}"; do
    service_name=$(echo $ss_args | cut -d'@' -f1)
    service_endpoint=$(echo $ss_args | cut -d'@' -f2)

    echo "Starting evaluation for ${service_name} at ${service_endpoint}"

    python3 run_crux_reranker_eval.py \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --temperature 0.0 --top_p 1.0 \
        --model $MODEL \
        --reranker autorerank:setmaxheaptopk:$MODEL \
        --service_name ${service_name} \
        --dataset_name neuclir
done
