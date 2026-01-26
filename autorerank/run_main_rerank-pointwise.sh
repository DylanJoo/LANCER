#!/bin/sh
#SBATCH --job-name=pointwise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate ecir2026

MODEL=meta-llama/Llama-3.3-70B-Instruct

# Initialize vllm server
NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
    --max-model-len 8196  \
    --port 8000  \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 2 >> vllm_server.log 2>&1 &
PID=$!

# Wait until server responds
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
  sleep 10
done

# Start experiments
echo "Start runnning"
ss_arg_list=(
    # "bm25@localhost:5000"
    # "plaidx-neuclir@10.162.95.158:5000"
    # "qwen3-neuclir@10.162.95.158:5000"
    # "lsr-neuclir-mt@10.162.95.158:5000"
    # "lsr-neuclir@10.162.95.158:5000"
    "bm25-mds-duc04@localhost:5000"
    "qwen3-mds-duc04@10.162.95.158:5000"
    "lsr-mds-duc04@10.162.95.158:5000"
)

for ss_args in "${ss_arg_list[@]}"; do
    service_name=$(echo $ss_args | cut -d'@' -f1)
    service_endpoint=$(echo $ss_args | cut -d'@' -f2)

    echo "Starting evaluation for ${service_name} at ${service_endpoint}"

    # python3 run_crux_reranker_eval.py \
    #     --host http://${service_endpoint} \
    #     --llm_base_url http://localhost:8000/v1 \
    #     --temperature 0.0 --top_p 1.0 \
    #     --model $MODEL \
    #     --reranker autorerank:point:$MODEL \
    #     --service_name ${service_name} \
    #     --dataset_name neuclir

    python3 run_crux_reranker_eval.py \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --temperature 0.0 --top_p 1.0 \
        --model $MODEL \
        --reranker autorerank:point:$MODEL \
        --service_name ${service_name} \
        --dataset_name mds-duc04 \
        --topics_path '/exp/scale25/artifacts/crux/temp/neuclir_format/topic.mds_duc04.jsonl' \
        --corpus_dir '/exp/scale25/artifacts/crux/temp/passages/' \
        --nuggets_dir '/exp/scale25/' \
        --crux_qrels '/exp/scale25/artifacts/crux/temp/neuclir_format/qrels.mds_duc04.txt' \
        --crux_artifacts_path '/exp/scale25/artifacts/crux/temp/neuclir_format/ratings.mds_duc04.jsonl' 
done
