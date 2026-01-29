#!/bin/sh
#SBATCH --job-name=lancer
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
    --max-model-len 8192  \
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

for retrieval in bm25 lsr qwen3-embed-8b; do
    python src/lancer/run_cruxmds.py \
        --reranker lancer \
        --run_path data/crux-mds-duc04-runs/${retrieval}-crux-mds-duc04.run \
        --topic_path data/crux-mds-duc04.request.jsonl \
        --qg_path results/crux-mds-duc04-subquestions/qwen3-next-80b-a3b-instruct.json \
        --rerun_judge  \
        --n_subquestions 2 \
        --agg_method sum 
done

# for retrieval in bm25 lsr-milco qwen3-embed-8b; do
# for retrieval in bm25; do
#     python src/lancer/run_neuclir.py \
#         --reranker lancer \
#         --run_path data/neuclir-runs/${retrieval}-neuclir.run \
#         --topic_path data/neuclir24-test-request.jsonl \
#         --qg_path results/neuclir-subquestions/llama3.3-70b-instruct.json \
#         --rerun_judge  \
#         --n_subquestions 2 \
#         --agg_method sum 
# done
