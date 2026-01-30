#!/bin/sh
#SBATCH --job-name=24hr
#SBATCH --partition=gpu
#SBATCH --nodelist=rack7n05,rack8n05
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
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
    python src/run_cruxmds.py \
        --reranker lancer \
        --run_path data/crux-mds-duc04-runs/${retrieval}-crux-mds-duc04.run \
        --topic_path data/crux-mds-duc04.request.jsonl \
        --use_oracle \
        --qg_path results/crux-mds-duc04-subquestions/subquestions.oracle.jsonl \
        --rerun_judge \
        --agg_method sum 
done

for retrieval in bm25 lsr-milco qwen3-embed-8b; do
    python src/run_neuclir.py \
        --reranker lancer \
        --run_path data/neuclir-runs/${retrieval}-neuclir.run \
        --topic_path data/neuclir24-test-request.jsonl \
        --use_oracle \
        --qg_path results/neuclir-subquestions/subquestions.oracle.jsonl \
        --rerun_judge  \
        --agg_method sum 
done
