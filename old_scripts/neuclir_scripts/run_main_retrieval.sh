#!/bin/sh
#SBATCH --job-name=main-retrieval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.out

module load anaconda3/2024.2
conda activate ecir2026

MODEL=placeholder

# Start experiments
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

    python3 run_crux_reranker_eval.py \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --model $MODEL \
        --reranker none \
        --service_name ${service_name} \
        --dataset_name neuclir
done
