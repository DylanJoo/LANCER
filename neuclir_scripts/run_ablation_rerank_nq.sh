module load anaconda3/2024.2
conda activate ecir2026

MODEL=meta-llama/Llama-3.3-70B-Instruct

# Start experiments
echo "Start runnning"
ss_arg_list=(
    "bm25@localhost:5000"
    # "plaidx-neuclir@10.162.95.158:5000"
    # "qwen3-neuclir@10.162.95.158:5000"
    # "lsr-neuclir-mt@10.162.95.158:5000"
    # "lsr-neuclir@10.162.95.158:5000"
)

for ss_args in "${ss_arg_list[@]}"; do
    service_name=$(echo $ss_args | cut -d'@' -f1)
    service_endpoint=$(echo $ss_args | cut -d'@' -f2)

    echo "Starting evaluation for ${service_name} at ${service_endpoint}"

    for agg in sum greedy-sum-tau=3;do
        for n in 1 2 3 4 5;do
            python3 run_crux_reranker_eval.py \
                --n_subquestions $n \
                --agg $agg \
                --host http://${service_endpoint} \
                --llm_base_url http://localhost:8000/v1 \
                --temperature 0.0 --top_p 1.0 \
                --model $MODEL \
                --reranker crux_reranking \
                --service_name ${service_name} \
                --dataset_name neuclir
        done
    done
done
