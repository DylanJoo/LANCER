module load anaconda3/2024.2
conda activate ecir2026

MODEL=meta-llama/Llama-3.3-70B-Instruct

# Start experiments
# ss_arg_list=(
#     "bm25@localhost:5000"
#     "qwen3-neuclir@10.162.95.158:5000"
#     "lsr-neuclir@10.162.95.158:5000"
# )
# for ss_args in "${ss_arg_list[@]}"; do
#     service_name=$(echo $ss_args | cut -d'@' -f1)
#     service_endpoint=$(echo $ss_args | cut -d'@' -f2)
#
#     echo "Starting evaluation for ${service_name} at ${service_endpoint}"
#
#     python3 run_crux_reranker_eval.py \
#         --concat_original \
#         --n_subquestions 2 \
#         --agg sum \
#         --host http://${service_endpoint} \
#         --llm_base_url http://localhost:8000/v1 \
#         --temperature 0.0 --top_p 1.0 \
#         --model $MODEL \
#         --reranker crux_reranking \
#         --service_name ${service_name} \
#         --dataset_name neuclir
# done

ss_arg_list=(
    "bm25-mds-duc04@localhost:5000"
    "qwen3-mds-duc04@10.162.95.158:5000"
    "lsr-mds-duc04@10.162.95.158:5000"
)
for ss_args in "${ss_arg_list[@]}"; do
    service_name=$(echo $ss_args | cut -d'@' -f1)
    service_endpoint=$(echo $ss_args | cut -d'@' -f2)

    echo "Starting evaluation for ${service_name} at ${service_endpoint}"
    python3 run_crux_reranker_eval.py \
        --concat_original \
        --n_subquestions 3 \
        --agg greedy-sum-tau=3 \
        --host http://${service_endpoint} \
        --llm_base_url http://localhost:8000/v1 \
        --temperature 0.0 --top_p 1.0 \
        --model $MODEL \
        --reranker crux_reranking \
        --service_name ${service_name} \
        --dataset_name mds-duc04 \
        --topics_path '/exp/scale25/artifacts/crux/temp/neuclir_format/topic.mds_duc04.jsonl' \
        --corpus_dir '/exp/scale25/artifacts/crux/temp/passages/' \
        --nuggets_dir '/exp/scale25/' \
        --crux_qrels '/exp/scale25/artifacts/crux/temp/neuclir_format/qrels.mds_duc04.txt' \
        --crux_artifacts_path '/exp/scale25/artifacts/crux/temp/neuclir_format/ratings.mds_duc04.jsonl' 
done
