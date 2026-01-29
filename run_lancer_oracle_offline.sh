module load anaconda3/2024.2
conda activate ecir2026

for retrieval in bm25 lsr qwen3-embed-8b; do
    python src/lancer/run_cruxmds.py \
        --reranker lancer \
        --run_path data/crux-mds-duc04-runs/${retrieval}-crux-mds-duc04.run \
        --topic_path data/crux-mds-duc04.request.jsonl \
        --use_oracle \
        --qg_path results/crux-mds-duc04-subquestions/subquestions.oracle.jsonl \
        --judge_path results/crux-mds-duc04-ratings/ratings.oracle.${retrieval}.json \
        --agg_method sum 
done

for retrieval in bm25 lsr-milco qwen3-embed-8b; do
    python src/lancer/run_neuclir.py \
        --reranker lancer \
        --run_path data/neuclir-runs/${retrieval}-neuclir.run \
        --topic_path data/neuclir24-test-request.jsonl \
        --use_oracle \
        --qg_path results/neuclir-subquestions/subquestions.oracle.jsonl \
        --judge_path results/neuclir-ratings/ratings.oracle.${retrieval}.json \
        --agg_method sum 
done
