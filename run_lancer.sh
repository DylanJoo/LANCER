# python src/lancer/run_cruxmds.py \
#     --reranker lancer \
#     --run_path data/crux-mds-duc04-runs/bm25-crux-mds-duc04.run \
#     --topic_path data/crux-mds-duc04.request.jsonl \
#     --n_subquestions 2 \
#     --agg_method sum 

python src/lancer/run_neuclir.py \
    --reranker lancer \
    --run_path data/neuclir-runs/bm25-neuclir.run \
    --topic_path data/neuclir24-test-request.jsonl \
    --n_subquestions 2 \
    --agg_method sum 
