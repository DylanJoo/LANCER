<h1>LANCER: LLM Reranking for Nugget Coverage</h1>

### Prerequisite
We use the crux evaluation toolkit to load the data and evaluate results.
```
pip intsall git+https://github.com/DylanJoo/crux.git
pip install ir_measures>=0.3.7
pip install ir-measures[pyndeval]
pip install datasets
pip install pyserini
pip install transformers
pip install vllm
pip intsall git+https://github.com/DylanJoo/APRIL.git
```

### Data
We evalaute on the NeuCLIR 2024 report generation (neuclir) and CRUX Multi-Document Summarization (crux-mds) DUC04 datasets.
Each dataset includes:

1. Topic (query): 
- [neuclir](data/neuclir24-test-request.jsonl)
- [crux-mds](data/crux-mds-duc04.request.jsonl)

2. Document corpus:
* [neuclir1-mtdocs](https://huggingface.co/datasets/neuclir/neuclir1/tree/main/data):
For NeuCLIR, we use the translated English version of the documents. Note that the original data has some parsing issues, we recommend to directly download the datasets via wget. Then, you can fix the parsing error with the codes: [preprocessing code](data/preprocessing_codes).

- [crux-mds-corpus](https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus):
We combine the train and test subset into one.

### Evaluation scripts
We provide the evaluation results of LANCER and several baselines on the two datasets. Each dataset has 4 reported metrics: P/nDCG/A-nDCG/Cov. All metrics are truncated at rank 10.

We use the [crux-eval](https://github.io/DylanJoo/crux) to evaluate. 
Please refer to [crux-eval](https://github.com/DylanJoo/crux?tab=readme-ov-file#preparation) for more details. You will need `git clone` the entire Huggingface dataset repo. Then, you will need to setup the `CRUX_ROOT=/your_downloaded_dir/` before evaluation

For example, to evaluate the first-stage retrieved results, we use the script below (``bash eval_first_stage.sh``):
```
export CRUX_ROOT=/datasets/crux

for run_file in data/neuclir-runs/*.run;do
    python -m crux.evaluation.rac_eval \
        --run $run_file \
        --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
        --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl 
done

for run_file in data/crux-mds-duc04-runs/*.run;do
    python -m crux.evaluation.rac_eval \
        --run $run_file \
        --qrel $CRUX_ROOT/crux-mds-duc04/qrels/div_qrels-tau3.txt \
        --filter_by_oracle \
        --judge $CRUX_ROOT/crux-mds-duc04/judge 
done
```

#### The first-stage retrieval results
Please ensure crux evaluation is installed with the dataset downloaded. See the dataseet instruction in [crux](https://github.io/DylanJoo/crux).

| Dataset             | Run File                          | Metric | Score   | Metric | Score   | Metric | Score       | Metric | Score  |
|---------------------|------------------------           |--------|---------|--------|---------|--------|-------------|--------|--------|
| neuclir-runs        | bm25-neuclir.run                  | P@10 | 0.6526 | nDCG@10 | 0.6767 | alpha_nDCG@10 | 0.5295 | Cov@10 | 0.6407 | 
| neuclir-runs        | lsr-milco-neuclir.run             | P@10 | 0.8158 | nDCG@10 | 0.8305 | alpha_nDCG@10 | 0.6294 | Cov@10 | 0.7367 | 
| neuclir-runs        | plaidx-neuclir.run                | P@10 | 0.5895 | nDCG@10 | 0.6126 | alpha_nDCG@10 | 0.4184 | Cov@10 | 0.4945 | 
| neuclir-runs        | qwen3-embed-8b-neuclir.run        | P@10 | 0.8684 | nDCG@10 | 0.8862 | alpha_nDCG@10 | 0.6269 | Cov@10 | 0.6948 | 

| Dataset             | Run File                          | Metric | Score   | Metric | Score   | Metric | Score       | Metric | Score  |
|---------------------|------------------------           |--------|---------|--------|---------|--------|-------------|--------|--------|
| crux-mds-duc04-runs | bm25-crux-mds-duc04.run           | P@10 | 0.5140 | nDCG@10 | 0.5298 | alpha_nDCG@10 | 0.4454 | Cov@10 | 0.5444 | 
| crux-mds-duc04-runs | lsr-crux-mds-duc04.run            | P@10 | 0.6800 | nDCG@10 | 0.7035 | alpha_nDCG@10 | 0.5579 | Cov@10 | 0.6241 | 
| crux-mds-duc04-runs | qwen3-embed-8b-crux-mds-duc04.run | P@10 | 0.7380 | nDCG@10 | 0.7586 | alpha_nDCG@10 | 0.6078 | Cov@10 | 0.6637 | 

#### The LANCER results
We conduct the baseline reranking using [autorerank](https://github.io/APRIL). We compare with pointwise/setwise/pointwise.

| Dataset             | Run File                          | Metric | Score   | Metric | Score   | Metric | Score       | Metric | Score  |
|---------------------|------------------------           |--------|---------|--------|---------|--------|-------------|--------|--------|
| crux-mds-duc04-runs | bm25-autorerank:point:meta-llama.Llama-3.3-70B-Instruct.run           | P@10 | 0.7460 | nDCG@10 | 0.7578 | alpha_nDCG@10 | 0.5909 | Cov@10 | 0.6541 | 
| crux-mds-duc04-runs | bm25-lancer:agg_sum:nq_2.run                                          | P@10 | 0.7280 | nDCG@10 | 0.7388 | alpha_nDCG@10 | 0.6075 | Cov@10 | 0.6645 | 
| crux-mds-duc04-runs | bm25-lancer:oracle:agg_sum:nq_2.run                                   | P@10 | 0.7680 | nDCG@10 | 0.7997 | alpha_nDCG@10 | 0.7177 | Cov@10 | 0.7054 | 
| crux-mds-duc04-runs | lsr-autorerank:point:meta-llama.Llama-3.3-70B-Instruct.run            | P@10 | 0.8300 | nDCG@10 | 0.8218 | alpha_nDCG@10 | 0.6332 | Cov@10 | 0.6957 | 
| crux-mds-duc04-runs | lsr-lancer:agg_sum:nq_2.run                                           | P@10 | 0.7820 | nDCG@10 | 0.7906 | alpha_nDCG@10 | 0.6378 | Cov@10 | 0.6833 | 
| crux-mds-duc04-runs | qwen3-embed-8b-autorerank:point:meta-llama.Llama-3.3-70B-Instruct.run | P@10 | 0.8420 | nDCG@10 | 0.8295 | alpha_nDCG@10 | 0.6325 | Cov@10 | 0.6998 | 
| crux-mds-duc04-runs | qwen3-embed-8b-lancer:agg_sum:nq_2.run                                | P@10 | 0.7980 | nDCG@10 | 0.8050 | alpha_nDCG@10 | 0.6407 | Cov@10 | 0.6929 | 


#### Other articfacts for reproduction
You can reproduce LANCER results with  the subquestions we generated or the oracle sub-questions.

- Generated subquestions: 
[crux-mds-duc04-subquestions/qwen3-next-80b-a3b-instruct.json](results/crux-mds-duc04-subquestions/qwen3-next-80b-a3b-instruct.json), 
[neuclir-subquestions/llama3.3-70b-instruct.json](results/neuclir-subquestions/llama3.3-70b-instruct.json)
- Oracle subquestions: 
[crux-mds-duc04-subquestions/subquestions.oracle.jsonl](results/crux-mds-duc04-subquestions/subquestions.oracle.jsonl), 
[neuclir-subquestions/subquestions.oracle.jsonl](results/neuclir-subquestions/subquestions.oracle.jsonl)

- Reranked results: See all of them in [results/crux-mds-duc04-runs](results/crux-mds-duc04-runs) and [results/neuclir-runs](results/neuclir-runs).
