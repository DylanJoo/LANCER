<h1>LANCER: LLM Reranking for Nugget Coverage</h1>


### Prerequisite
We use the crux evaluation toolkit to load the data and evaluate results.
```
pip install git+https://github.com/DylanJoo/crux.git
pip install ir_measures>=0.3.7
pip install ir-measures[pyndeval]
pip install datasets
pip install pyserini
pip install transformers
pip install vllm
pip install git+https://github.com/DylanJoo/APRIL.git
```
Installing these libraries should work. We also release the [requirements.yml](requirements.yml) just in case we miss some trivial dependencies.

### Data
We evaluate on the NeuCLIR 2024 report generation (neuclir) and CRUX Multi-Document Summarization (crux-mds) DUC04 datasets.

Each dataset includes:

1. Topic (query): 
- [neuclir](data/neuclir24-test-request.jsonl)
- [crux-mds](data/crux-mds-duc04.request.jsonl)

2. Document corpus:
* [neuclir1-mtdocs](https://huggingface.co/datasets/neuclir/neuclir1/tree/main/data):
For NeuCLIR, we use the translated English version of the documents. Note that the original data has some parsing issues, we recommend to directly download the datasets via wget. Then, you can fix the parsing errors if needed.

* [crux-mds-corpus](https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus):
We combine the train and test subset into one.

### Evaluation scripts
We provide the evaluation results of LANCER and several baselines on the two datasets. Each dataset has 4 reported metrics: P/nDCG/A-nDCG/Cov. All metrics are truncated at rank 10.

We use the [crux-eval](https://github.com/DylanJoo/crux) to evaluate. 
Please refer to [crux-eval](https://github.com/DylanJoo/crux?tab=readme-ov-file#preparation) for more details. You will need `git clone` the entire Huggingface dataset repo. Then, you will need to set it up as described.

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
Installing these libraries should work. We also release the [requirements.yml](requirements.yml) just in case we miss some trivial dependencies.

### Results
Please ensure crux evaluation is installed with the dataset downloaded. See the dataset instruction in [crux-eval](https://github.com/DylanJoo/crux).

| Dataset             | Run File                          | P@10   | nDCG@10 | alpha_nDCG@10 | Cov@10 |
|---------------------|-----------------------------------|--------|---------|---------------|--------|
| neuclir-runs        | bm25-neuclir.run                  | 0.6526 | 0.6767  | 0.5295        | 0.6407 |
| neuclir-runs        | lsr-milco-neuclir.run             | 0.8158 | 0.8305  | 0.6294        | 0.7367 |
| neuclir-runs        | plaidx-neuclir.run                | 0.5895 | 0.6126  | 0.4184        | 0.4945 |
| neuclir-runs        | qwen3-embed-8b-neuclir.run        | 0.8684 | 0.8862  | 0.6269        | 0.6948 | 

| Dataset             | Run File                          | P@10   | nDCG@10 | alpha_nDCG@10 | Cov@10 |
|---------------------|-----------------------------------|--------|---------|---------------|--------|
| crux-mds-duc04-runs | bm25-crux-mds-duc04.run           | 0.5140 | 0.5298  | 0.4454        | 0.5444 |
| crux-mds-duc04-runs | lsr-crux-mds-duc04.run            | 0.6800 | 0.7035  | 0.5579        | 0.6241 |
| crux-mds-duc04-runs | qwen3-embed-8b-crux-mds-duc04.run | 0.7380 | 0.7586  | 0.6078        | 0.6637 | 

#### Other artifacts for reproduction
You can reproduce LANCER results with the subquestions we generated or the oracle sub-questions.

- Generated subquestions: 
[crux-mds-duc04-subquestions/qwen3-next-80b-a3b-instruct.json](results/crux-mds-duc04-subquestions/qwen3-next-80b-instruct.json), 
[neuclir-subquestions/llama3.3-70b-instruct.json](results/neuclir-subquestions/llama3.3-70b-instruct.json)
- Oracle subquestions: 
[crux-mds-duc04-subquestions/subquestions.oracle.jsonl](results/crux-mds-duc04-subquestions/subquestions.oracle.jsonl), 
[neuclir-subquestions/subquestions.oracle.jsonl](results/neuclir-subquestions/subquestions.oracle.jsonl)
- Reranked results: 
See all of them in [results/crux-mds-duc04-runs](results/crux-mds-duc04-runs) and [results/neuclir-runs](results/neuclir-runs).

### Citation
This paper has been accepted at the European Conference on Information Retrieval (ECIR) 2026. If you use or build upon our work, please cite us as follows:
```
@inproceedings{Ju2026LANCER,
  title={LANCER: LLM Reranking for Nugget Coverage},
  author={Jia-Huei Ju and François G. Landry and Eugene Yang and Suzan Verberne and Andrew Yates},
  booktitle={ European Conference on Information Retrieval},
  year={2026}
}
```
