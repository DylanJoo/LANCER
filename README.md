<h1>LANCER: LLM Reranking for Nugget Coverage</h1>
---

### Data
We use the NeuCLIR 2024 report generation (neuclir) and CRUX Multi-Document Summarization (crux-mds) datasets for evaluation.
Each dataset includes: (1) topic/query (2) document coputs and (3) reference nuggets (i.e., multiple subquestions or question-answer pairs)

1. Topic/query:
- [neuclir](data/neuclir24-test-request.jsonl)
- [crux-mds](data/crux-mds-test-request.jsonl)

2. Document corpus:
* [neuclir1-mtdocs](https://huggingface.co/datasets/neuclir/neuclir1):
We use the translated English version of the documents. Since the data for has some parsing issue, we recommend to directly download the three datasets via wget. 
We fix the dataset and relase the [preprocessing code](data/preprocessing_codes).

- [crux-mds-corpus](https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus):
We combine the train and test corpus because our retrieved documents are from these two corpoara.

3. Reference nugget: 
We adopt the [crux-eval](https://github.io/DylanJoo/crux) evaluation toolkit to evaluate nugget coverage
- [neuclir (TBD)](data/neuclir24-test-nuggets.jsonl)
- [crux-mds (TBD)](data/crux-mds-test-nuggets.jsonl)

### Prerequisite
We use the crux evaluation toolkit to load the data and evaluate results. 
```
pip intsall git+https://github.com/DylanJoo/crux.git
pip install ir-measures[pyndeval]
pip install datasets
pip install ir_measures>=0.3.7
pip install pyserini
pip install transformers
pip install vllm
```

### Results
We provide the evaluation results of LANCER and several baselines on the two datasets.
Each dataset has 4 reported metrics: P/nDCG/A-nDCG/Cov. All metrics are truncated at rank 10.

#### The first-stage retrieved results
Please ensure crux evaluation is installed with the dataset downloaded. See the dataseet instruction in [crux](https://github.io/DylanJoo/crux).
```
bash evaluate_first_stage.sh
```
| Dataset             | Run File                          | Metric | Score   | Metric | Score   | Metric | Score       | Metric | Score  |
|---------------------|------------------------           |--------|---------|--------|---------|--------|-------------|--------|--------|
| neuclir-runs        | bm25-neuclir.run                  | P@10 | 0.6526 | nDCG@10 | 0.6767 | alpha_nDCG@10 | 0.5295 | Cov@10 | 0.6407 | 
| neuclir-runs        | lsr-milco-neuclir.run             | P@10 | 0.8158 | nDCG@10 | 0.8305 | alpha_nDCG@10 | 0.6294 | Cov@10 | 0.7367 | 
| neuclir-runs        | plaidx-neuclir.run                | P@10 | 0.5895 | nDCG@10 | 0.6126 | alpha_nDCG@10 | 0.4184 | Cov@10 | 0.4945 | 
| neuclir-runs        | qwen3-embed-8b-neuclir.run        | P@10 | 0.8684 | nDCG@10 | 0.8862 | alpha_nDCG@10 | 0.6269 | Cov@10 | 0.6948 | 
| crux-mds-duc04-runs | bm25-crux-mds-duc04.run           | P@10 | 0.5140 | nDCG@10 | 0.5298 | alpha_nDCG@10 | 0.4454 | Cov@10 | 0.5444 | 
| crux-mds-duc04-runs | lsr-crux-mds-duc04.run            | P@10 | 0.6800 | nDCG@10 | 0.7035 | alpha_nDCG@10 | 0.5579 | Cov@10 | 0.6241 | 
| crux-mds-duc04-runs | qwen3-embed-8b-crux-mds-duc04.run | P@10 | 0.7380 | nDCG@10 | 0.7586 | alpha_nDCG@10 | 0.6078 | Cov@10 | 0.6637 | 
