<h1>LANCER: LLM Reranking for Nugget Coverage</h1>
---

### Data
We use the NeuCLIR 2024 report generation (neuclir) and CRUX Multi-Document Summarization (crux-mds) datasets for evaluation.
Each dataset includes: (1) topic/query (2) document coputs and (3) reference nuggets (i.e., multiple subquestions or question-answer pairs)

1. Topic/query:
- [neuclir](data/neuclir24-test-request.jsonl)
- [crux-mds](data/crux-mds-test-request.jsonl)

2. Document corpus:
- [neuclir](https://huggingface.co/datasets/neuclir/neuclir1) (PS: we use the translated English version of the documents)
- [crux-mds](https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus) (PS: we combine the train and test corpus)

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
| Dataset                | Run | P@10 | nDCG@10 | alpha_nDCG@10 | Cov@10 |
data/neuclir-runs        | bm25-neuclir.run | P@10 | 0.6526 | nDCG@10 | 0.6767 | alpha_nDCG@10 | 0.5295 | Cov@10 | 0.6407 | 
data/neuclir-runs        | lsr-milco-neuclir.run | P@10 | 0.8158 | nDCG@10 | 0.8305 | alpha_nDCG@10 | 0.6294 | Cov@10 | 0.7367 | 
data/neuclir-runs        | plaidx-neuclir.run | P@10 | 0.5895 | nDCG@10 | 0.6126 | alpha_nDCG@10 | 0.4184 | Cov@10 | 0.4945 | 
data/neuclir-runs        | qwen3-embed-8b-neuclir.run | P@10 | 0.8684 | nDCG@10 | 0.8862 | alpha_nDCG@10 | 0.6269 | Cov@10 | 0.6948 | 
data/crux-mds-duc04-runs | bm25-crux-mds-duc04.run | P@10 | 0.5140 | nDCG@10 | 0.5298 | alpha_nDCG@10 | 0.4454 | Cov@10 | 0.5444 | 
data/crux-mds-duc04-runs | lsr-crux-mds-duc04.run | P@10 | 0.6800 | nDCG@10 | 0.7035 | alpha_nDCG@10 | 0.5579 | Cov@10 | 0.6241 | 
data/crux-mds-duc04-runs | qwen3-embed-8b-crux-mds-duc04.run | P@10 | 0.7380 | nDCG@10 | 0.7586 | alpha_nDCG@10 | 0.6078 | Cov@10 | 0.6637 | 
