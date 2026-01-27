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

### Results
We provide the evaluation results of LANCER and several baselines on the two datasets.
Each dataset has 4 reported metrics: P/nDCG/A-nDCG/Cov. All metrics are truncated at rank 10.

| Reranking     |  NeuCLIR 2024 Report Generation | CRUX Multi-Document Summarization |
|-------------- |:-------------------------------:|:---------------------------------:|
| BM25 + LANCER | 0.352 / 0.366 / 0.298 / 0.421   | 0.423 / 0.441 / 0.367 / 0.512     |




### Prerequisite
We use the crux evaluation toolkit to load the data and evaluate results. 
```
pip intsall git+https://github.com/DylanJoo/crux.git
pip install datasets
pip install ir_measures>=0.3.7
pip install pyserini
pip install transformers
pip install vllm
```
