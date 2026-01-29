# LANCER: LLM Reranking for Nugget Coverage

[![ECIR 2026](https://img.shields.io/badge/ECIR-2026-blue.svg)](https://ecir2026.org)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**LANCER** is a novel LLM-based reranking framework that optimizes document ranking for nugget coverage in retrieval-augmented generation tasks.

---
## Overview

LANCER leverages LLMs to rerank documents by decomposing complex queries into sub-questions and evaluating document relevance based on nugget coverage.
---

## Installation

### Prerequisites

Install the required dependencies using pip:

```bash
# Core dependencies
pip install ir_measures>=0.3.7
pip install ir-measures[pyndeval]
pip install datasets
pip install pyserini
pip install transformers
pip install vllm
```

### Prerequisites

We use the crux framework for evaluation. APRIL is used for implementing baseline LLM reranking methods.

```bash
pip install git+https://github.com/DylanJoo/crux.git
pip install git+https://github.com/DylanJoo/APRIL.git
```

> **Note:** For a complete environment setup, see [requirements.yml](requirements.yml).

---

## Datasets

We evaluate LANCER on two datasets:

| Dataset | Description | Topics |
|---------|-------------|--------|
| **NeuCLIR 2024** | Report generation task | [neuclir24-test-request.jsonl](data/neuclir24-test-request.jsonl) |
| **CRUX-MDS DUC'04** | Multi-document summarization | [crux-mds-duc04.request.jsonl](data/crux-mds-duc04.request.jsonl) |

### Document Corpora

- **NeuCLIR**: [neuclir1-mtdocs](https://huggingface.co/datasets/neuclir/neuclir1/tree/main/data)
  - Uses translated English documents
  - We recommend downloading directly via `wget` to avoid parsing issues
  
- **CRUX-MDS**: [crux-mds-corpus](https://huggingface.co/datasets/DylanJHJ/crux-mds-corpus)
  - Combined train and test subsets

---

## Usage

### 1. Launch vLLM Service (Optional for 4)

Start a vLLM server to serve the LLM for sub-question generation and relevance judgment:

```bash
MODEL=meta-llama/Llama-3.3-70B-Instruct

NCCL_P2P_DISABLE=1 VLLM_SKIP_P2P_CHECK=1 vllm serve $MODEL \
    --max-model-len 8192 \
    --port 8000 \
    --dtype bfloat16 \
    --disable-custom-all-reduce \
    --tensor-parallel-size 2 > vllm_server.log 2>&1 &
PID=$!

# Wait until server is ready
echo "Waiting for vLLM server (PID=$PID) to start..."
until curl -s http://localhost:8000/v1/models >/dev/null; do
    echo "vLLM server not yet available, retrying in 10 seconds..."
    sleep 10
done
echo "vLLM server is up and running on port 8000."
```

### 2. Run LANCER with LLM-generated Sub-questions

Run LANCER using different first-stage retrieval methods. Pre-computed first-stage results are available in [`data/`](data/), and reranked results can be found in [`results/`](results/).

#### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--rerun_qg` | Re-run the LLM sub-question generation stage (generates new sub-questions instead of loading from `--qg_path`) |
| `--qg_path` | Path to pre-generated sub-questions file (used when `--rerun_qg` is not set) |
| `--n_subquestions` | Number of sub-questions to use |
| `--rerun_judge` | Re-run the LLM relevance judgment stage |
| `--agg_method` | Aggregation method for combining scores (e.g., `sum`) |

> **Note:** The examples below use pre-generated sub-questions via `--qg_path`. To run the sub-question generation step with the LLM instead, add `--rerun_qg` to the command. When `--rerun_qg` is specified, the generated sub-questions will be saved to the path specified by `--qg_path`.

#### CRUX-MDS (DUC'04)

```bash
for retrieval in bm25 lsr qwen3-embed-8b; do
    python src/run_cruxmds.py \
        --reranker lancer \
        --run_path data/crux-mds-duc04-runs/${retrieval}-crux-mds-duc04.run \
        --topic_path data/crux-mds-duc04.request.jsonl \
        --qg_path results/crux-mds-duc04-subquestions/qwen3-next-80b-a3b-instruct.json \
        --rerun_judge \
        --n_subquestions 2 \
        --agg_method sum
done
```

#### NeuCLIR'24

```bash
for retrieval in bm25 lsr-milco qwen3-embed-8b; do
    python src/run_neuclir.py \
        --reranker lancer \
        --run_path data/neuclir-runs/${retrieval}-neuclir.run \
        --topic_path data/neuclir24-test-request.jsonl \
        --qg_path results/neuclir-subquestions/llama3.3-70b-instruct.json \
        --rerun_judge \
        --n_subquestions 2 \
        --agg_method sum
done
```

### 3. Run LANCER Oracle (with Ground-truth Sub-questions)

LANCER Oracle uses ground-truth sub-questions instead of LLM-generated ones, providing an upper-bound performance useful for ablation studies.

#### Additional Parameters

| Parameter | Description |
|-----------|-------------|
| `--use_oracle` | Enable oracle mode |
| `--qg_path` | Points to oracle sub-questions file |

#### CRUX-MDS (DUC'04)

```bash
for retrieval in bm25 lsr qwen3-embed-8b; do
    python src/run_cruxmds.py \
        --reranker lancer \
        --run_path data/crux-mds-duc04-runs/${retrieval}-crux-mds-duc04.run \
        --topic_path data/crux-mds-duc04.request.jsonl \
        --use_oracle \
        --qg_path results/crux-mds-duc04-subquestions/subquestions.oracle.json \
        --rerun_judge \
        --agg_method sum
done
```

#### NeuCLIR'24

```bash
for retrieval in bm25 lsr-milco qwen3-embed-8b; do
    python src/run_neuclir.py \
        --reranker lancer \
        --run_path data/neuclir-runs/${retrieval}-neuclir.run \
        --topic_path data/neuclir24-test-request.jsonl \
        --use_oracle \
        --qg_path results/neuclir-subquestions/subquestions.oracle.json \
        --rerun_judge \
        --agg_method sum
done
```

### 4. Run LANCER Oracle (Offline Mode)

For faster experimentation, use pre-computed oracle ratings to skip the LLM judgment stage entirely.

#### Additional Parameter

| Parameter | Description |
|-----------|-------------|
| `--judge_path` | Path to pre-computed relevance ratings file |

#### CRUX-MDS (DUC'04)

```bash
for retrieval in bm25 lsr qwen3-embed-8b; do
    python src/run_cruxmds.py \
        --reranker lancer \
        --run_path data/crux-mds-duc04-runs/${retrieval}-crux-mds-duc04.run \
        --topic_path data/crux-mds-duc04.request.jsonl \
        --use_oracle \
        --qg_path results/crux-mds-duc04-subquestions/subquestions.oracle.json \
        --judge_path results/crux-mds-duc04-ratings/ratings.oracle.${retrieval}.json \
        --agg_method sum
done
```

#### NeuCLIR'24

```bash
for retrieval in lsr-milco qwen3-embed-8b; do
    python src/run_neuclir.py \
        --reranker lancer \
        --run_path data/neuclir-runs/${retrieval}-neuclir.run \
        --topic_path data/neuclir24-test-request.jsonl \
        --use_oracle \
        --qg_path results/neuclir-subquestions/subquestions.oracle.json \
        --judge_path results/neuclir-ratings/ratings.oracle.${retrieval}.json \
        --agg_method sum
done
```

### 5. Run Baseline LLM Reranking Methods

We provide baseline LLM reranking methods using the [APRIL](https://github.com/DylanJoo/APRIL.git) framework. The supported methods include pointwise, listwise (RankGPT), and setwise reranking. Below is an example using pointwise reranking.

#### Reranker Parameter Format

| Format | Description |
|--------|-------------|
| `autorerank:point:MODEL` | Pointwise reranking |
| `autorerank:rankgpt:MODEL` | Listwise reranking (RankGPT) |
| `autorerank:setwise:MODEL` | Setwise reranking |

#### CRUX-MDS (DUC'04)

```bash
for retrieval in bm25 lsr qwen3-embed-8b; do
    python src/run_cruxmds.py \
        --reranker autorerank:point:meta-llama/Llama-3.3-70B-Instruct \
        --run_path data/crux-mds-duc04-runs/${retrieval}-crux-mds-duc04.run \
        --topic_path data/crux-mds-duc04.request.jsonl
done
```

#### NeuCLIR'24

```bash
for retrieval in bm25 lsr-milco qwen3-embed-8b; do
    python src/run_neuclir.py \
        --reranker autorerank:point:meta-llama/Llama-3.3-70B-Instruct \
        --run_path data/neuclir-runs/${retrieval}-neuclir.run \
        --topic_path data/neuclir24-test-request.jsonl
done
```

---

## Evaluation

We evaluate using four metrics (all truncated at rank 10):
- **P@10**: Precision at 10
- **nDCG@10**: Normalized Discounted Cumulative Gain
- **α-nDCG@10**: Alpha-normalized DCG (diversity-aware)
- **Cov@10**: Nugget Coverage

### Setup

1. Install [crux-eval](https://github.com/DylanJoo/crux)
2. Clone the Huggingface dataset repository (see [preparation guide](https://github.com/DylanJoo/crux?tab=readme-ov-file#preparation))
3. Set the `CRUX_ROOT` environment variable

### Evaluate Retrieval/Reranked Results

```bash
export CRUX_ROOT=/datasets/crux

# NeuCLIR evaluation
for run_file in data/neuclir-runs/*.run; do
    python -m crux.evaluation.rac_eval \
        --run $run_file \
        --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
        --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl
done

# CRUX-MDS evaluation
for run_file in data/crux-mds-duc04-runs/*.run; do
    python -m crux.evaluation.rac_eval \
        --run $run_file \
        --qrel $CRUX_ROOT/crux-mds-duc04/qrels/div_qrels-tau3.txt \
        --filter_by_oracle \
        --judge $CRUX_ROOT/crux-mds-duc04/judge
done
```

---

## Reproducibility Artifacts

We provide all artifacts needed to reproduce LANCER results:

### Sub-questions

| Type | CRUX-MDS DUC'04 | NeuCLIR |
|------|-----------------|---------|
| **Generated** | [qwen3-next-80b-instruct.json](results/crux-mds-duc04-subquestions/qwen3-next-80b-instruct.json) | [llama3.3-70b-instruct.json](results/neuclir-subquestions/llama3.3-70b-instruct.json) |
| **Oracle** | [subquestions.oracle.json](results/crux-mds-duc04-subquestions/subquestions.oracle.json) | [subquestions.oracle.json](results/neuclir-subquestions/subquestions.oracle.json) |

### Reranked Results

- [CRUX-MDS DUC'04 runs](results/crux-mds-duc04-runs)
- [NeuCLIR runs](results/neuclir-runs)

---

## Citation

This paper has been accepted at the **European Conference on Information Retrieval (ECIR) 2026**. If you use or build upon our work, please cite us:

```bibtex
@inproceedings{Ju2026LANCER,
  title     = {LANCER: LLM Reranking for Nugget Coverage},
  author    = {Jia-Huei Ju and François G. Landry and Eugene Yang and Suzan Verberne and Andrew Yates},
  booktitle = {European Conference on Information Retrieval},
  year      = {2026}
}
```
