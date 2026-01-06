### Prerequisite

You can install the required packages with:
```
pip install -r requirements.txt
```

# Evaluating retrieval - Report Generation

IR evaluation metrics can be computed by running the following:

```
python3 run_rac_nugget_eval.py \
--host http://127.0.0.1 \
--port 443 \
--service_name plaidx-neuclir \
--dataset_name neuclir
```

This will compute evaluation metrics using the NeuCLIR reference nuggets test set (neuclir24-test-request.jsonl). The script expects that a search service is running on the specified host and port, and that the service can accept requests in the same format as the PLAID-X search service.

# Evaluating retrieval - NeuCLIR topics retrieval
Setting the initial limit as 50 (retry with increased limit till maximum 500).
```bash
#  DONT use public endpoint: --search_endpoint https://scale25.hltcoe.org
python neuclir_ir_eval.py \
    --service_endpoint 10.162.95.158:5000 \
    --service_name plaidx-neuclir \
    --limit 1000 \
    --output_dir results.plaidx

# results.plaidx/ir_result.json
{
    "2022": {
        "zho": 0.49819554036552616,
        "rus": 0.475934498970835,
        "fas": 0.4442176517213668
    },
    "2023": {
        "zho": 0.45467164407992805,
        "rus": 0.5050548618457017,
        "fas": 0.5051494408547021,
        "mlir": 0.40227682169926543
    },
    "2024": {
        "zho": 0.5291277922852041,
        "rus": 0.4893684473514584,
        "fas": 0.5831951736456751,
        "mlir": 0.46445297716404205
    }
}
```

# CRUX reranking - Evaluation on NeuCLIR Data (Report Generation)

To evaluate CRUX reranking on NeuCLIR data (report generation IR metrics), run the following (after changing parameters to point to the search service):

```
python3 run_crux_reranker_eval.py \
--host http://127.0.0.1 \
--port 443 \
--service_name plaidx-neuclir \
--dataset_name neuclir
```

# CRUX

### Prepare source multi-document summarization dataset
- NeuCLIR: nothing to do, files will be read from `/exp/scale25/neuclir`.
- researchy_questions:
Download the dataset to a local path (as this is a big dataset, you likely want to download it somewhere on `/exp/<user-id>`). You can download the dataset from huggingface by running: `python3 -m data.download_researchy_dataset` and setting the download path at the top of the file.

### Answerability

Here is an example command to obtain sub-question scores:

```
python3 -m augmentation.gen_ratings \
    --shard_dir <crux_dir> --shard_size 1000 \
    --config configs/scale.llama3-70b.yaml \
    --split test \
    --model llama3.3-70b-instruct \
    --model_tag metallama3.3-70b \
	--port 4000 \
    --tag ratings-gen/70b \
    --load_mode api \
	--temperature 0 --top_p 1 \
    --max_new_tokens 3 \
	--output_dir <crux_dir> \
    --ampere_gpu
```

# Other Instructions

### Prepare source multi-document summarization dataset
- Multi-News and DUC'04: 
We also upload these two raw datasets in our huggingface dataset repo. See the original dataset and their terms of use: [Multi-News](https://huggingface.co/datasets/alexfabbri/multi_news) and [DUC'04](https://www-nlpir.nist.gov/projects/duc/data.html).

### Evaluation data generation
We have already shared all the dataset (raw, intermediate and final data) in [Huggingface repo](#). But you can also follow the steps to generate and customize. 

## Passages, topics, questions 
Here is an example of generating passages. And you will find the generated data in 
`<path-to-crux>/shard_data/psg-gen/metallama3.1-8b-{train/test/testb}-${shard_i}.json`
    * Passages (mulit-news train and test)
	* Topics and Questions (multi-news test and DUC'04)
	* To run them in parallel, you can set the `shard` parameter and assign the `shard size (size of one shard)`

```
python3 -m augmentation.gen_passages  \
	--shard ${shard_i} --shard_size 1000 \
	--multi_news_file <path-to-multi_news> \
	--config configs/mds-decontextualize.llama3-8b.yaml \
	--split train \
	--model meta-llama/Meta-Llama-3.1-8B-Instruct \
	--model_tag metallama3.1-8b \
	--tag psgs-gen \
	--load_mode vllm \
	--temperature 0.7 \
	--max_new_tokens 640 \
	--output_dir <path-to-crux>/shard_data/
	--ampere_gpu
done
```
The topic/question generation can be found in `script/`

### Answerability

```
for split in train test;do
python3 -m augmentation.gen_ratings \
	--shard_dir <path-to-crux>/shard_data --shard_size 1000 \
	--config configs/mds-decontextualize.llama3-8b.yaml \
	--split ${split} \
	--model meta-llama/Meta-Llama-3.1-8B-Instruct \
	--model_tag metallama3.1-8b \
	--tag ratings-gen \
	--load_mode vllm \
	--temperature 0 --top_p 1 \
	--max_new_tokens 3 \
	--output_dir <path-to-crux>/shard_data \
	--ampere_gpu
done

python3 -m augmentation.gen_ratings \
    --shard_dir <path-to-crux>/shard_data --shard_size 1000 \
    --config configs/mds-decontextualize.llama3-8b.yaml \
    --split testb \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_tag metallama3.1-8b \
    --tag ratings-gen/8b \
    --load_mode vllm \
	--temperature 0 --top_p 1 \
    --max_new_tokens 3 \
	--output_dir <path-to-crux>/shard_data \
    --ampere_gpu
```

### Data example
```
# Topic:
# Questions:
# Passages (oracle) and ratings:
```

- Compile them into the crux data scheme (default answerability tau is 3)
Once we had the ratings, we can create the subsets of `required`, `partially redundant` and `fully redundant`.

```
bash create_context_ranking_data.sh
```

- Get documents and passages
```
bash create_dataset.sh
```

- Data statisitcs
```
## context
python -m tools.get_stats --dataset_dir /home/dju/datasets/crux --split testb

## ranking qrels
cat ${DATASET_DIR}/crux/ranking_3/test_qrels_oracle_context_pr.txt  | cut -f 4 -d ' ' | sort | uniq -c 
```

### Baseline settings
* The first-stage retrieval: BM25, contriever-MS, SPLADE-MS
```
Indexing and Retrieve top-100
```

* The former-stage augmentation: vanilla, BART-summarization, ReComp summarization

### Main findings
* Table2: The oracle retrieval context for baseline retrievla-augmentation
