import ir_measures
import asyncio
import requests
from argparse import Namespace
import re
import os
import glob
from collections import defaultdict, OrderedDict
import json
import pickle

from ir_measures import Qrel
from tqdm import tqdm
import pandas as pd

path = "/exp/scale25/neuclir/eval/nuggets/nuggets_388.json"

from tools.neuclir.create_run_file import create_run_file
from tools.neuclir.subtopics_generation import async_generate_subtopics, async_rank_subtopics
from tools.neuclir.subtopics_generation import async_generate_persona_criterias_mcranker, generate_subtopics, generate_complementary_subtopics


def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx : min(ndx + size, l)]


def load_qrels(path, threshold=1):
    data = defaultdict(dict)
    with open(path) as f:
        for line in f:
            qid, facet, docid, score = line.strip().split()
            if int(score) >= threshold:
                data[str(qid).strip()].update({docid: int(score)})
    return data


def load_diversity_qrels(path: str) -> list:
    qrels = pd.read_csv(path, sep="\s+", names=["query_id", "iteration", "doc_id", "relevance"])
    return qrels
    #
    # diversity_qrels = [Qrel(str(row.query_id), row.doc_id, row.relevance, row.iteration) for row in qrels.itertuples(index=False)]
    #
    # return diversity_qrels
    # return ir_measures.read_trec_qrels(path)


def load_query(args: Namespace):
    """Returns:
    queries [dict]: queries used as part of CRUX
    queries_for_search [dict]: queries used when performing search
                               (i.e. we might not want to include the background)
    raw_topics [dict]: raw topics as provided in topic file (i.e. example-request.jsonl)
    """
    path = args.topics_path
    queries = {}
    queries_for_search = {}
    raw_topics = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            queries[data["request_id"]] = data["background"].strip() + " " + data["problem_statement"].strip()
            if args.search_with_subqueries:
                queries_for_search[data["request_id"]] = data["background"].strip() + " " + data["problem_statement"].strip()
            else:
                queries_for_search[data["request_id"]] = data["problem_statement"].strip()  # don't add background
            raw_topics.append(data)
    return queries, queries_for_search, raw_topics


# dylan: try to separate the loading function for NeuCLIR-IR and NeuCLIR-RAG tasks
def load_topic(path, use_translation=False):
    """title + description as query for search"""
    topics = {}
    if path.endswith("tsv"):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                qid, qtext = line.split("\t")
                topics[str(qid.strip())] = qtext.strip()

                if (i + 1) == debug:
                    break
    if path.endswith("jsonl"):
        with open(path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())

                title = data["topics"][0]["topic_title"]
                desc = data["topics"][0]["topic_description"]
                topics[str(data["topic_id"]).strip()] = title + " " + desc
    return topics


def prepreocess(texts):
    pattern = re.compile(r"^(\d+)*\.")
    texts = re.sub(r"\<q\>|\<\/q\>", "\n", texts)
    texts = re.sub(pattern, "\n", texts)
    pattern = re.compile(r"^(\d+)*\.")
    texts = re.sub(pattern, "", texts)
    return texts

def load_subtopics_human(path,
                         args, 
                         raw_topics=None,
                         create_new_subtopics=False):
    # [TODO] AND/OR in the pipeline
    files = [f for f in glob.glob(f"{path}/nuggets_*json")]
    subquestions = {}
    for file in files:
        match = re.search(r"nuggets_(\d+)\.json$", file)
        qid = str(match.group(1))
        data = json.load(open(file, "r"))
        subquestions[qid] = list(data.keys())
        # subanswer[id] = [subanswer for subanswer in subquestions[id]]

    if create_new_subtopics:
        extra_subtopics = generate_complementary_subtopics(args, raw_topics, subquestions)
        for qid in extra_subtopics:
            subquestions[qid].extend(extra_subtopics[qid])

    return subquestions


def load_subtopics(args, topic):
    # [TODO] replace with code from the auto-nuggetization team
    subtopics = generate_subtopics(args, topic)
    return subtopics

async def async_load_subtopics(args, topic, docs_to_rerank=None,
                               reranking=False, is_claims=False,
                               higher_cov_ratings=False):
    
    if args.reranker == "crux_reranking":
        # [TODO] replace with code from the auto-nuggetization team
        subtopics = await async_generate_subtopics(args, topic, docs_to_rerank=docs_to_rerank,
                                                reranking=reranking, is_claims=is_claims,
                                                higher_cov_ratings=higher_cov_ratings)
    elif args.reranker == "mcranker": # mcranker
        subtopics = await async_generate_persona_criterias_mcranker(
            args, topic, docs_to_rerank=docs_to_rerank,
            reranking=reranking, is_claims=is_claims,
            higher_cov_ratings=higher_cov_ratings)
    
    if args.rerank_crux_subquestions:
        subtopics = await async_rank_subtopics(args, subtopics, topic,
                                               docs_to_rerank=docs_to_rerank)
        
    return subtopics


def load_corpus_online(path=None):
    class get_content:
        def __len__(self):
            return 1

        def __getitem__(self, doc_id):
            doc = requests.post(url="http://10.162.95.158:5000", json={"collection": "neuclir", "id": doc_id}).json()
            text = doc.get("text", "").replace("\n", " ").strip()
            title = doc.get("title", "").replace("\n", " ").strip()
            return {"text": text, "title": title}

    return get_content()


def load_nuggets(path, include_answer=False):
    if os.path.isdir(path):
        files = [f for f in glob.glob(f"{path}/*")]
    else:
        files = [path]

    nuggets = {}
    for file in files:
        match = re.search(r"nuggets_(\d+)\.json$", file)
        if match:
            qid = str(match.group(1))
            nuggets[qid] = json.load(open(file, "r"))
    return nuggets


def load_runs_or_qrels(path, topic,
                       topk=100, threshold=3, qrel_missing=None, docs_to_rerank=None):
    if qrel_missing:
        return qrel_missing
    if docs_to_rerank:
        return {topic["request_id"]: docs_to_rerank}
    run_dict = defaultdict(list)
    with open(path, "r") as f:
        try:
            run_dict = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for i, line in enumerate(f):
                try:
                    qid, _, docid, rank, score, _ = line.strip().split()
                except:
                    rank = 0  # consider all for qrels
                    qid, _, docid, score = line.strip().split()

                if (int(rank) <= topk) and (float(score) >= threshold):
                    run_dict[qid] += [docid]
    return run_dict


def load_runs(path, topk=None, output_score=False):  # support .trec file only
    run_dict = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            if int(rank) <= (9999 or topk):
                run_dict[str(qid)] += [(docid, float(rank), float(score))]

    # sort by score and return static dictionary
    sorted_run_dict = OrderedDict()
    for qid, docid_ranks in run_dict.items():
        sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False)
        if output_score:
            # sorted_run_dict[qid] = [{docid, rel_score} for docid, rel_rank, rel_score in sorted_docid_ranks]
            sorted_run_dict[qid] = {docid: rel_score for docid, rel_rank, rel_score in sorted_docid_ranks}
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]

    return sorted_run_dict


def get_judgements_path(args):
    """judgements are computed by running augmentation/gen_ratings.py"""
    judgements_dir = os.path.join(args.crux_dir, args.dataset_name, args.tag)
    path = os.path.join(judgements_dir, f"crux_{args.model}.jsonl")
    return path


def sort_and_truncate(run, max_k_dict=None):
    truncated_run = {}
    for qid, docid_scores in run.items():
        topk = max_k_dict[qid]
        sorted_docs = dict(sorted(docid_scores.items(), key=lambda x: x[1], reverse=True)[:topk])
        truncated_run[qid] = sorted_docs
    return truncated_run


def binarize(qrels):
    binarized_qrels = {}
    for qid, docid_scores in qrels.items():
        docid_scores = {docid: 1 for docid, score in docid_scores.items()}
        binarized_qrels[qid] = docid_scores
    return binarized_qrels
