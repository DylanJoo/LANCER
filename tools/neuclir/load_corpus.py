import os
import glob
from collections import defaultdict
import json
import pickle

from tqdm import tqdm

class Singleton(type):
    """ Python implementation for a Singleton """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class Corpus(metaclass=Singleton):
    def __init__(self):
        pass

    def set_corpus(self, corpus):
        self.corpus = corpus

    def get_corpus(self):
        if not hasattr(self, 'corpus'):
            return None
        return self.corpus
    
class ClaimsCorpus(metaclass=Singleton):
    def __init__(self):
        pass

    def set_corpus(self, corpus):
        self.corpus = corpus

    def get_corpus(self):
        if not hasattr(self, 'corpus'):
            return None
        return self.corpus

def default_doc():
    return {"title": "Unknown", "text": "Document not found."}

def load_corpus(args, path, path_prefix="*", load_from_pkl_file=True, is_claims=False):

    filename = "corpus.pkl" if not is_claims else "corpus_claims.pkl"
    
    # Try to load corpus from singleton or from pickle file on disk
    if is_claims:
        corpus = ClaimsCorpus().get_corpus()
    else:
        corpus = Corpus().get_corpus()

    if corpus:
        return corpus
    else:
        if load_from_pkl_file and args.input:
            # check if corpus file is available on disk
            if os.path.exists(f"{args.input}/{filename}"):
                with open(f"{args.input}/{filename}", "rb") as file:
                    corpus = pickle.load(file)
                    # after loading from pickle file, save corpus in singleton for future use
                    if is_claims:
                        ClaimsCorpus().set_corpus(corpus)
                    else:
                        Corpus().set_corpus(corpus)
                    return corpus

    # If the previous options are not available, load corpus from jsonl files on disk
    # That will take significantly longer...
    corpus = defaultdict(default_doc)

    if os.path.isdir(path):
        files = [f for f in glob.glob(f"{path}/{path_prefix}")]
    else:
        files = [path]

    for file in tqdm(files, total=len(files)):
        with open(file, "r") as f:
            for line in tqdm(f):
                try:
                    corpus = read_claims_line(line, corpus) if is_claims else read_doc_line(line, corpus)
                except:
                    continue

    if load_from_pkl_file and args.input:
        with open(f"{args.input}/{filename}", "wb") as f:
            pickle.dump(corpus, f)
    
    if is_claims:
        ClaimsCorpus().set_corpus(corpus)
    else:
        Corpus().set_corpus(corpus)

    return corpus

def read_doc_line(line, corpus):
    id_field = "id"
    content_field = "contents"

    data = json.loads(line.strip())

    if "id" not in data.keys():
        id_field = "_id"

    if "contents" not in data.keys():
        content_field = "text"

    docid = data[id_field]
    title = data.get("title", "").strip()
    text = data.get(content_field, "").strip()
    corpus[str(docid)] = {"title": title, "text": text}
    return corpus

def read_claims_line(line, corpus):
    id_field = "id"
    statements_field = "statements"
    
    data = json.loads(line.strip())

    if "id" not in data.keys():
        id_field = "_id"

    if "statements" not in data.keys():
        statements_field = "text"

    docid = data[id_field]
    # title = data.get("title", "").strip()
    claims = data.get(statements_field, [])
    corpus[str(docid)] = claims
    return corpus
