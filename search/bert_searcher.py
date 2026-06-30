# BERT search backend for the web app (app.py)
# app.py imports search_bert() from this file to handle BERT-based queries

import os
import pickle

import faiss
import numpy as np


# --- index location ---
# FAISS indexes live under data/<model>_index by default (see indexing/bert_indexer.py).
# data/ is not committed to git (it's large, generated output), so override WIKI_DATA_DIR
# if you built the indexes somewhere else.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("WIKI_DATA_DIR", os.path.join(_REPO_ROOT, "data"))

# maps a model key to the folder where its FAISS index and metadata are stored
# "bert" uses bert-base-uncased, "msmarco" uses msmarco-distilbert-base-v4
INDEX_DIRS = {
    "bert": os.path.join(DATA_DIR, "bert_index"),
    "msmarco": os.path.join(DATA_DIR, "msmarco_index"),
}

# maps a model key to the actual huggingface model name
MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "msmarco": "msmarco-distilbert-base-v4",
}

# these dictionaries cache loaded models/indexes so we only load them once
# loading a model takes a few seconds, so we avoid doing it on every search
loaded_models = {}     # model_key -> SentenceTransformer model
loaded_indexes = {}    # model_key -> FAISS index
loaded_metadata = {}   # model_key -> metadata list (title, url, passage for each vector)


def load_bert(model_key):
    """Load a BERT model, FAISS index, and metadata. Only runs once per model."""

    # skip if already loaded (avoids reloading on every search)
    if model_key in loaded_indexes:
        return

    from sentence_transformers import SentenceTransformer
    import torch

    # use GPU if available, otherwise fall back to CPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Loading {model_key} on device: {device}")

    # load the BERT model that will convert search queries into embeddings
    # needs to be the same model used during indexing so the vectors are compatible
    model_name = MODEL_NAMES[model_key]
    print("Loading model:", model_name)
    loaded_models[model_key] = SentenceTransformer(model_name, device=device)

    # load the FAISS index (contains all the passage embeddings from bert_indexer.py)
    # and the metadata (title, url, passage text for each embedding)
    index_dir = INDEX_DIRS[model_key]
    index_path = os.path.join(index_dir, "faiss_index.bin")
    meta_path = os.path.join(index_dir, "metadata.pkl")

    print("Loading FAISS index from", index_path)
    loaded_indexes[model_key] = faiss.read_index(index_path)

    print("Loading metadata from", meta_path)
    with open(meta_path, "rb") as f:
        loaded_metadata[model_key] = pickle.load(f)

    print(f"{model_key} searcher ready.", loaded_indexes[model_key].ntotal, "passages in index.")


def search_bert(query, k=10, model_key="bert"):
    """
    Search a BERT FAISS index with a query string.
    model_key can be "bert" or "msmarco".
    Returns a list of results, each is a dictionary with:
        rank, score, title, url, passage
    """

    # load the model and index if this is the first search
    load_bert(model_key)

    model = loaded_models[model_key]
    faiss_index = loaded_indexes[model_key]
    metadata = loaded_metadata[model_key]

    # encode the query into a vector using the same BERT model
    # this vector will be compared against all passage vectors in the index
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalize so dot product = cosine similarity
    ).astype("float32")

    # search the FAISS index for the most similar passages
    # fetch 3x more than needed because some results may be from the same document
    fetch_k = k * 3
    scores, indices = faiss_index.search(query_vector, fetch_k)

    # deduplicate: a long document gets split into many passages, so multiple
    # passages from the same doc might match. we only keep the best one per URL
    seen_urls = set()
    results = []
    for score, idx in zip(scores[0], indices[0]):
        # FAISS returns -1 if there aren't enough results in the index
        if idx == -1:
            break

        meta = metadata[idx]
        url = meta["url"]

        # skip if we already have a result from this document
        if url in seen_urls:
            continue
        seen_urls.add(url)

        results.append({
            "rank": len(results) + 1,
            "score": round(float(score), 4),
            "title": meta["title"],
            "url": url,
            "passage": meta["passage"],
        })

        # stop once we have enough unique results
        if len(results) >= k:
            break

    return results
