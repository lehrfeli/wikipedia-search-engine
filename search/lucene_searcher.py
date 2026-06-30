# Lucene search backend for the web app (app.py)
# app.py imports search_lucene() from this file to handle keyword-based BM25 queries

import json
import os


# --- index location ---
# the Lucene index lives in data/indexes/wiki_index by default (see indexing/pyserini_indexer.py).
# data/ is not committed to git (it's large, generated output), so override WIKI_DATA_DIR
# if you built the index somewhere else.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.environ.get("WIKI_DATA_DIR", os.path.join(_REPO_ROOT, "data"))
DEFAULT_INDEX_PATH = os.path.join(DEFAULT_DATA_DIR, "indexes", "wiki_index")

# cached searcher so we only load the index once (loading takes a few seconds)
searcher = None


def load_lucene():
    """Load the Lucene searcher. Only runs once."""
    global searcher

    # skip if already loaded (avoids reloading on every search)
    if searcher is not None:
        return

    # import here so the app doesn't crash if pyserini isn't installed
    # pyserini is a Python wrapper around the Java Lucene search library
    from pyserini.search.lucene import LuceneSearcher

    print("Loading Lucene index from", DEFAULT_INDEX_PATH)
    searcher = LuceneSearcher(DEFAULT_INDEX_PATH)
    print("Lucene searcher ready.")


def search_lucene(query, k=10):
    """
    Search the Lucene BM25 index with a query string.
    Returns a list of results, each is a dictionary with:
        rank, score, title, url, passage
    """

    # load the index if this is the first search
    load_lucene()

    # run the search using BM25 scoring (keyword-based ranking)
    # BM25 scores documents based on how often query terms appear,
    # adjusted for document length. unlike BERT, it doesn't understand meaning
    hits = searcher.search(query, k=k)

    # loop through each hit and pull out the document info
    results = []
    for rank in range(len(hits)):
        hit = hits[rank]

        # retrieve the full stored document from the index using its ID
        # doc.raw() returns the original JSON we indexed (title, contents, url)
        doc = searcher.doc(hit.docid)

        if doc:
            doc_json = json.loads(doc.raw())
            title = doc_json.get("title", "")
            contents = doc_json.get("contents", "")
        else:
            title = "No doc found"
            contents = ""

        # grab the first 500 characters as a preview snippet
        snippet = contents[:500]

        results.append({
            "rank": rank + 1,
            "score": round(float(hit.score), 4),
            "title": title,
            "url": hit.docid,        # the docid is the wikipedia url
            "passage": snippet,
        })

    return results
