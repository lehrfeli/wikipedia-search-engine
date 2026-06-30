# data/

This folder holds generated artifacts: the crawled Wikipedia corpus, the Lucene
BM25 index, and the two FAISS dense-vector indexes. Everything in here except
this file is excluded from git (see `.gitignore`) because it can total several
gigabytes.

Expected layout once you've run the pipeline (see the root `README.md` for the
exact commands):

```
data/
├── crawled/            # wiki_data.jsonl - raw crawler output
├── pyserini_input/      # crawled JSONL converted into Pyserini's JsonCollection format
├── indexes/
│   └── wiki_index/      # Lucene BM25 index (built by indexing/pyserini_indexer.py)
├── bert_index/           # FAISS index for bert-base-uncased
└── msmarco_index/        # FAISS index for msmarco-distilbert-base-v4
```
