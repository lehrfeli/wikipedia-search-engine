# Wikipedia Search Engine

An end-to-end information retrieval system built for UCR CS242 (Information Retrieval). It crawls
Wikipedia, indexes the crawled pages with two different retrieval paradigms — classic lexical
search (BM25 via Lucene/Pyserini) and dense neural search (BERT embeddings via FAISS) — and serves
all of it through a small Flask app so the three methods can be compared side by side on the same
queries.

## Architecture

```
crawler/                indexing/                       search/                  app/
┌────────────────┐      ┌─────────────────────────┐     ┌────────────────────┐   ┌──────────┐
│ wiki_spider.py  │ ───> │ data/crawled/            │     │                     │   │          │
│ (Scrapy, BFS    │      │   wiki_data.jsonl        │     │                     │   │          │
│ from seed URL)  │      └─────────────────────────┘     │                     │   │          │
└────────────────┘               │         │              │                     │   │          │
                                  │         │              │                     │   │          │
                  pyserini_indexer.py    bert_indexer.py    │                     │   │          │
                          │                  │              │                     │   │          │
                          v                  v              │                     │   │          │
              data/indexes/wiki_index   data/bert_index/      lucene_searcher.py ─┼──>│  app.py   │
                                          data/msmarco_index/  bert_searcher.py  ─┘   │ (Flask)  │
                                                                                       └──────────┘
```

1. **Crawler** (`crawler/`) walks Wikipedia starting from a seed article and writes one JSON object
   per page (`url`, `title`, `text`) to a JSONL file.
2. **Indexing** (`indexing/`) turns that JSONL into two kinds of search index:
   - a Lucene **BM25** index (via Pyserini)
   - two FAISS **dense vector** indexes (one per sentence-transformers model)
3. **Search** (`search/`) provides query-time functions that load an index and return ranked
   results.
4. **App** (`app/`) is a Flask web app that lets a user pick a retrieval method and see results,
   with timing, for the same query.

## Demo

![Search results page](docs/screenshot.png)

## How the three retrieval methods differ

| Method | Backend | How it ranks documents |
|---|---|---|
| **BM25** | Lucene (via Pyserini) | Classic lexical/keyword scoring. Ranks documents by term-frequency and inverse-document-frequency, adjusted for document length. Fast, exact keyword matching, but has no notion of meaning — synonyms or paraphrases won't match. |
| **BERT Base** | `bert-base-uncased` (sentence-transformers) | Dense semantic search. Encodes each passage and each query into a 768-dim vector and ranks by cosine similarity (via FAISS inner-product search on normalized vectors). `bert-base-uncased` is a general-purpose language model, not fine-tuned for retrieval, so it's used here as a semantic-search baseline. |
| **MS-MARCO DistilBERT** | `msmarco-distilbert-base-v4` (sentence-transformers) | Same dense-vector approach as BERT Base, but with a DistilBERT model fine-tuned specifically on the MS-MARCO passage-ranking dataset for query/passage relevance. It's expected to retrieve more relevant passages than the generic BERT Base model, at similar speed (it's distilled, so cheaper to run than full BERT). |

Both dense methods chunk long documents into overlapping 512-token passages before encoding
(BERT's max input length), then deduplicate results by URL at query time so a single
highly-relevant document doesn't fill the whole results page.

## Repository structure

```
crawler/
  wiki_spider.py        Scrapy spider: crawls from a seed URL, saves url/title/text per page
  crawler.bat            Windows runner: scrapy runspider wiki_spider.py with CLI args
  seed.txt                Starting URL (https://en.wikipedia.org/wiki/Law)

indexing/
  pyserini_indexer.py    Converts crawled JSONL -> Pyserini format, builds the Lucene BM25 index
  bert_indexer.py         Chunks + encodes documents with a sentence-transformers model, builds a FAISS index
  bert_benchmark.py       Benchmarks BERT indexing time at increasing document counts
  indexer.bat              Windows runner for bert_indexer.py

search/
  lucene_searcher.py     search_lucene(query, k) - BM25 search against the Lucene index
  bert_searcher.py        search_bert(query, k, model_key) - dense search against a FAISS index

app/
  app.py                  Flask app: routes for the home page and /search
  templates/                index.html, results.html
  static/                   style.css

notebooks/
  cs242_A2.ipynb          Original exploration notebook (indexing benchmarks/plots, ad-hoc Lucene queries)

data/                      Generated artifacts (crawled JSONL, Lucene index, FAISS indexes) - not in git, see below
```

## Setup

Requires Python 3.9+ and a JDK (Java 21) for Pyserini/Lucene.

```bash
pip install -r requirements.txt
```

## Running the pipeline (in order)

All commands below assume your current directory is the matching subfolder (`crawler/`,
`indexing/`, `search/`, or `app/`), and that paths are given relative to that folder.

### 1. Crawl Wikipedia

```bash
cd crawler
crawler.bat seed.txt 100000 6 ..\data\crawled
```

equivalently, without the batch wrapper:

```bash
scrapy runspider wiki_spider.py -a seed_file=seed.txt -a num_pages=100000 -a hops=6 -a output_dir=../data/crawled
```

This crawls breadth-first from the seed URL (the "Law" article), following in-article links,
stopping after `num_pages` pages, and writes `data/crawled/wiki_data.jsonl` — one JSON object
per page: `{"url": ..., "title": ..., "text": ...}`. Pages under 500 characters of body text are
skipped.

### 2. Build the BM25 (Lucene) index

```bash
cd indexing
python pyserini_indexer.py --input ../data/crawled/wiki_data.jsonl --pyserini_dir ../data/pyserini_input --index_dir ../data/indexes/wiki_index
```

This rewrites the crawled JSONL into Pyserini's `JsonCollection` format (`id`, `contents`,
`title`) and runs Pyserini's Lucene indexer over it.

### 3. Build the BERT FAISS indexes

The app compares two embedding models, so `bert_indexer.py` needs to be run twice — once per
model — into two different output folders:

```bash
cd indexing
python bert_indexer.py --input_dir ../data/crawled --output_dir ../data/msmarco_index --model_name msmarco-distilbert-base-v4
python bert_indexer.py --input_dir ../data/crawled --output_dir ../data/bert_index --model_name bert-base-uncased
```

(`indexer.bat <input-dir> <output-dir>` runs the same script with the default model,
`msmarco-distilbert-base-v4`.)

Each run chunks every document into overlapping 512-token passages, encodes them in batches with
the chosen sentence-transformers model, and writes:
- `faiss_index.bin` — the FAISS `IndexFlatIP` vector index
- `metadata.pkl` — title/url/passage text for every vector, used to render results
- `timing_log.json` — processing time at each 100-document checkpoint

### 4. Run the web app

```bash
cd app
waitress-serve --port=5000 app:app
```

`app.py` doesn't define a `__main__` block, so it's meant to be served with `waitress-serve` (or
any other WSGI server) rather than run directly with `python app.py`.

Open `http://localhost:5000`, enter a query, choose **Lucene (BM25)**, **BERT Base (Semantic)**, or
**MSMARCO DistilBERT (Semantic)**, and submit. The results page shows, for each hit: rank, score,
title, URL, and a passage snippet, plus the total query time and how many results were returned.

By default the search backends look for indexes under `<repo root>/data/`. If you built your
indexes somewhere else, point `search/lucene_searcher.py` and `search/bert_searcher.py` at them by
setting the `WIKI_DATA_DIR` environment variable to that parent folder before starting the app.

## Data is not included in this repo

`data/crawled/wiki_data.jsonl`, the Lucene index, and the two FAISS indexes are excluded from git
(see `.gitignore`) — together they're several gigabytes, and they're fully reproducible from the
crawler and indexing scripts. To regenerate everything from scratch, follow steps 1–3 above in
order. `data/README.md` documents the expected folder layout.

## Notebook

`notebooks/cs242_A2.ipynb` is the original exploration notebook used during development: ad-hoc
Lucene queries, and the runtime benchmarking/plotting (Lucene indexing time vs. document count,
compared against `bert_benchmark.py`'s output) used to evaluate indexing performance. It's kept for
reference; the indexing/search logic it explores now lives in `indexing/pyserini_indexer.py`,
`indexing/bert_indexer.py`, and `search/`.
