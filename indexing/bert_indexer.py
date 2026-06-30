# Builds a dense passage index: chunks crawled documents into BERT-sized
# passages, encodes them with a sentence-transformers model, and writes a
# FAISS index + metadata file that bert_searcher.py can load at query time.
#
# Usage:
#   python bert_indexer.py --input_dir ../data/crawled --output_dir ../data/msmarco_index
#   python bert_indexer.py --input_dir ../data/crawled --output_dir ../data/bert_index --model_name bert-base-uncased

import argparse
import json
import os
import pickle
import time
import glob

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# --- parse command line arguments ---
# lets you run: python bert_indexer.py --input_dir input --output_dir msmarco_index
# argparse reads the flags from the terminal and stores them in 'args'
parser = argparse.ArgumentParser(description="Build a FAISS index from BERT embeddings.")
parser.add_argument("--input_dir", type=str, required=True, help="Folder with .jsonl files")
parser.add_argument("--output_dir", type=str, required=True, help="Folder to save index files")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
parser.add_argument(
    "--model_name",
    type=str,
    default="msmarco-distilbert-base-v4",
    help="sentence-transformers model to encode passages with "
         "(e.g. msmarco-distilbert-base-v4 or bert-base-uncased)",
)
args = parser.parse_args()

# create output folder if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# --- pick device ---
# GPU is way faster for BERT encoding, but falls back to CPU if no GPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Using device:", device)

# --- load the sentence-transformers model ---
# msmarco-distilbert-base-v4 is a BERT model fine-tuned for search/retrieval tasks
# it converts text into 768 dimensional vectors (embeddings)
model_name = args.model_name
print("Loading model:", model_name)
model = SentenceTransformer(model_name, device=device)
tokenizer = model.tokenizer  # grab the tokenizer so we can split long docs into chunks

# --- read all the documents from JSONL files ---
# each line in the JSONL file is one wikipedia document with url, title, and text
print("Loading documents from", args.input_dir)
documents = []
jsonl_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
for filepath in jsonl_files:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

print("Loaded", len(documents), "documents")
if len(documents) == 0:
    print("No documents found, exiting.")
    exit()

# --- split documents into passages ---
# BERT has a max input of 512 tokens, so long documents need to be split into smaller chunks
# each chunk becomes a "passage" that gets its own embedding
all_passages = []   # list of passage text strings
metadata = []       # stores title, url, and text for each passage so we can show results later
timing_log = []     # tracks how long processing takes at each 100-doc interval

start_time = time.time()

for i, doc in enumerate(documents):
    text = doc.get("text", "")
    title = doc.get("title", "")
    url = doc.get("url", "")

    # tokenize the full document text into token IDs
    # token IDs are numbers that represent words/subwords in BERT's vocabulary
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    max_tokens = 512
    stride = 64  # overlap between chunks so we don't lose context at chunk boundaries

    # slide a window of 512 tokens across the document, moving forward by (512 - 64) = 448 tokens each step
    pos = 0
    while pos < len(token_ids):
        end = min(pos + max_tokens, len(token_ids))
        chunk = token_ids[pos:end]
        # convert token IDs back to readable text for this chunk
        passage_text = tokenizer.decode(chunk, skip_special_tokens=True).strip()

        if passage_text:
            all_passages.append(passage_text)
            metadata.append({
                "title": title,
                "url": url,
                "passage": passage_text,
            })

        # if we reached the end of the document, stop chunking
        if end == len(token_ids):
            break
        pos += max_tokens - stride

    # print progress every 100 documents so we know it's still working
    docs_done = i + 1
    if docs_done % 100 == 0 or docs_done == len(documents):
        elapsed = time.time() - start_time
        timing_log.append({"docs_processed": docs_done, "elapsed_sec": round(elapsed, 2)})
        if docs_done % 100 == 0:
            print(f"  Processed {docs_done}/{len(documents)} docs - {elapsed:.1f}s so far")

print("Total passages:", len(all_passages))

if len(all_passages) == 0:
    print("No passages created, exiting.")
    exit()

# --- encode all passages into embeddings ---
# this is the most expensive step - BERT reads each passage and outputs a 768-dim vector
# similar passages will have similar vectors (close together in vector space)
# normalize_embeddings=True makes it so dot product = cosine similarity
print("Encoding passages into embeddings...")
embeddings = model.encode(
    all_passages,
    batch_size=args.batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

embeddings = embeddings.astype("float32")
total_time = time.time() - start_time

# --- build the FAISS index ---
# FAISS is Facebook's library for fast similarity search over large sets of vectors
# IndexFlatIP = brute-force inner product search (exact, not approximate)
# when vectors are normalized, inner product = cosine similarity
dim = embeddings.shape[1]  # 768 for msmarco-distilbert-base-v4

index = faiss.IndexFlatIP(dim)
index.add(embeddings)  # add all passage embeddings to the index

# --- save everything to disk so we don't have to re-encode every time ---
# faiss_index.bin = the searchable vector index
# metadata.pkl = title/url/passage info for each vector (needed to display results)
# timing_log.json = how long each step took (for the runtime plot)
index_path = os.path.join(args.output_dir, "faiss_index.bin")
meta_path = os.path.join(args.output_dir, "metadata.pkl")
timing_path = os.path.join(args.output_dir, "timing_log.json")

faiss.write_index(index, index_path)

with open(meta_path, "wb") as f:
    pickle.dump(metadata, f)

with open(timing_path, "w", encoding="utf-8") as f:
    json.dump(timing_log, f, indent=2)

# --- print summary ---
print()
print("Done! Total time:", round(total_time, 2), "seconds")
print("Documents indexed:", len(documents))
print("Passages indexed:", len(metadata))
print("FAISS index saved to:", index_path)
print("Metadata saved to:", meta_path)
print("Timing log saved to:", timing_path)
