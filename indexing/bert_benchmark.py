"""
Benchmark full BERT indexing time (chunking + encoding + FAISS)
at different document counts, comparable to the Lucene runtime plot.

Usage: python bert_benchmark.py --input_dir input --output_dir msmarco_index
"""

import argparse
import json
import os
import glob
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# --- parse command line arguments ---
parser = argparse.ArgumentParser(description="Benchmark BERT indexing runtime at different doc counts.")
parser.add_argument("--input_dir", type=str, required=True, help="Folder with .jsonl files")
parser.add_argument("--output_dir", type=str, required=True, help="Folder to save benchmark results")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# --- pick device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --- load model (done once, not included in timing) ---
model_name = "msmarco-distilbert-base-v4"
print("Loading model:", model_name)
model = SentenceTransformer(model_name, device=device)
tokenizer = model.tokenizer

# --- load all documents ---
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

# --- document counts to benchmark (same as Lucene plot) ---
doc_counts = [100, 500, 1000, 2000, 5000, 10000]
benchmark_results = []

for n in doc_counts:
    if n > len(documents):
        print(f"Skipping {n} (only {len(documents)} docs available)")
        break

    subset = documents[:n]
    print(f"\n--- Benchmarking {n} documents ---")

    start_time = time.time()

    # Step 1: Split documents into passages
    all_passages = []
    metadata = []

    for doc in subset:
        text = doc.get("text", "")
        title = doc.get("title", "")
        url = doc.get("url", "")

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        max_tokens = 512
        stride = 64

        pos = 0
        while pos < len(token_ids):
            end = min(pos + max_tokens, len(token_ids))
            chunk = token_ids[pos:end]
            passage_text = tokenizer.decode(chunk, skip_special_tokens=True).strip()

            if passage_text:
                all_passages.append(passage_text)
                metadata.append({"title": title, "url": url, "passage": passage_text})

            if end == len(token_ids):
                break
            pos += max_tokens - stride

    # Step 2: Encode all passages into embeddings
    embeddings = model.encode(
        all_passages,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Step 3: Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    elapsed = time.time() - start_time

    benchmark_results.append({
        "docs_indexed": n,
        "passages": len(all_passages),
        "elapsed_sec": round(elapsed, 2),
    })
    print(f"  {n} docs -> {len(all_passages)} passages in {elapsed:.2f}s")

# --- save benchmark results ---
benchmark_path = os.path.join(args.output_dir, "benchmark_results.json")
with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark_results, f, indent=2)

print(f"\nBenchmark results saved to: {benchmark_path}")
for r in benchmark_results:
    print(f"  {r['docs_indexed']:>6} docs | {r['passages']:>6} passages | {r['elapsed_sec']:>8.2f}s")
