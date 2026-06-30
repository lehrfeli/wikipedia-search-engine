# Converts crawled Wikipedia JSONL data into Pyserini's JsonCollection format
# and builds a Lucene BM25 index from it using Pyserini.
#
# This is the standalone, script-ified version of the conversion + indexing
# cell from notebooks/cs242_A2.ipynb (same logic, just parameterized with
# argparse instead of hardcoded filenames).
#
# Usage:
#   python pyserini_indexer.py --input ../data/crawled/wiki_data.jsonl \
#       --pyserini_dir ../data/pyserini_input --index_dir ../data/indexes/wiki_index

import argparse
import json
import os
import subprocess


# --- convert crawled JSONL into Pyserini's expected document format ---
def convert_to_pyserini_format(input_path, pyserini_dir):
    """
    Reads crawled JSONL (url, title, text) and rewrites it into Pyserini's
    JsonCollection format (id, contents, title). Returns the folder Pyserini
    should index from.
    """
    os.makedirs(pyserini_dir, exist_ok=True)
    output_path = os.path.join(pyserini_dir, "wiki_docs.jsonl")

    with open(input_path, "r", encoding="utf-8") as src_file, \
         open(output_path, "w", encoding="utf-8") as dest_file:

        # apply to each line in each scraped wikipedia page
        for line in src_file:
            data = json.loads(line)

            # create Pyserini document
            pyserini_doc = {
                "id": data["url"],          # url acts as unique identifier
                "contents": data["text"],   # contents to be indexed
                "title": data["title"],
            }

            # write one JSON object per line
            dest_file.write(json.dumps(pyserini_doc) + "\n")

    return pyserini_dir


# --- build the Lucene index from the converted documents ---
def build_lucene_index(pyserini_dir, index_dir, threads=1):
    """Runs Pyserini's Lucene indexer over the converted JsonCollection folder."""
    command = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",              # input format
        "--input", pyserini_dir,                        # directory containing jsonl docs
        "--index", index_dir,                            # output index directory
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    # --- parse command line arguments ---
    parser = argparse.ArgumentParser(
        description="Convert crawled JSONL into Pyserini format and build a Lucene BM25 index."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the crawled wiki_data.jsonl file")
    parser.add_argument("--pyserini_dir", type=str, required=True, help="Folder to write the converted Pyserini-format JSONL")
    parser.add_argument("--index_dir", type=str, required=True, help="Folder to write the Lucene index")
    parser.add_argument("--threads", type=int, default=1, help="Number of indexing threads")
    args = parser.parse_args()

    print("Converting", args.input, "to Pyserini format...")
    convert_to_pyserini_format(args.input, args.pyserini_dir)

    print("Building Lucene index at", args.index_dir)
    build_lucene_index(args.pyserini_dir, args.index_dir, args.threads)

    print("Done. Lucene index saved to:", args.index_dir)
