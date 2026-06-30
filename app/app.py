# Main web app for the search engine
# Run with: waitress-serve --port=5000 app:app
# Then open http://localhost:5000 in your browser

import os
import sys
import time

from flask import Flask, render_template, request

# --- make the search backends importable ---
# bert_searcher.py and lucene_searcher.py live in ../search relative to this file,
# so add that folder to the import path before importing them.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "search"))

# import the search functions from our backend files
from bert_searcher import search_bert
from lucene_searcher import search_lucene


# create the Flask app instance
# Flask looks for html templates in the "templates/" folder by default
app = Flask(__name__)


# home page route — shows the search bar
@app.route("/")
def home():
    return render_template("index.html")


# search route: handles the query when the user hits search
# the browser sends the query as URL parameters like /search?query=law&index_type=lucene&top_k=10
@app.route("/search")
def search():

    # pull the query, model choice, and number of results from the URL parameters
    query = request.args.get("query", "").strip()
    index_type = request.args.get("index_type", "lucene")  # "lucene", "bert", or "msmarco"
    top_k = request.args.get("top_k", "10")

    # convert top_k to int (default to 10 if something weird is entered)
    try:
        top_k = int(top_k)
    except ValueError:
        top_k = 10

    # if the query is empty, just go back to the homepage
    if not query:
        return render_template("index.html")

    # run the search using whichever model the user selected and time it
    start_time = time.time()

    if index_type == "bert":
        results = search_bert(query, k=top_k, model_key="bert")
    elif index_type == "msmarco":
        results = search_bert(query, k=top_k, model_key="msmarco")
    else:
        results = search_lucene(query, k=top_k)

    elapsed = time.time() - start_time
    elapsed = round(elapsed, 4)

    # pass everything to the results template so it can display the search results
    return render_template(
        "results.html",
        query=query,
        index_type=index_type,
        top_k=top_k,
        results=results,
        elapsed=elapsed,
        num_results=len(results),
    )
