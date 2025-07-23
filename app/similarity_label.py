from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE = os.getenv("PINECONE")
model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE)
index_name = "semantic-search-vietnamese"

index = pc.Index(index_name)

def search_label(query, top_k=3):
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [(match["metadata"]["label"]) for match in results["matches"]]

def get_best_label_from_content(content, labels, top_k=3):
    for label in labels:
        if label in content:
            result = search_label(label, top_k=top_k)
            if result:
                return result

    for label in labels:
        result = search_label(label, top_k=top_k)
        if result:
            return result

    return []
