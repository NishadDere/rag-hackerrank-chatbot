# file: retriever_chroma.py

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# NEW Chroma client (same as indexer)
client = chromadb.PersistentClient(path="chroma_db")

# Load the same collection created by index_chroma.py
collection = client.get_collection("hackerrank_chunks")

# Load the same embedding model used earlier
model = SentenceTransformer("all-mpnet-base-v2")


def retrieve_chunks(query, top_k=4):
    """
    Retrieves the most relevant chunks for the query using ChromaDB.
    """

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []

    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "score": results["distances"][0][i],
            "metadata": results["metadatas"][0][i]
        })

    return retrieved


# Debug test
if __name__ == "__main__":
    chunks = retrieve_chunks("What is regression?")
    for c in chunks:
        print("\n----")
        print(c)
