# file: retriever_chroma.py

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ───────────────────────────────────────────────
# 1. Load Chroma Client
# ───────────────────────────────────────────────
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="hackerrank_chunks",
    metadata={"hnsw:space": "cosine"}
)

# ───────────────────────────────────────────────
# 2. Load Embedding Model (Same as indexing)
# ───────────────────────────────────────────────
embedding_model = SentenceTransformer("all-mpnet-base-v2")


# ───────────────────────────────────────────────
# 3. Load Free Reranker Model
# ───────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")


# ───────────────────────────────────────────────
# Multi-query expansion (improves retrieval)
# ───────────────────────────────────────────────
def expand_question(question, num_expansions=3):
    """
    Rephrase the query internally to improve retrieval.
    (Basic version; can be replaced with LLM-based expansion)
    """
    expansions = [
        question,
        f"Explain {question}",
        f"What does {question} mean?",
        f"Definition of {question}",
    ]
    return expansions[:num_expansions]


# ───────────────────────────────────────────────
# Rerank retrieved chunks using BGE reranker
# ───────────────────────────────────────────────
def rerank(question, retrieved_docs):
    """
    Takes a list of retrieved docs and sorts them by relevance score.
    """

    if len(retrieved_docs) == 0:
        return []

    pairs = [[question, doc["text"]] for doc in retrieved_docs]

    # Prepare input to the reranker
    texts = [q + " [SEP] " + d for q, d in pairs]
    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        scores = reranker(**inputs).logits.squeeze()

    # Pair scores with docs
    scored_docs = list(zip(scores.tolist(), retrieved_docs))

    # Sort by relevance (descending)
    ranked = sorted(scored_docs, key=lambda x: x[0], reverse=True)

    return [doc for score, doc in ranked]


# ───────────────────────────────────────────────
# MAIN RETRIEVAL FUNCTION
# ───────────────────────────────────────────────
def retrieve_chunks(question, top_k=4):
    """
    Retrieve relevant chunks using Chroma + reranker.
    """

    # 1. Expand question to improve recall
    expanded_queries = expand_question(question)

    # 2. Embed queries
    query_embeddings = embedding_model.encode(expanded_queries).tolist()

    # 3. Query ChromaDB
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=10,  # get more for reranking
        include=["documents", "metadatas", "embeddings"]
    )

    # 4. Flatten and dedupe results
    seen = set()
    retrieved_docs = []

    for docs, metas in zip(results["documents"], results["metadatas"]):
        for text, meta in zip(docs, metas):
            if text not in seen:
                seen.add(text)
                retrieved_docs.append({
                    "text": text,
                    "metadata": meta
                })

    # 5. Rerank using BGE Reranker
    ranked_docs = rerank(question, retrieved_docs)

    # 6. Return top K
    return ranked_docs[:top_k]
