# file: embed_chunks.py

from sentence_transformers import SentenceTransformer
import numpy as np

# 1) Load an open-source embedding model (no API required)
model = SentenceTransformer("all-mpnet-base-v2")  
# This is one of the best free embedding models

def embed_chunks(chunks):
    """
    Takes a list of chunk objects (from Step 1)
    Adds an embedding vector to each chunk.
    """

    # Extract the text for each chunk
    texts = [chunk["text"] for chunk in chunks]

    # 2) Convert each chunk of text into a vector (embedding)
    vectors = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # important for similarity search
    )

    # 3) Attach embedding to each chunk
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.astype(np.float32)

    return chunks


# Manual test example (optional)
if __name__ == "__main__":
    import pickle
    from code.ingest_and_chunk import ingest_document

    # 1. Load your document
    chunks = ingest_document("data/hackerrank_doc.txt")

    # 2. Generate embeddings
    embedded_chunks = embed_chunks(chunks)

    # 3. Save to a pickle file
    with open("data/chunks_with_embeddings.pkl", "wb") as f:
        pickle.dump(embedded_chunks, f)

    print("Saved chunks_with_embeddings.pkl successfully!")
