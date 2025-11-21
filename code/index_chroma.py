# file: index_chroma.py

import chromadb

# NEW Chroma client style (2024+)
client = chromadb.PersistentClient(path="chroma_db")

def index_in_chroma(chunks):
    """
    Saves chunk embeddings and metadata into a Chroma vector database.
    """

    # Create or load collection
    collection = client.get_or_create_collection(
        name="hackerrank_chunks",
        metadata={"hnsw:space": "cosine"}  # similarity metric
    )

    ids = [chunk["chunk_id"] for chunk in chunks]
    embeddings = [chunk["embedding"].tolist() for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "doc_id": chunk["doc_id"],
            "source": chunk["source"],
            "para_idx": chunk["para_idx"],
            "chunk_idx": chunk["chunk_idx"]
        }
        for chunk in chunks
    ]

    # Insert into vector DB
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Indexed {len(ids)} chunks into Chroma!")


if __name__ == "__main__":
    import pickle
    
    with open("data/chunks_with_embeddings.pkl", "rb") as f:
        chunks = pickle.load(f)

    index_in_chroma(chunks)
