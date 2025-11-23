A Retrieval-Augmented Generation (RAG) chatbot built using:

ChromaDB (Vector database)

MPNet embeddings (all-mpnet-base-v2)

BGE Reranker (BAAI/bge-reranker-base)

Groq LLM (llama-3.1-8b-instant)

Conversational memory

Strict & Hybrid answer modes

Typing animation + chunk previews + confidence scoring

This chatbot answers questions by retrieving the most relevant sections (“chunks”) from a HackerRank-style document and generating answers with citations that trace back to the source text.

✨ Features
🔍 1. Retrieval-Augmented Generation (RAG)

Document → chunking → embeddings → stored in ChromaDB

Multi-query expansion improves retrieval recall

BGE reranker improves ranking relevance

Provenance: each answer links back to exact text chunks

🎭 2. Answer Modes

Strict Mode → Only uses document context

Hybrid Mode → Mixes document + model knowledge with disclaimer

Switch anytime using:

/mode strict
/mode hybrid

💬 3. Conversational Memory

Keeps last 4 conversation turns

Allows follow-up questions like:
“Explain it in simple words.”
“Give an example.”

🔎 4. Chunk Previews

Enable:

/preview on


Shows where the answer came from.

📚 5. Citations

Enable/disable:

/citations on
/citations off

🧠 6. Confidence Score

Each answer returns a 0.0–1.0 confidence value based on context coverage.

🎨 7. ChatGPT-style typing animation

Realistic type-writer effect in console.

📁 Project Structure
rag-hackerrank-chatbot/
│
├── code/
│   ├── ingest_and_chunk.py
│   ├── embed_chunks.py
│   ├── index_chroma.py
│   ├── retriever_chroma.py
│   ├── answer_with_provenance.py
│   ├── chatbot.py
│   └── __init__.py
│
├── data/
│   └── hackerrank_doc.txt
│
├── chroma_db/            # ignored
├── venv/                 # ignored
├── .gitignore
└── README.md

⚙️ Installation
1. Clone the repository
git clone https://github.com/NishadDere/rag-hackerrank-chatbot.git
cd rag-hackerrank-chatbot

2. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install dependencies
pip install -r requirements.txt


If you don’t have requirements.txt, generate it:

pip freeze > requirements.txt

4. Add your Groq API key

Create a .env file:

GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

🏗️ Data Processing & Indexing
Step 1 — Chunk the document
python -m code.ingest_and_chunk

Step 2 — Create embeddings
python -m code.embed_chunks

Step 3 — Index into ChromaDB
python -m code.index_chroma

🤖 Running the Chatbot
python -m code.chatbot

Example Commands:
/mode hybrid
/citations off
/preview on

Example Questions:
What is regression?
Explain in simple words.
Is regression supervised or unsupervised?
What are the steps of KNN?

🧪 Example Output (Strict Mode)
Bot: Regression predicts continuous values. [Chunk 0]
Regression models relationships between variables. [Chunk 2]
Confidence: 88%

🛡️ .gitignore Summary

This project safely ignores:

venv/

chroma_db/

.env

*.pkl

__pycache__/

So no API keys or local DB data are ever uploaded to GitHub.

🔮 Future Improvements

Web UI (FastAPI + React or Streamlit Support)

Better memory summarization

UI components for chunk previews

Evaluation metrics for retrieval quality

PDF/document ingestion

Fine-tuned domain models