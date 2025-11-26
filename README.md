🚀 RAG Hackerrank Chatbot

A full Retrieval-Augmented Generation (RAG) system with conversational memory, strict/hybrid answer modes, citations, confidence scoring, chunk previews, reranking, and a simple FastAPI web UI.

This project demonstrates a complete end-to-end RAG pipeline, from document ingestion → semantic chunking → embeddings → vector search → reranking → LLM answering with provenance → frontend chat interface.

Perfect for learning, extending, or adapting into a personal AI assistant.

⭐ Features
🔍 Retrieval & Ranking

ChromaDB persistent vector store

MPNet embeddings (768-dim) for high-quality retrieval

BGE Reranker for improved relevance ordering

Multi-query expansion for better recall

🧠 Smart Answering (RAG)

Strict mode → answer only from document (no hallucination)

Hybrid mode → uses document first, but can extend with external knowledge

Citation support ([Chunk X])

Confidence scoring (based on retrieved chunks)

Chunk previews for transparency

💬 Conversation Features

ChatGPT-style typing animation

Multi-turn memory (configurable context window)

Local browser session memory

Per-session system prompt

Toggleable UI controls (mode, citations, previews, dark mode)

📄 Document Support

Upload documents through /upload endpoint

Auto-save uploaded files

Future-ready pipeline for multi-document RAG

🌐 Web Frontend

Clean minimal UI

Dark mode

Confidence bar

Chunk preview panel

Local session persistence

Built with pure HTML/CSS/JS (no build tools)

🏗 Project Structure
rag-hackerrank-chatbot/
│
├── app.py                     # FastAPI backend
├── static/
│     └── index.html           # Web UI
│
├── code/
│     ├── ingest_and_chunk.py
│     ├── embed_chunks.py
│     ├── index_chroma.py
│     ├── retriever_chroma.py  # embeddings + reranker + multi-query
│     ├── answer_with_provenance.py
│     └── chatbot.py           # CLI version
│
├── data/
│     └── hackerrank_doc.txt
│
├── chroma_db/                 # vector store (ignored in git)
├── venv/                      # virtual environment (ignored)
├── .gitignore
└── README.md

🛠 Installation
1. Clone
git clone https://github.com/NishadDere/rag-hackerrank-chatbot.git
cd rag-hackerrank-chatbot

2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # on Windows

3. Install dependencies
pip install -r requirements.txt


(If no requirements.txt exists, generate one:)

pip freeze > requirements.txt

🔐 Environment Variables

Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-8b-instant


This is automatically loaded by dotenv.

📥 Prepare Your Document (RAG Pipeline)
Step 1 — Ingest & Chunk
python -m code.ingest_and_chunk

Step 2 — Embed
python -m code.embed_chunks

Step 3 — Index
python -m code.index_chroma

▶ Running the Backend Server
uvicorn app:app --reload


Backend should run at:

http://localhost:8000


Open the UI:

http://localhost:8000/static/index.html

🎨 Frontend UI Screenshots (Optional)

(You can add screenshots later here.)

🧪 CLI Version
python -m code.chatbot


Supports:

/mode strict|hybrid

/citations on|off

/preview on|off

chat history awareness

typing animation

🧩 Future Roadmap

Multi-document RAG

Document search & filtering

Semantic highlighting of cited chunks

Chunk heatmap visualization

User accounts + cloud session persistence

Optional Postgres/MongoDB for chat logs

Switchable embeddings & reranker models

Streaming responses (SSE / WebSockets)
