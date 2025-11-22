# file: answer_with_provenance.py
from textwrap import dedent
from code.retriever_chroma import retrieve_chunks
from groq import Groq
import os

# Load Groq client from env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# CONFIG: thresholds and defaults
CONFIDENCE_HEURISTIC_BASE = 0.4   # base confidence when at least one chunk found
CONFIDENCE_PER_CHUNK = 0.15       # add per supporting chunk (capped)
MAX_CONFIDENCE = 0.95

def compute_confidence(retrieved_chunks):
    """
    Heuristic confidence:
    - If zero chunks -> 0.0
    - else -> base + per_chunk * n (capped)
    NOTE: replace this with real scoring when retriever returns scores.
    """
    if not retrieved_chunks:
        return 0.0
    n = len(retrieved_chunks)
    conf = CONFIDENCE_HEURISTIC_BASE + CONFIDENCE_PER_CHUNK * min(n, 4)
    return min(conf, MAX_CONFIDENCE)

def build_prompt_strict(question, retrieved_chunks, chat_history_text=""):
    """
    Strict prompt: must only use provided chunks and cite them.
    """
    context = ""
    for idx, chunk in enumerate(retrieved_chunks):
        src = chunk["metadata"].get("source", "Unknown")
        context += f"\n[Chunk {idx}] (Source: {src})\n{chunk['text']}\n"

    prompt = f"""
You are a fact-focused assistant. You MUST answer ONLY using the information in the CONTEXT chunks.
If the answer is not present in the chunks, reply exactly: "I don’t have information about this in the document."

Be concise. Cite every factual statement with its chunk number, e.g., [Chunk 0].
Do NOT use outside knowledge.
Chat history (for context):
{chat_history_text}

QUESTION:
{question}

CONTEXT:
{context}

Answer following the rules above.
"""
    return dedent(prompt)

def build_prompt_hybrid(question, retrieved_chunks, chat_history_text=""):
    """
    Hybrid prompt: prefer chunks but allow careful synthesis and a short disclaimer if
    answer primarily uses outside knowledge.
    """
    context = ""
    for idx, chunk in enumerate(retrieved_chunks):
        src = chunk["metadata"].get("source", "Unknown")
        context += f"\n[Chunk {idx}] (Source: {src})\n{chunk['text']}\n"

    prompt = f"""
You are a helpful assistant that uses document context to answer. Use the CONTEXT chunks first.
If you must go beyond the chunks to complete an answer, say a short disclaimer like:
'Disclaimer: the following is a best-effort answer based on partial information.'

Cite chunks for each factual claim when possible with [Chunk 0], [Chunk 1], etc.
Chat history (for context):
{chat_history_text}

QUESTION:
{question}

CONTEXT:
{context}

Now answer concisely, cite chunks for facts, and if you used extra knowledge add a one-line disclaimer.
"""
    return dedent(prompt)


def format_chunk_preview(retrieved_chunks, max_chars=300):
    """
    Build short previews for UI display (trim long chunks).
    Returns a list of (idx, source, preview_text).
    """
    previews = []
    for idx, c in enumerate(retrieved_chunks):
        src = c["metadata"].get("source", "Unknown")
        txt = c["text"].strip().replace("\n", " ")
        preview = txt[:max_chars]
        if len(txt) > max_chars:
            preview += "..."
        previews.append((idx, src, preview))
    return previews


def answer_question(question, chat_history=None, mode="strict", show_citations=True):
    """
    Main RAG answering function.

    - question: string (could include recent chat history text)
    - chat_history: list of {question, answer} dicts (optional)
    - mode: "strict" or "hybrid"
    - show_citations: True/False - whether to include citations in returned text

    Returns:
      dict {
        "answer": str,
        "confidence": float (0.0-1.0),
        "chunks": [retrieved chunk dicts],
        "previews": [ (idx, source, preview_text) ]
      }
    """

    # Build chat history text for context (if provided)
    chat_history_text = ""
    if chat_history:
        for turn in chat_history[-4:]:
            chat_history_text += f"User: {turn['question']}\nBot: {turn['answer']}\n\n"

    # 1) Retrieve candidate chunks (retriever must return list of dicts with text & metadata)
    retrieved = retrieve_chunks(question, top_k=6)  # get up to 6 for hybrid rerank/summary

    # 2) Compute confidence heuristic
    confidence = compute_confidence(retrieved)

    # 3) Build prompt depending on mode
    if mode == "strict":
        prompt = build_prompt_strict(question, retrieved, chat_history_text)
    else:
        prompt = build_prompt_hybrid(question, retrieved, chat_history_text)

    # 4) Query Groq model
    # Use a modern supported model name — set via env or change here (example: 'llama-3.1-8b-instant')
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    # groq returns a message object
    text = response.choices[0].message.content

    # 5) If strict and the model didn't follow instructions, do a safety-check:
    #    If the text contains phrases like "I don’t have information" - it's OK.
    #    Otherwise, if strict mode but model produced content that cites nothing and retrieved empty, override.
    if mode == "strict" and not retrieved:
        # Enforce strict behavior
        return {
            "answer": "I don’t have information about this in the document.",
            "confidence": 0.0,
            "chunks": [],
            "previews": []
        }

    # 6) Build chunk previews (for UI)
    previews = format_chunk_preview(retrieved)

    # 7) Optionally strip or keep citations in returned answer based on show_citations
    if not show_citations:
        # naive remove of bracketed chunk citations (e.g., [Chunk 0])
        import re
        text = re.sub(r"\[Chunk\s*\d+\]", "", text)

    return {
        "answer": text.strip(),
        "confidence": round(confidence, 2),
        "chunks": retrieved,
        "previews": previews
    }
