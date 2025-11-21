# file: answer_with_provenance.py

from groq import Groq
from code.retriever_chroma import retrieve_chunks
from textwrap import dedent
import os

# Load your Groq API key
client = Groq(api_key="GROQ_API_KEY")

def build_prompt(question, retrieved_chunks):
    """
    Builds a prompt that forces the LLM to ONLY answer using the retrieved chunks.
    Includes citations like [Chunk 0], [Chunk 1], etc.
    """

    context = ""
    for idx, chunk in enumerate(retrieved_chunks):
        source = chunk["metadata"].get("source", "Unknown Source")
        context += f"\n[Chunk {idx}] (Source: {source})\n{chunk['text']}\n"

    prompt = f"""
You are a helpful AI assistant. You must answer ONLY using the information from the provided chunks.

If the answer is not found in the chunks, say:
"I don’t have information about this in the document."

Use citations like [Chunk 0], [Chunk 1], etc.

--------------------------
QUESTION: {question}
--------------------------

CONTEXT:
{context}

Now give the best possible answer using ONLY the context chunks.
Cite every statement with its chunk number.
"""
    return dedent(prompt)


def answer_question(question):
    """
    Full pipeline:
    1. Retrieve relevant chunks
    2. Build prompt
    3. Ask Groq LLaMA-3 to answer using only provided chunks
    """

    retrieved = retrieve_chunks(question, top_k=4)

    prompt = build_prompt(question, retrieved)

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)


    return response.choices[0].message.content


# TEST
if __name__ == "__main__":
    q = "Explain regression."
    print(answer_question(q))
