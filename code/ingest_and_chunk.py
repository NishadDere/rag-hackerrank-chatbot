# file: ingest_and_chunk.py
from typing import List
import re
import uuid
import tiktoken
from pathlib import Path

def naive_paragraph_split(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paras

def chunk_paragraph(paragraph: str, max_tokens=350, overlap_tokens=50, tokenizer_name="cl100k_base"):
    enc = tiktoken.get_encoding(tokenizer_name)
    toks = enc.encode(paragraph)
    chunks = []
    start = 0
    while start < len(toks):
        end = min(start + max_tokens, len(toks))
        chunk_tokens = toks[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(toks):
            break
        start = end - overlap_tokens
    return chunks

def ingest_document(path: str):
    text = Path(path).read_text(encoding='utf-8')
    paras = naive_paragraph_split(text)
    all_chunks = []
    for pidx, para in enumerate(paras):
        para_chunks = chunk_paragraph(para)
        for cidx, ch in enumerate(para_chunks):
            chunk_obj = {
                "doc_id": Path(path).stem,
                "para_idx": pidx,
                "chunk_idx": cidx,
                "chunk_id": str(uuid.uuid4()),
                "text": ch,
                "source": f"{path}#para{pidx}:chunk{cidx}"
            }
            all_chunks.append(chunk_obj)
    return all_chunks

if __name__ == "__main__":
    chunks = ingest_document("data/hackerrank_doc.txt")
    print(f"Total chunks: {len(chunks)}")
    print(chunks[0])
