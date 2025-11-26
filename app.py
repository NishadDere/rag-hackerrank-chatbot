# file: app.py
from dotenv import load_dotenv
load_dotenv()
import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from code.answer_with_provenance import answer_question  # our upgraded function
from pathlib import Path
import uuid


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    system_prompt: Optional[str] = ""
    user_prompt: str
    mode: Optional[str] = "strict"      # "strict" or "hybrid"
    show_citations: Optional[bool] = True
    session_id: Optional[str] = None
    chat_history: Optional[List[dict]] = None  # [{question,answer},...]

@app.post("/chat")
async def chat(req: ChatRequest):
    # Build combined question: include system prompt as explicit instruction at top
    # We will send system_prompt separately to answer_question so it can be integrated into LLM prompt.
    # answer_question currently signature: answer_question(question, chat_history=None, mode="strict", show_citations=True)
    # We'll pass system_prompt by prepending to the question for now (or update answer_question to accept it).
    system_part = ""
    if req.system_prompt:
        system_part = f"SYSTEM_INSTRUCTION:\n{req.system_prompt}\n\n"

    # Combine system instruction and user prompt (also pass chat_history separately)
    combined_question = system_part + req.user_prompt

    result = answer_question(combined_question, chat_history=req.chat_history or [], mode=req.mode, show_citations=req.show_citations)

    # Add session id if not provided
    if not req.session_id:
        sid = str(uuid.uuid4())
    else:
        sid = req.session_id

    return {
        "session_id": sid,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "chunks": result.get("chunks", []),
        "previews": result.get("previews", [])
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Simple upload endpoint: backend should write file to data/ then call ingest/embed/index pipeline.
    dest = Path("data") / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    # Optionally call your ingest/embed/index scripts programmatically here.
    # For now return success and let the user run embed/index steps in terminal or we can trigger them.
    return {"status": "ok", "filename": str(dest)}

@app.get("/")
async def root():
    return {"message": "RAG backend running. Visit /static/index.html for UI."}
