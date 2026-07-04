"""HTTP API: POST /ingest (text), POST /ask, GET /search."""
import hmac
import os
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from . import agent
from .store import Store

def require_token(authorization: str = Header("")):
    token = os.getenv("RAG_API_TOKEN", "")
    if not token or not hmac.compare_digest(authorization.removeprefix("Bearer ").strip(), token):
        raise HTTPException(401, "missing or invalid token (set RAG_API_TOKEN)")


app = FastAPI(title="agentic-rag", dependencies=[Depends(require_token)])
store = Store(os.getenv("RAG_DB", "rag.db"))
if os.getenv("DOCS_DIR"):
    store.ingest_dir(os.environ["DOCS_DIR"])


class Doc(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2_000_000)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


@app.post("/ingest")
def ingest(doc: Doc):
    source = f"api:{doc.source}"  # API docs live in their own namespace; can't overwrite DOCS_DIR files
    return {"source": source, "chunks": store.add(source, doc.text)}


@app.get("/search")
def search(q: str, k: int = 5):
    return store.search(q, max(1, min(k, 20)))


@app.post("/ask")
def ask(q: Question):
    try:
        return agent.answer(q.question, store)
    except agent.anthropic.APIStatusError as e:
        raise HTTPException(502, f"LLM error: {e.status_code}")
