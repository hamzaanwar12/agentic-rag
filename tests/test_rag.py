from types import SimpleNamespace as NS
from fastapi.testclient import TestClient
from agentic_rag import agent, store as store_mod
from agentic_rag.store import Store, chunk


def test_chunk_packs_paragraphs():
    text = "\n\n".join(["a" * 500] * 5)
    parts = chunk(text, size=1200)
    assert len(parts) == 3 and all(len(p) <= 1200 for p in parts)
    assert chunk("") == []


def test_search_ranks_relevant_chunk_first():
    s = Store()
    s.add("a.md", "Cats sleep a lot.\n\nDeploys happen on Tuesdays.")
    s.add("b.md", "Expenses are due in 30 days.")
    hits = s.search("when do deploys happen?")
    assert hits[0]["id"] == "a.md#0" and "Tuesdays" in hits[0]["text"]


def test_search_ignores_fts_syntax():
    s = Store()
    s.add("a.md", "hello world")
    assert s.search('hello" OR NEAR(') and s.search("***") == []


def test_reingest_replaces_chunks():
    s = Store()
    s.add("a.md", "old text")
    s.add("a.md", "new text")
    assert [h["text"] for h in s.search("text")] == ["new text"]


class FakeRunner:
    """Simulates Claude: calls search_docs once, then answers citing the top hit."""
    def __init__(self, **kw):
        self.kw = kw

    def until_done(self):
        import json
        hits = json.loads(self.kw["tools"][0].call({"query": "deploy days"}))
        top = hits[0]["id"]
        return NS(stop_reason="end_turn", content=[NS(type="text", text=f"Tuesdays and Thursdays [{top}].")])


def fake_client():
    return NS(beta=NS(messages=NS(tool_runner=lambda **kw: FakeRunner(**kw))))


def test_agent_answer_cites_sources():
    s = Store()
    s.ingest_dir("docs")
    out = agent.answer("When can I deploy?", s, client=fake_client())
    assert out["sources"] == ["onboarding.md#0"] and out["searches"] >= 1


def test_api_ingest_search_ask(monkeypatch):
    from agentic_rag import api
    monkeypatch.setattr(api, "store", Store())
    monkeypatch.setattr(api.agent, "answer", lambda q, s: {"answer": "ok", "sources": [], "searches": 0})
    monkeypatch.setenv("RAG_API_TOKEN", "t0ken")
    c = TestClient(api.app, headers={"Authorization": "Bearer t0ken"})
    assert c.post("/ingest", json={"source": "x.md", "text": "pgvector vs fts5"}).json()["chunks"] == 1
    assert c.get("/search", params={"q": "fts5"}).json()[0]["id"] == "api:x.md#0"
    assert c.post("/ask", json={"question": "hi"}).json()["answer"] == "ok"
    assert c.post("/ask", json={"question": ""}).status_code == 422


def test_api_requires_token(monkeypatch):
    from agentic_rag import api
    c = TestClient(api.app)
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    assert c.get("/search", params={"q": "x"}).status_code == 401  # no token configured -> closed
    monkeypatch.setenv("RAG_API_TOKEN", "t0ken")
    assert c.get("/search", params={"q": "x"}, headers={"Authorization": "Bearer nope"}).status_code == 401
