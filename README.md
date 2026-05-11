# agentic-rag

Retrieval-augmented Q&A where **Claude drives the retrieval**: it decides whether a question needs the docs,
writes its own search queries, re-searches when results are weak, and answers with **passage-level citations**.

```
question ─▶ Claude (tool runner) ──search_docs(query)──▶ SQLite FTS5 / BM25
                 ▲                                            │
                 └──────────── passages + ids ◀───────────────┘
            answer with [file#chunk] citations
```

## Why this design
- **Agentic, not fixed top-k:** the model reformulates queries and stops when it has enough evidence.
- **Zero infra:** SQLite FTS5 (BM25) is built into Python — no vector DB needed to start. The `Store`
  interface (`add` / `search`) is the seam for swapping in pgvector or hybrid search.
- **Grounded answers:** the API returns only sources the answer actually cites.
- **Safe input:** user text is quoted before reaching FTS, so it can't inject query syntax.

## Run

```bash
uv sync
export ANTHROPIC_API_KEY=...
DOCS_DIR=docs uv run uvicorn agentic_rag.api:app --reload
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"When can I deploy?"}'
```

| Endpoint | |
|---|---|
| `POST /ingest` `{source, text}` | add or replace a document |
| `GET /search?q=` | raw BM25 search |
| `POST /ask` `{question}` | agentic answer + cited sources |

## Evaluate retrieval

```bash
uv run python -m agentic_rag.eval docs eval/questions.jsonl   # hit@1/3/5, no API calls
```

## Test

```bash
uv run pytest -q    # Claude is faked in tests; no key needed
```

## Stack
Python · Anthropic SDK (tool runner, `claude-opus-5-5`) · FastAPI · SQLite FTS5 · Docker · GitHub Actions
