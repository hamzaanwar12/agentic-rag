"""Retrieval eval: hit@k over a JSONL of {"question", "source"} pairs. No API calls.

    uv run python -m agentic_rag.eval docs eval/questions.jsonl
"""
import json
import sys
from .store import Store


def hit_rate(store: Store, cases: list[dict], k: int = 3) -> float:
    hits = sum(any(h["id"].startswith(c["source"]) for h in store.search(c["question"], k)) for c in cases)
    return hits / len(cases)


if __name__ == "__main__":
    s = Store()
    s.ingest_dir(sys.argv[1])
    cases = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
    for k in (1, 3, 5):
        print(f"hit@{k}: {hit_rate(s, cases, k):.2%}  ({len(cases)} questions)")
