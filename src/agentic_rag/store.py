"""Document store: paragraph chunking + SQLite FTS5 (BM25) search. No external services."""
import re
import sqlite3
from pathlib import Path

CHUNK_CHARS = 1200


def chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Greedy paragraph packing; a paragraph longer than `size` becomes its own chunk."""
    out, cur = [], ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    return out + [cur] if cur else out


class Store:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("create virtual table if not exists chunks using fts5(source, idx unindexed, body)")

    def add(self, source: str, text: str) -> int:
        self.db.execute("delete from chunks where source = ?", (source,))
        parts = chunk(text)
        self.db.executemany("insert into chunks values (?, ?, ?)", [(source, i, c) for i, c in enumerate(parts)])
        self.db.commit()
        return len(parts)

    def ingest_dir(self, folder: str) -> dict[str, int]:
        return {str(f): self.add(str(f), f.read_text(errors="ignore"))
                for f in sorted(Path(folder).rglob("*")) if f.suffix in {".md", ".txt"}}

    def search(self, query: str, k: int = 5) -> list[dict]:
        terms = " OR ".join(f'"{w}"' for w in re.findall(r"\w+", query.lower()))  # quoted: user text can't inject FTS syntax
        if not terms:
            return []
        rows = self.db.execute(
            "select source, idx, body, bm25(chunks) from chunks where chunks match ? order by bm25(chunks) limit ?",
            (terms, k)).fetchall()
        return [{"id": f"{s}#{i}", "text": b, "score": round(-r, 3)} for s, i, b, r in rows]
