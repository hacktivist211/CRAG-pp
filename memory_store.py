#!/usr/bin/env python3
"""Long-term memory for IRIS.

Storage: a single flat SQLite file (memory.db) beside this module. No server,
no extra VRAM. Embeddings come from your llama-server
(start it with `--embeddings`), or optionally from a local
sentence-transformers model on CPU (EMBED_BACKEND=sbert in .env) - useful
for comparing retrieval quality with bench_retrieval.py. Rows are
deduplicated by content hash.

Embedding dimensions are validated: the first successful embed fixes the
expected dimension; stored vectors of a different dimension are skipped
(rather than crashing the search) with a one-time warning, so switching
embedding models degrades to keyword search instead of breaking.

Retrieval is hybrid:
  - dense  : cosine similarity on embeddings (llama-server /v1/embeddings)
  - keyword: SQLite FTS5 match (BM25-ish), with a LIKE fallback
  - priors : importance (set at import or by memory_add) + recency decay

The store degrades gracefully: if llama-server embeddings are unavailable it
still stores memories (embedding NULL) and falls back to keyword search; run
`python memory_store.py backfill` once the server is up to embed everything.

Tools exposed to the LLM (via tool_runtime):
  memory_search(query, k)  -> top memories as JSON
  memory_add(content)      -> save something the user asks to remember

CLI:
  python memory_store.py stats
  python memory_store.py search "what do you know about my setup" -k 5
  python memory_store.py backfill
  python memory_store.py add "text" --speaker user --source manual
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import requests

import iris_config as cfg

# Shared lazy sentence-transformers model for the optional sbert backend.
_SBERT = None

# ------------------------------------------------------------------ tool schemas
MEMORY_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Search the user's long-term memory (facts, preferences, past conversations "
            "imported from ChatGPT, Claude, Gemini, and IRIS itself). Use this before "
            "answering personal questions or when the user refers to something discussed before."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in the user's words."},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Top results to return. Default 4."},
            },
            "required": ["query"],
        },
    },
}

MEMORY_ADD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "memory_add",
        "description": (
            "Save a lasting memory about the user. Use when the user says remember, note, "
            "or states a durable fact, preference, or instruction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact to remember, one plain sentence."},
            },
            "required": ["content"],
        },
    },
}

FTS_QUERY_CHARS = set(" \t-|\"'()[]{}*:")


@dataclass
class Hit:
    id: int
    content: str
    score: float
    source: str = ""
    speaker: str = ""
    kind: str = ""
    created_at: float | None = None
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "score": round(float(self.score), 4),
            "source": self.source,
            "speaker": self.speaker,
            "kind": self.kind,
            "date": cfg.human_date(self.created_at),
        }


class MemoryStore:
    def __init__(self, db_path: str | None = None, embed: bool = True,
                 embed_backend: str | None = None):
        self.db_path = str(db_path or cfg.MEMORY_DB)
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self.embed_enabled = embed
        self.embed_backend = (embed_backend or cfg.EMBED_BACKEND).lower()
        self._embed_warned = False
        self._dim: int | None = None
        self._dim_warned = False
        self._cache: dict[int, np.ndarray] = {}
        self._cache_loaded = False

    # -------------------------------------------------------------- schema
    def _create_schema(self):
        cur = self.db
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                speaker TEXT,
                source TEXT,
                kind TEXT,
                conv_title TEXT,
                turn_index INTEGER,
                created_at REAL,
                importance REAL DEFAULT 0.5,
                tags TEXT,
                content_hash TEXT UNIQUE,
                embedding BLOB
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON memories(source)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kind ON memories(kind)")
        self.fts = True
        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content)")
        except sqlite3.OperationalError:
            self.fts = False
        cur.commit()

    # -------------------------------------------------------------- embeddings
    def _embed_batch(self, texts: list[str]) -> np.ndarray | None:
        """Embed texts via the configured backend. Returns float32 (n, d) or None.

        Backends: 'llamaserver' (default, the resident Qwen model via
        /v1/embeddings) or 'sbert' (local sentence-transformers on CPU,
        EMBED_SBERT_MODEL, used for quality comparison). The result is
        L2-normalized so dot product == cosine similarity.
        """
        if not self.embed_enabled or not texts:
            return None
        try:
            if self.embed_backend == "sbert":
                mat = self._embed_sbert(texts)
            else:
                mat = self._embed_llamaserver(texts)
            if mat is None:
                return None
            mat = np.asarray(mat, dtype=np.float32).reshape(len(texts), -1)
            dim = int(mat.shape[1])
            if self._dim is None:
                self._dim = dim
            elif dim != self._dim:
                raise RuntimeError(
                    f"embedding dimension changed (expected {self._dim}, got {dim}); "
                    "re-run backfill or switch EMBED_BACKEND consistently"
                )
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms
        except Exception as e:
            if not self._embed_warned:
                print(
                    f"[memory] embeddings unavailable ({e}); storing without vectors, "
                    "keyword search still works. Run 'python memory_store.py backfill' later.",
                    file=sys.stderr,
                )
                self._embed_warned = True
            return None

    def _embed_llamaserver(self, texts: list[str]) -> np.ndarray | None:
        """Embed via llama-server /v1/embeddings (the resident Qwen model)."""
        r = requests.post(
            cfg.EMBED_ENDPOINT,
            json={"input": texts, "model": cfg.EMBED_MODEL},
            timeout=15,
        )
        if r.status_code == 404:
            raise RuntimeError("endpoint 404 (start llama-server with --embeddings)")
        r.raise_for_status()
        data = r.json().get("data", [])
        if len(data) != len(texts):
            raise RuntimeError("embedding count mismatch")
        return np.asarray([d["embedding"] for d in data], dtype=np.float32)

    def _embed_sbert(self, texts: list[str]) -> np.ndarray | None:
        """Embed with a local sentence-transformers model on CPU (optional).

        Chosen via EMBED_BACKEND=sbert. The default model
        (BAAI/bge-small-en-v1.5, ~130 MB) fits comfortably in system RAM
        and never touches the GPU, so it cannot starve the LLM or TTS.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "EMBED_BACKEND=sbert needs sentence-transformers "
                "(pip install sentence-transformers)"
            ) from None
        global _SBERT
        if _SBERT is None:
            _SBERT = SentenceTransformer(cfg.EMBED_SBERT_MODEL, device="cpu")
        return np.asarray(_SBERT.encode(texts, convert_to_numpy=True), dtype=np.float32)

    def _hash(self, content: str) -> str:
        norm = " ".join(str(content).lower().split())
        return hashlib.sha1(norm.encode("utf-8", "ignore")).hexdigest()

    # -------------------------------------------------------------- writes
    def add(
        self,
        content: str,
        speaker: str = "user",
        source: str = "manual",
        kind: str = "fact",
        conv_title: str | None = None,
        turn_index: int | None = None,
        created_at: float | None = None,
        importance: float = 0.5,
        tags: str = "",
        embed: bool | None = None,
    ) -> int | None:
        """Add one memory. Returns row id, or None if duplicate. Never raises."""
        content = str(content).strip()
        if not (1 <= len(content) <= 8000):
            return None
        h = self._hash(content)
        emb_blob = None
        if (self.embed_enabled if embed is None else embed):
            vec = self._embed_batch([content])
            if vec is not None:
                emb_blob = vec[0].tobytes()
        try:
            cur = self.db.execute(
                """INSERT OR IGNORE INTO memories
                   (content, speaker, source, kind, conv_title, turn_index,
                    created_at, importance, tags, content_hash, embedding)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    content,
                    speaker,
                    source,
                    kind,
                    conv_title,
                    turn_index,
                    created_at if created_at is not None else time.time(),
                    max(0.0, min(1.0, float(importance))),
                    tags,
                    h,
                    emb_blob,
                ),
            )
            self.db.commit()
            if cur.rowcount == 0:
                return None
            rid = cur.lastrowid
            if self.fts:
                try:
                    self.db.execute("INSERT INTO memories_fts(rowid, content) VALUES (?,?)", (rid, content))
                    self.db.commit()
                except sqlite3.OperationalError:
                    self.fts = False
            if emb_blob is not None:
                self._load_cache(force=True)
            return rid
        except sqlite3.IntegrityError:
            return None

    def add_many(self, rows: list[dict], embed: bool | None = None) -> tuple[int, int]:
        """Batch add. Each row: dict of add() kwargs minus 'embed'. Returns (added, skipped)."""
        rows = [r for r in rows if r.get("content") and 1 <= len(str(r["content"]).strip()) <= 8000]
        # Embed all at once (much faster than per-row HTTP calls).
        emb_blobs: list[bytes | None] = [None] * len(rows)
        if self.embed_enabled if embed is None else embed:
            vecs = self._embed_batch([str(r["content"]).strip() for r in rows])
            if vecs is not None:
                emb_blobs = [v.tobytes() for v in vecs]
        added = skipped = 0
        try:
            for r, blob in zip(rows, emb_blobs):
                content = str(r["content"]).strip()
                h = self._hash(content)
                cur = self.db.execute(
                    """INSERT OR IGNORE INTO memories
                       (content, speaker, source, kind, conv_title, turn_index,
                        created_at, importance, tags, content_hash, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        content,
                        r.get("speaker", "user"),
                        r.get("source", "import"),
                        r.get("kind", "chat_turn"),
                        r.get("conv_title"),
                        r.get("turn_index"),
                        r.get("created_at") if r.get("created_at") is not None else time.time(),
                        max(0.0, min(1.0, float(r.get("importance", 0.5)))),
                        r.get("tags", ""),
                        h,
                        blob,
                    ),
                )
                if cur.rowcount == 0:
                    skipped += 1
                    continue
                rid = cur.lastrowid
                added += 1
                if self.fts:
                    try:
                        self.db.execute(
                            "INSERT INTO memories_fts(rowid, content) VALUES (?,?)", (rid, content)
                        )
                    except sqlite3.OperationalError:
                        self.fts = False
            self.db.commit()
        except sqlite3.Error as e:
            self.db.rollback()
            print(f"[memory] batch insert failed, transaction rolled back: {e}", file=sys.stderr)
        if added:
            self._load_cache(force=True)
        return added, skipped

    def backfill_embeddings(self, limit: int | None = None, batch: int = cfg.EMBED_BATCH) -> int:
        """Embed rows stored without vectors (e.g., imported while server was down)."""
        sql = "SELECT id, content FROM memories WHERE embedding IS NULL"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.db.execute(sql).fetchall()
        if not rows:
            return 0
        done = 0
        for i in range(0, len(rows), batch):
            chunk = rows[i : i + batch]
            vecs = self._embed_batch([c for _, c in chunk])
            if vecs is None:
                print("[memory] embeddings still unavailable; stopping backfill", file=sys.stderr)
                break
            for (rid, _), v in zip(chunk, vecs):
                self.db.execute("UPDATE memories SET embedding=? WHERE id=?", (v.tobytes(), rid))
                done += 1
            self.db.commit()
        self._load_cache(force=True)
        return done

    # -------------------------------------------------------------- search
    def _load_cache(self, force: bool = False) -> None:
        if self._cache_loaded and not force:
            return
        self._cache = {}
        for rid, blob in self.db.execute("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"):
            v = np.frombuffer(blob, dtype=np.float32)
            n = np.linalg.norm(v)
            self._cache[rid] = v / n if n > 0 else v
        self._cache_loaded = True

    def _fts_ids(self, query: str) -> set[int]:
        if not self.fts:
            return set()
        toks = [t for t in "".join(ch if ch not in FTS_QUERY_CHARS else " " for ch in query.lower()).split() if len(t) >= 2]
        if not toks:
            return set()
        fts_q = " OR ".join(f'"{t}"*' for t in toks[:8])
        try:
            return {r[0] for r in self.db.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?", (fts_q,))}
        except sqlite3.OperationalError:
            return set()

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """Hybrid search. Order: 0.75*cos + 0.15*importance + 0.10*recency, +0.15 if keyword hit."""
        query = str(query or "").strip()
        if not query:
            return []
        k = max(1, min(int(k), 12))
        self._load_cache()
        qvec = None
        _qv = self._embed_batch([query])
        if _qv is not None:
            qvec = _qv[0]
        fts_hits = self._fts_ids(query)
        rows = self.db.execute(
            "SELECT id, content, speaker, source, kind, created_at, importance FROM memories"
        ).fetchall()
        scored: list[Hit] = []
        now = time.time()
        dim_skipped = 0
        for rid, content, speaker, source, kind, created_at, importance in rows:
            cos = 0.0
            if qvec is not None and rid in self._cache:
                vec = self._cache[rid]
                if vec.shape[0] != qvec.shape[0]:
                    dim_skipped += 1  # stored with a different embedding model
                    cos = 0.0
                else:
                    cos = float(np.dot(qvec, vec))
            score = 0.75 * cos + 0.15 * float(importance or 0.5)
            if created_at:
                days = max(0.0, (now - created_at) / 86400.0)
                score += 0.10 * (1.0 / (1.0 + days / 90.0))
            else:
                score += 0.05
            if rid in fts_hits:
                score += 0.15
            if cos == 0.0 and rid not in fts_hits:
                continue  # no signal at all
            scored.append(
                Hit(
                    id=rid,
                    content=content,
                    score=score,
                    source=source or "",
                    speaker=speaker or "",
                    kind=kind or "",
                    created_at=created_at,
                )
            )
        if dim_skipped and not self._dim_warned:
            self._dim_warned = True
            print(
                f"[memory] {dim_skipped} stored vector(s) have a different dimension "
                "than the current embedding backend; they are skipped for cosine "
                "ranking (keyword search still covers them). Run "
                "'python memory_store.py backfill' after switching backends.",
                file=sys.stderr,
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def as_context(self, query: str, k: int = 4, max_words: int = 110) -> str:
        """Memory block for the system prompt. Compact, grounded, not spoken verbatim."""
        hits = self.search(query, k=k)
        if not hits:
            return ""
        lines = []
        for h in hits:
            tag = f"{h.source or 'memory'}" + (f", {cfg.human_date(h.created_at)}" if h.created_at else "")
            text = cfg.speech_clean(h.content, max_len=240)
            lines.append(f"- ({tag}) {text}")
        block = "RELEVANT MEMORY CONTEXT (grounding facts; weave in naturally, do not read aloud verbatim):\n" + "\n".join(lines)
        words = block.split()
        if len(words) > max_words + 20:
            block = " ".join(words[: max_words + 20])
        return block

    # -------------------------------------------------------------- tools
    def run_tool(self, name: str, args: dict) -> dict:
        if name == "memory_search":
            try:
                hits = self.search(str((args or {}).get("query", "")), k=(args or {}).get("k", 4))
                return {"ok": True, "memories": [h.row() for h in hits]}
            except Exception as e:
                return {"ok": False, "error": f"memory search failed: {e}"}
        if name == "memory_add":
            content = str((args or {}).get("content", "")).strip()
            if not content:
                return {"ok": False, "error": "content is required"}
            rid = self.add(
                content=content, speaker="user", source="live", kind="fact",
                importance=0.85, created_at=time.time(),
            )
            if rid is None:
                return {"ok": True, "note": "already known, not duplicated"}
            return {"ok": True, "id": rid, "note": "saved"}
        return {"ok": False, "error": f"unknown tool {name}"}

    # -------------------------------------------------------------- misc
    def stats(self) -> dict:
        n, = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()
        emb, = self.db.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL").fetchone()
        by_source = dict(self.db.execute("SELECT source, COUNT(*) FROM memories GROUP BY source").fetchall())
        return {
            "db_path": self.db_path,
            "total": n,
            "embedded": emb,
            "pending_embeddings": n - emb,
            "fts": self.fts,
            "by_source": by_source,
        }

    def close(self):
        try:
            self.db.commit()
            self.db.close()
        except Exception:
            pass


# ------------------------------------------------------------------ shared instance
_SHARED: MemoryStore | None = None


def get_store() -> MemoryStore:
    global _SHARED
    if _SHARED is None:
        _SHARED = MemoryStore()
    return _SHARED


def tool_schemas() -> list[dict]:
    return [MEMORY_SEARCH_SCHEMA, MEMORY_ADD_SCHEMA]


if __name__ == "__main__":
    args = sys.argv[1:]
    store = MemoryStore()
    if not args or args[0] == "stats":
        print(json.dumps(store.stats(), indent=2))
    elif args[0] == "search":
        q = " ".join(args[1:])
        k = 4
        if "-k" in args:
            i = args.index("-k")
            k = int(args[i + 1]) if len(args) > i + 1 else 4
            q = " ".join([a for j, a in enumerate(args[1:]) if j + 1 not in (i, i + 1)])
        for h in store.search(q, k=k):
            print(f"[{h.score:.3f}] ({h.source}|{h.speaker}|{cfg.human_date(h.created_at)}) {h.content[:140]}")
    elif args[0] == "backfill":
        n = store.backfill_embeddings()
        print(f"embedded {n} memories")
    elif args[0] == "add":
        i = args.index("--speaker") if "--speaker" in args else -1
        j = args.index("--source") if "--source" in args else -1
        text = " ".join(a for idx, a in enumerate(args[1:], 1) if idx not in (i, i + 1, j, j + 1))
        rid = store.add(text, speaker=args[i + 1] if i > 0 else "user", source=args[j + 1] if j > 0 else "manual")
        print("saved id" if rid else "duplicate or empty", rid or "")
    else:
        print(__doc__)
    store.close()
