#!/usr/bin/env python3
"""Import memories from your other models into IRIS.

Feed it any export file; the format is auto-detected:
  - ChatGPT  : conversations.json from the ChatGPT export ZIP (mapping tree)
  - Claude   : conversations.json from the Claude data export (chat_messages)
  - Gemini   : Takeout MyActivity.json / prompt-response JSON
  - Generic  : JSON or JSONL of {prompt, response}, {role, content},
               or {messages: [...]} conversation objects
  - Text     : any .txt / .md / .chat file, including IRIS chat.txt dumps and
               pasted transcripts with User:/Assistant: markers

Everything is stored "correctly and properly":
  - deduplicated by content hash (safe to re-run the same file twice)
  - one row per turn, with source, speaker, conversation title, timestamps
  - one compact index row per conversation for quick topical recall
  - importance scored by simple heuristics (facts about you rank higher)
  - embeddings computed in batches when llama-server is reachable; otherwise
    rows are stored and you run `python memory_store.py backfill` later

CLI:
  python memory_import.py chat.txt
  python memory_import.py ~/Downloads/conversations.json
  python memory_import.py file1.json file2.txt --dry-run
  python memory_import.py weird.json --source generic
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import memory_store
from memory_store import MemoryStore

# ------------------------------------------------------------------ turn model
@dataclass
class Turn:
    speaker: str          # 'user' | 'assistant' | 'import'
    text: str
    ts: float | None = None
    conv_title: str = ""
    idx: int = 0
    kind: str = "chat_turn"


_IMPORTANCE_CUES = (
    "my name", "i am", "i'm", "i live", "i work", "i use", "i prefer", "i like",
    "i love", "i hate", "i need", "remember", "always", "never", "my birthday",
    "my phone", "my address", "my wife", "my husband", "my son", "my daughter",
    "my dog", "my cat", "my project", "my setup", "my gpu", "my favorite", "my goal",
)


def score_importance(text: str, speaker: str) -> float:
    """Heuristic importance in [0.1, 1.0]. Facts about the user rank highest."""
    t = text.lower()
    s = 0.55 if speaker == "user" else 0.35
    if any(cue in t for cue in _IMPORTANCE_CUES):
        s += 0.25
    ln = len(text)
    s += 0.15 if 40 <= ln <= 500 else (0.05 if ln > 500 else 0.0)
    if speaker == "user" and ln < 8:
        s -= 0.25
    return max(0.1, min(1.0, s))


# ------------------------------------------------------------------ parsers
def parse_chatgpt(items: list) -> list[Turn]:
    turns: list[Turn] = []
    for conv in items:
        if not isinstance(conv, dict):
            continue
        title = str(conv.get("title") or "Untitled").strip() or "Untitled"
        rows = []
        for node in (conv.get("mapping") or {}).values():
            if not isinstance(node, dict):
                continue
            msg = node.get("message") or {}
            role = (msg.get("author") or {}).get("role")
            if role not in ("user", "assistant"):
                continue  # skip system / tool / function messages
            content = msg.get("content") or {}
            parts = content.get("parts") or []
            text = " ".join(p.strip() for p in parts if isinstance(p, str)).strip()
            if not text:
                continue
            ct = msg.get("create_time") or conv.get("create_time")
            try:
                ct = float(ct) if ct else None
            except (TypeError, ValueError):
                ct = None
            rows.append((role, text, ct))
        if any(r[2] for r in rows):
            rows.sort(key=lambda r: (r[2] is None, r[2] or 0.0))
        for i, (role, text, ct) in enumerate(rows):
            turns.append(Turn(role, text, ct, title, i))
    return turns


def _parse_iso(ts) -> float | None:
    if not ts:
        return None
    s = str(ts).strip().replace("Z", "+00:00")
    try:
        import datetime
        return datetime.datetime.fromisoformat(s).timestamp()
    except ValueError:
        pass
    for fmt in ("%b %d, %Y, %I:%M %p", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            import datetime
            return datetime.datetime.strptime(str(ts).strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def parse_claude(items: list) -> list[Turn]:
    turns: list[Turn] = []
    for conv in items:
        if not isinstance(conv, dict):
            continue
        title = str(conv.get("name") or conv.get("title") or "Untitled").strip() or "Untitled"
        for i, m in enumerate(conv.get("chat_messages") or []):
            if not isinstance(m, dict):
                continue
            sender = str(m.get("sender") or "").lower()
            if sender in ("human", "user"):
                role = "user"
            elif sender in ("assistant", "ai"):
                role = "assistant"
            else:
                continue
            text = str(m.get("text") or "").strip()
            if not text:
                continue
            turns.append(Turn(role, text, _parse_iso(m.get("created_at")), title, i))
    return turns


_GEMINI_SPLIT = re.compile(r"(?i)here'?s what gemini said|gemini(?:'?s)? (?:reply|response)|assistant\s*:")


def parse_gemini(items: list) -> list[Turn]:
    turns: list[Turn] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        ts = _parse_iso(it.get("time") or it.get("created_at"))
        user, assistant = "", ""
        if it.get("prompt") or it.get("response"):
            user = str(it.get("prompt") or "").strip()
            assistant = str(it.get("response") or "").strip()
        else:
            body = str(it.get("body") or it.get("text") or "").strip()
            if body:
                pieces = _GEMINI_SPLIT.split(body, maxsplit=1)
                first = pieces[0]
                second = pieces[1].strip() if len(pieces) > 1 else ""
                first = re.sub(r"(?i)^you (asked|said|typed)\s*:?\s*", "", first).strip()
                if second:
                    user, assistant = first, second
                elif first:
                    user = first
        if not user and title:
            m = re.match(r"(?i)^asked gemini\s*(?:to|about)?\s*(.*)$", title)
            if m and m.group(1):
                user = m.group(1).strip()
        if user:
            turns.append(Turn("user", user, ts, title or "Gemini", 0))
        if assistant:
            turns.append(Turn("assistant", assistant, ts, title or "Gemini", 1))
    return turns


def parse_generic(items: list) -> list[Turn]:
    """JSON array/JSONL of pairs, messages, or conversation wrappers."""
    turns: list[Turn] = []
    if isinstance(items, dict):
        items = items.get("conversations") or items.get("messages") or [items]
    for it in items:
        if not isinstance(it, dict):
            continue
        if "prompt" in it or "response" in it:
            p, r = str(it.get("prompt") or "").strip(), str(it.get("response") or "").strip()
            if p:
                turns.append(Turn("user", p, _parse_iso(it.get("created_at")), "", 0))
            if r:
                turns.append(Turn("assistant", r, _parse_iso(it.get("created_at")), "", 1))
        elif "role" in it and ("content" in it or "text" in it):
            role = str(it.get("role") or "").lower()
            if role not in ("user", "assistant"):
                continue
            text = str(it.get("content") or it.get("text") or "").strip()
            if text:
                turns.append(Turn(role, text, _parse_iso(it.get("created_at")), "", len(turns)))
        elif isinstance(it.get("messages"), list):
            title = str(it.get("title") or it.get("name") or "")
            sub = parse_generic(it["messages"])
            for t in sub:
                t.conv_title = title
            turns.extend(sub)
    return turns


_PLAIN_MARKERS = re.compile(
    r"(?im)^\s*(?:user|you|human|me|neeraj)\s*[:>]\s*(.*)$"
)
_PLAIN_AI_MARKERS = re.compile(
    r"(?im)^\s*(?:assistant|ai|chatgpt|claude|gemini|copilot|iris|ultron)\s*[:>]\s*(.*)$"
)


def parse_plain_text(text: str) -> list[Turn]:
    """Plain text transcripts and notes.

    Handles User:/Assistant: marked transcripts AND standalone note paragraphs
    mixed between them. Continuation lines join the current speaker block.
    """
    blocks: list[tuple[str | None, list[str]]] = []  # (speaker | None-for-note, lines)
    cur: tuple[str | None, list[str]] = (None, [])
    for line in text.splitlines():
        um = _PLAIN_MARKERS.match(line)
        am = _PLAIN_AI_MARKERS.match(line)
        if um or am:
            if cur[1]:
                blocks.append(cur)
            cur = ("user" if um else "assistant", [(um or am).group(1)])
        elif line.strip():
            cur = (cur[0], cur[1] + [line.strip()])
        else:  # blank line ends the current block
            if cur[1]:
                blocks.append(cur)
            cur = (None, [])
    if cur[1]:
        blocks.append(cur)

    turns: list[Turn] = []
    for speaker, buf in blocks:
        joined = " ".join(b for b in buf if b).strip()
        if not joined:
            continue
        if speaker:
            turns.append(Turn(speaker, joined, None, "", len(turns)))
        elif 20 <= len(joined) <= 4000:
            turns.append(Turn("import", joined, None, "notes", len(turns), kind="note"))
    return turns


# ------------------------------------------------------------------ sniffing
def _classify_items(items) -> str:
    for it in items[:20]:
        if isinstance(it, dict):
            if "mapping" in it:
                return "chatgpt"
            if "chat_messages" in it:
                return "claude"
            if "title" in it and ("body" in it or "time" in it or "titleUrl" in it) and "chat_messages" not in it:
                return "gemini"
    return "generic"


def _jsonl_items(raw: str) -> list | None:
    """Parse newline-delimited JSON. Returns list of dicts or None."""
    items = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            return None
        items.append(obj)
    return items or None


def sniff_format(path: Path, forced: str = "auto") -> str:
    if forced != "auto":
        return forced
    if path.name.lower().endswith((".txt", ".md", ".chat", ".log")):
        return "text"
    raw = path.read_text(encoding="utf-8", errors="ignore")[:2_000_000]
    try:
        data = json.loads(raw)
    except Exception:
        items = _jsonl_items(raw)
        if items is not None and all(isinstance(x, dict) for x in items[:5]):
            return _classify_items(items)
        return "text"
    items = _as_items(data)
    if items is not None:
        return _classify_items(items)
    return "generic" if isinstance(data, dict) else "text"


def _as_items(data):
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data[:5]):
        return data
    if isinstance(data, dict):
        if "conversations" in data and isinstance(data["conversations"], list):
            return data["conversations"]
        if "activity" in data and isinstance(data["activity"], list):
            return data["activity"]
    return None


def load_items(path: Path):
    """Return (fmt, payload). payload is the raw string for text format."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.name.lower().endswith((".txt", ".md", ".chat", ".log")):
        return "text", raw
    try:
        data = json.loads(raw)
        items = _as_items(data)
        if items is not None:
            return _classify_items(items), items
        if isinstance(data, dict):
            return "generic", [data]
        return "text", raw
    except Exception:
        pass
    items = _jsonl_items(raw)
    if items is not None and all(isinstance(x, dict) for x in items[:5]):
        return _classify_items(items), items
    return "text", raw


PARSERS = {
    "chatgpt": parse_chatgpt,
    "claude": parse_claude,
    "gemini": parse_gemini,
    "generic": parse_generic,
}


def parse_file(path: Path, forced: str = "auto") -> tuple[str, list[Turn]]:
    fmt, payload = load_items(path)
    if forced != "auto" and forced in PARSERS:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(raw)
            payload = _as_items(data) or ([data] if isinstance(data, dict) else data)
            if not isinstance(payload, list):
                payload = [payload]
            fmt = forced
        except Exception:
            pass  # forced format on a non-JSON file: keep text parsing
    if fmt == "text":
        turns = parse_plain_text(str(payload))
    else:
        turns = PARSERS[fmt](payload)
    return fmt, _clean(turns)


def _clean(turns: list[Turn]) -> list[Turn]:
    seen, out = set(), []
    for t in turns:
        t.text = " ".join(t.text.split())
        if not (1 <= len(t.text) <= 4000):
            continue
        if t.text.lower() in ("<|start|>", "<|end|>", "image", "[image]"):
            continue
        key = (t.speaker, t.text.lower()[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# ------------------------------------------------------------------ import
def build_rows(turns: list[Turn], source: str) -> list[dict]:
    rows = []
    for t in turns:
        rows.append(
            {
                "content": t.text,
                "speaker": t.speaker,
                "source": source,
                "kind": t.kind,
                "conv_title": t.conv_title or None,
                "turn_index": t.idx,
                "created_at": t.ts,
                "importance": score_importance(t.text, t.speaker),
            }
        )
    return rows


def build_conv_summaries(turns: list[Turn], source: str, cap: int = 400) -> list[dict]:
    """One compact index row per conversation for topical recall."""
    by_title: dict[str, list[Turn]] = {}
    for t in turns:
        key = t.conv_title or ""
        by_title.setdefault(key, []).append(t)
    rows = []
    for title, ts in list(by_title.items())[:cap]:
        if not title:
            continue
        first_user = next((x.text[:150] for x in ts if x.speaker == "user"), "")
        first_ai = next((x.text[:150] for x in ts if x.speaker == "assistant"), "")
        content = f"Imported {source} conversation titled {title}. The user said: {first_user} The reply was: {first_ai}"
        if len(content) < 60:
            continue
        ts_min = min((x.ts for x in ts if x.ts), default=None)
        rows.append(
            {
                "content": content,
                "speaker": "import",
                "source": source,
                "kind": "conv_summary",
                "conv_title": title,
                "turn_index": None,
                "created_at": ts_min,
                "importance": 0.4,
            }
        )
    return rows


def import_file(path: Path, store: MemoryStore, forced: str = "auto", dry_run: bool = False) -> dict:
    fmt, turns = parse_file(path, forced)
    source = {"chatgpt": "chatgpt", "claude": "claude", "gemini": "gemini"}.get(fmt, "import")
    stats = {"file": str(path), "format": fmt, "parsed": len(turns), "stored": 0, "duplicates": 0}
    n_user = sum(1 for t in turns if t.speaker == "user")
    n_ai = sum(1 for t in turns if t.speaker == "assistant")
    stats["user_turns"], stats["assistant_turns"] = n_user, n_ai
    if dry_run:
        print(f"  [dry-run] {path.name}: format={fmt}, {len(turns)} turns ({n_user} user / {n_ai} assistant)")
        for t in turns[:6]:
            print(f"    {t.speaker:9s} | {t.text[:90]}")
        return stats
    rows = build_rows(turns, source) + build_conv_summaries(turns, source)
    added, skipped = store.add_many(rows)
    stats["stored"], stats["duplicates"] = added, skipped
    print(
        f"  {path.name}: format={fmt}, parsed {len(turns)} turns "
        f"({n_user} user / {n_ai} assistant), stored {added}, skipped {skipped} duplicates"
    )
    return stats


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    forced, dry = "auto", False
    files: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry-run":
            dry = True
        elif a == "--source":
            i += 1
            forced = argv[i] if i < len(argv) else "auto"
        elif a == "--db":
            i += 1
            globals()["_DB_OVERRIDE"] = argv[i] if i < len(argv) else None
        else:
            files.append(a)
        i += 1

    db = globals().get("_DB_OVERRIDE")
    store = MemoryStore(db_path=db) if db else memory_store.get_store()
    t0 = time.time()
    total = {"stored": 0, "duplicates": 0, "parsed": 0}
    for f in files:
        p = Path(f).expanduser()
        if not p.exists():
            print(f"  SKIP {f}: not found")
            continue
        try:
            s = import_file(p, store, forced, dry)
            total["stored"] += s["stored"]
            total["duplicates"] += s["duplicates"]
            total["parsed"] += s["parsed"]
        except Exception as e:
            print(f"  ERROR {f}: {type(e).__name__}: {e}")
    if not dry:
        st = store.stats()
        pending = st["pending_embeddings"]
        print(
            f"\ndone in {time.time() - t0:.1f}s: parsed {total['parsed']}, stored {total['stored']}, "
            f"skipped {total['duplicates']} duplicates. database now holds {st['total']} memories."
        )
        if pending:
            print(
                f"{pending} memories have no embeddings yet (llama-server down?). "
                "Start llama-server with --embeddings, then run: python memory_store.py backfill"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
