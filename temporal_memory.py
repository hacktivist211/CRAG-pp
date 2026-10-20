"""Temporal episodic memory for the IRIS self-corrective RAG pipeline.

Pragmatic implementation of the time-aware memory plan:

- PRIMARY memory: SQLite with WAL journaling next to the script. Every turn is
  written synchronously (sub-millisecond, zero VRAM, non-lossy) and doubles as
  the fallback store if the vector database is ever corrupted or unavailable.
- SECONDARY memory: a background PNG visual archive rendered on a daemon thread
  with Pillow when available. No OCR ever runs at inference time, so the backup
  path costs no VRAM and adds no latency to the answer loop.
- Soft forgetting: memory retrieval blends keyword overlap with an exponential
  recency decay (half-life in days) instead of degrading stored data.

A new session starts on every process start, or mid-run after
SESSION_IDLE_MINUTES of inactivity. Every tunable can be overridden with a
CRAG_* environment variable.
"""

import datetime as _dt
import json
import os
import re
import shutil
import sqlite3
import textwrap
import threading
import time
import uuid

# Store identity (see iris_config.py MEMORY MAP for the project-wide picture):
# this database is the CRAG stack's EPISODIC turn log - distinct from IRIS's
# durable personal memory (memory.db owned by memory_store.py). The file is
# now named temporal_memory.db to prevent confusion between the two stores;
# a legacy iris_memory/memory.db from an older run is migrated automatically
# (copied - the original is kept untouched as a backup).
MEMORY_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris_memory")
LEGACY_MEMORY_DB_PATH = os.path.join(MEMORY_DIRECTORY, "memory.db")
MEMORY_DB_PATH = os.getenv("IRIS_TEMPORAL_DB") or os.path.join(
    MEMORY_DIRECTORY, "temporal_memory.db"
)
VISUAL_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_memory")


def _migrate_legacy_db(db_path):
    """Copy a legacy iris_memory/memory.db to temporal_memory.db once.

    Non-destructive by design: the legacy file is never modified or deleted;
    if the new file already exists, nothing happens.
    """
    try:
        if db_path == LEGACY_MEMORY_DB_PATH:
            return
        if os.path.isfile(LEGACY_MEMORY_DB_PATH) and not os.path.isfile(db_path):
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            shutil.copy2(LEGACY_MEMORY_DB_PATH, db_path)
            print(
                "[memory] legacy episodic store copied to temporal_memory.db "
                "(original memory.db kept as backup)"
            )
    except Exception as error:
        print(f"[memory] legacy db migration skipped: {error}")


def _env_int(name, default):
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        value = float(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


SESSION_IDLE_MINUTES = _env_int("CRAG_SESSION_IDLE_MINUTES", 30)
RECENT_TURNS_CONTEXT = _env_int("CRAG_RECENT_TURNS", 2)
MEMORY_SEARCH_LIMIT = _env_int("CRAG_MEMORY_LIMIT", 5)
MEMORY_RANGE_LIMIT = _env_int("CRAG_MEMORY_RANGE_LIMIT", 12)
SCAN_ROW_LIMIT = _env_int("CRAG_MEMORY_SCAN", 4000)
RECENCY_HALF_LIFE_DAYS = _env_float("CRAG_RECENCY_HALF_LIFE_DAYS", 14.0)
MEMORY_MIN_SCORE = _env_float("CRAG_MEMORY_MIN_SCORE", 0.32)
VISUAL_ARCHIVE_ENABLED = (
    os.getenv("CRAG_VISUAL_ARCHIVE", "1").strip().lower() not in ("0", "false", "off", "no")
)

# End-of-session summarization: one compact LLM call turns a session's episodic
# turns into durable, extractive facts in memory.db. Disabled entirely with
# CRAG_SESSION_SUMMARY_ENABLED=0; short sessions are skipped by default.
SESSION_SUMMARY_ENABLED = (
    os.getenv("CRAG_SESSION_SUMMARY_ENABLED", "1").strip().lower()
    not in ("0", "false", "off", "no")
)
SESSION_SUMMARY_MIN_TURNS = _env_int("CRAG_SESSION_SUMMARY_MIN_TURNS", 5)
SESSION_SUMMARY_MAX_TOKENS = 600
SESSION_SUMMARY_TEMPERATURE = 0.2
SESSION_SUMMARY_TIMEOUT = 3.0
SESSION_SUMMARY_TRANSCRIPT_CHARS = 12000

SESSION_SUMMARY_SYSTEM_PROMPT = (
    "You extract durable facts from one conversation session for long-term memory. "
    "Return strict JSON only, exactly in this shape: "
    '{"facts": [{"topic": "short_topic", "fact": "one self-contained sentence", '
    '"importance": 0.8}], "narrative": "two or three sentence summary"} '
    "Rules: "
    "(1) Extractive discipline: every fact must quote or closely paraphrase the user's own words, "
    "or name an entity (product, place, number, date, decision) that appears verbatim in the "
    "transcript. Reject abstractive generalities. "
    "(2) One fact per row: do not merge independent facts. "
    "(3) Every fact must be self-contained and understandable without the rest of the session. "
    "(4) importance is a number in [0.1, 1.0]; decisions and preferences rank higher than "
    "observations. "
    "(5) Skip chit-chat, greetings, retries. Only durable details. "
    '(6) Empty is allowed: if the session contained only casual conversation, return '
    '{"facts": [], "narrative": ""}. Do not invent facts. '
    "(7) Never infer beyond the transcript: if the user said 'I have 6 GB of VRAM' but did not "
    "name the GPU model, do not add one. "
    "Answer immediately with the JSON object only, no markdown fences and no explanation."
)

SESSION_SUMMARY_TEXT_SYSTEM_PROMPT = (
    "You extract durable facts from one conversation session. "
    "Reply with plain text only, one fact per line, in exactly this pipe-delimited format: "
    "topic | fact | importance "
    "Use the same extractive rules: quote or closely paraphrase the user's own words, name only "
    "entities that appear verbatim in the transcript, keep one fact per line, skip chit-chat, "
    "and never invent details. If there is nothing durable, reply with the single word none."
)

TURN_SNIPPET_CHARS = 400
MEMORY_HIT_CHARS = 420

WEEKDAY_NAMES = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)
MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)

STOPWORDS = frozenset(
    """
    what when where which who whom whose why how
    did does do done was were is are am be been being
    have has had can could would should will shall may might must
    the an of to in on at for with about from by and or but so if then
    that this these those it its we us our ours me my mine myself
    you your yours he she they them their there here
    all any some more much many very really just also only
    tell show give let get got say said asked
    talk talked discuss discussed speak spoke chat chatted
    remember recall memory last previous earlier prior again back
    please thanks thank okay yes not no
    one two three four five six seven eight nine ten
    day days week weeks month months year years ago time times
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WEEKDAY_RE = re.compile(
    r"\b(?:last\s+|on\s+|this\s+)?(" + "|".join(WEEKDAY_NAMES) + r")\b", re.IGNORECASE
)
_MONTH_RE = re.compile(
    r"\b(?:in|of|from|during)\s+(" + "|".join(MONTH_NAMES) + r")\s*(\d{4})?\b", re.IGNORECASE
)
_AGO_RE = re.compile(r"\b(\d+)\s*(day|week|month|year)s?\s+ago\b", re.IGNORECASE)


def _now_local():
    return _dt.datetime.now()


def _time_of_day(moment):
    hour = moment.hour
    if hour < 5:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def humanize_delta(delta):
    """Turn a timedelta into a natural phrase such as '3 weeks and 2 days ago'."""
    try:
        seconds = int(delta.total_seconds())
    except (AttributeError, TypeError, ValueError):
        return "at an unknown time"
    if seconds < 0:
        seconds = 0
    if seconds < 45:
        return "moments ago" if seconds < 10 else f"{seconds} seconds ago"
    minutes = seconds // 60
    if minutes < 55:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours, minutes = divmod(minutes, 60)
    if hours < 23:
        base = f"{hours} hour{'s' if hours != 1 else ''}"
        if minutes >= 5:
            base += f" and {minutes} minutes"
        return base + " ago"
    days = hours // 24
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 30:
        weeks, extra = divmod(days, 7)
        base = f"{weeks} week{'s' if weeks != 1 else ''}"
        if extra:
            base += f" and {extra} day{'s' if extra != 1 else ''}"
        return base + " ago"
    if days < 335:
        months = days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def _tokenize(text):
    tokens = _TOKEN_RE.findall((text or "").lower())
    return [token for token in tokens if len(token) > 1 or token.isdigit()]


def _collapse(text):
    return " ".join((text or "").split())


def _parse_ts(value):
    if not value:
        return None
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _dt.datetime.strptime(str(value), pattern)
        except ValueError:
            continue
    try:
        return _dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None


class VisualArchive:
    """Secondary optical memory: renders each turn to PNG pages in the background.

    This is the pragmatic replacement for running a DeepSeek-OCR style backup at
    inference time: rendering happens off the critical path, nothing is OCR'd
    while answering, and the whole archive is plain files on disk.
    """

    WIDTH = 1120
    LINE_HEIGHT = 20
    MARGIN = 16
    WRAP_COLUMNS = 108
    PAGE_LINES = 64
    MAX_PAGES = 4
    FONT_CANDIDATES = (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    )

    def __init__(self, root, enabled=True):
        self.root = root
        self.enabled = bool(enabled)
        self._font = None
        if self.enabled:
            try:
                import PIL  # noqa: F401

                os.makedirs(self.root, exist_ok=True)
            except Exception:
                self.enabled = False

    def archive(self, content, session_id, turn_no, ts_local):
        """Render one turn to PNG page(s); safe to call from a daemon thread."""
        if not self.enabled or not content or not content.strip():
            return []
        try:
            from PIL import Image, ImageDraw, ImageFont

            font = self._load_font(ImageFont)
            lines = []
            for raw_line in content.splitlines():
                if not raw_line.strip():
                    lines.append("")
                    continue
                lines.extend(textwrap.wrap(raw_line, self.WRAP_COLUMNS) or [""])
            if not lines:
                return []
            pages = [lines[start:start + self.PAGE_LINES]
                     for start in range(0, len(lines), self.PAGE_LINES)]
            pages = pages[: self.MAX_PAGES]
            day_directory = os.path.join(
                self.root, f"{ts_local:%Y}", f"{ts_local:%m}", f"{ts_local:%d}"
            )
            os.makedirs(day_directory, exist_ok=True)
            written = []
            for page_number, page in enumerate(pages, start=1):
                height = self.MARGIN * 2 + self.LINE_HEIGHT * len(page)
                image = Image.new("RGB", (self.WIDTH, height), "white")
                draw = ImageDraw.Draw(image)
                y = self.MARGIN
                for line in page:
                    draw.text((self.MARGIN, y), line, fill="black", font=font)
                    y += self.LINE_HEIGHT
                name = (
                    f"{ts_local:%Y%m%d_%H%M%S}_{session_id}"
                    f"_turn{turn_no:03d}_p{page_number}.png"
                )
                path = os.path.join(day_directory, name)
                image.save(path, "PNG")
                written.append(path)
            return written
        except Exception:
            return []

    def _load_font(self, image_font_module):
        if self._font is not None:
            return self._font
        for candidate in self.FONT_CANDIDATES:
            try:
                if os.path.isfile(candidate):
                    self._font = image_font_module.truetype(candidate, 16)
                    return self._font
            except Exception:
                continue
        try:
            self._font = image_font_module.load_default()
        except Exception:
            self._font = None
        return self._font

    def count_files(self, limit=50000):
        total = 0
        if not os.path.isdir(self.root):
            return 0
        for _root, _directories, names in os.walk(self.root):
            total += sum(1 for name in names if name.lower().endswith(".png"))
            if total >= limit:
                break
        return total


class TemporalMemory:
    """Time-aware episodic memory backed by SQLite (primary) and PNG (secondary)."""

    def __init__(self, db_path=MEMORY_DB_PATH, visual=True):
        _migrate_legacy_db(db_path)
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self._archive = VisualArchive(VISUAL_DIRECTORY, enabled=visual and VISUAL_ARCHIVE_ENABLED)
        if visual and VISUAL_ARCHIVE_ENABLED and not self._archive.enabled:
            print("[memory] png visual archive disabled: install pillow to enable it (pip install pillow)")
        self._session_started = None
        self._session_turns = 0
        self._pending_turn = 0
        self._start_new_session(_now_local())
        self._last_activity = self._fetch_last_activity()

    # ------------------------------------------------------------------ schema

    def _create_schema(self):
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_no INTEGER NOT NULL,
                    ts_utc TEXT NOT NULL,
                    ts_local TEXT NOT NULL,
                    weekday TEXT,
                    time_of_day TEXT,
                    verdict TEXT,
                    query TEXT NOT NULL,
                    answer TEXT,
                    sources TEXT,
                    elapsed_seconds REAL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_ts_local ON turns(ts_local)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)")

    # ---------------------------------------------------------------- sessions

    def _start_new_session(self, now):
        stamp = now.strftime("%Y%m%d-%H%M%S")
        self.session_id = f"s-{stamp}-{uuid.uuid4().hex[:6]}"
        self._session_started = now
        self._session_turns = 0

    def _fetch_last_activity(self):
        try:
            row = self._conn.execute(
                "SELECT ts_local FROM turns ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        return _parse_ts(row["ts_local"])

    def begin_turn(self):
        """Start a new turn; rolls the session over after the idle threshold."""
        now = _now_local()
        if (
            self._last_activity is not None
            and (now - self._last_activity).total_seconds() > SESSION_IDLE_MINUTES * 60
        ):
            self._start_new_session(now)
        self._pending_turn = self._session_turns + 1
        return self._pending_turn

    # ------------------------------------------------------------------ writes

    def save_turn(self, query, answer, verdict=None, sources=None, turn_no=None, elapsed=None):
        """Dual-write: SQLite synchronously, PNG visual archive in the background."""
        now_local = _now_local()
        now_utc = _dt.datetime.now(_dt.timezone.utc)
        if turn_no is None:
            if self._pending_turn > self._session_turns:
                turn_no = self._pending_turn
            else:
                turn_no = self._session_turns + 1
        try:
            sources_json = json.dumps(sources or [], ensure_ascii=False)
        except (TypeError, ValueError):
            sources_json = "[]"
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    """
                    INSERT INTO turns (
                        session_id, turn_no, ts_utc, ts_local, weekday, time_of_day,
                        verdict, query, answer, sources, elapsed_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.session_id,
                        int(turn_no),
                        now_utc.isoformat(timespec="seconds"),
                        now_local.strftime("%Y-%m-%d %H:%M:%S"),
                        now_local.strftime("%A"),
                        _time_of_day(now_local),
                        verdict,
                        (query or "").strip(),
                        (answer or "").strip(),
                        sources_json,
                        elapsed,
                    ),
                )
                turn_id = cursor.lastrowid
        except Exception as error:
            print(f"[memory] could not store the turn: {error}")
            return None
        self._session_turns += 1
        self._last_activity = now_local
        if self._archive.enabled:
            content = f"User: {query}\n\nIRIS: {answer or ''}"
            try:
                threading.Thread(
                    target=self._archive.archive,
                    args=(content, self.session_id, turn_no, now_local),
                    daemon=True,
                    name=f"visual-archive-turn-{turn_no}",
                ).start()
            except Exception:
                pass
        return turn_id

    def forget(self, scope="last"):
        scope = "all" if str(scope).lower() == "all" else "last"
        deleted = 0
        try:
            with self._lock, self._conn:
                if scope == "last":
                    cursor = self._conn.execute(
                        "DELETE FROM turns WHERE id = (SELECT MAX(id) FROM turns)"
                    )
                else:
                    cursor = self._conn.execute("DELETE FROM turns")
                deleted = cursor.rowcount or 0
        except Exception as error:
            print(f"[memory] could not forget: {error}")
            return 0
        self._last_activity = self._fetch_last_activity()
        return deleted

    # ------------------------------------------------------------------- reads

    def _recent_rows(self, limit):
        try:
            cursor = self._conn.execute(
                "SELECT id, session_id, turn_no, ts_local, weekday, query, answer "
                "FROM turns ORDER BY id DESC LIMIT ?",
                (int(limit),),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        except Exception:
            return []
        for row in rows:
            row["parsed_ts"] = _parse_ts(row.get("ts_local"))
        return rows

    def _rows_between(self, start, end, limit=MEMORY_RANGE_LIMIT):
        try:
            cursor = self._conn.execute(
                "SELECT id, session_id, turn_no, ts_local, weekday, query, answer "
                "FROM turns WHERE ts_local >= ? AND ts_local <= ? ORDER BY ts_local",
                (f"{start:%Y-%m-%d} 00:00:00", f"{end:%Y-%m-%d} 23:59:59"),
            )
            rows = [dict(row) for row in cursor.fetchmany(int(limit))]
        except Exception:
            return []
        for row in rows:
            row["parsed_ts"] = _parse_ts(row.get("ts_local"))
        return rows

    def _mark_accessed(self, rows):
        if not rows:
            return
        now = _now_local().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._lock, self._conn:
                self._conn.executemany(
                    "UPDATE turns SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    [(now, row["id"]) for row in rows],
                )
        except Exception:
            pass

    def _row_to_hit(self, row, score=None):
        parsed = row.get("parsed_ts")
        hit = {
            "id": row.get("id"),
            "query": row.get("query") or "",
            "answer": _collapse(row.get("answer"))[:MEMORY_HIT_CHARS],
            "when": humanize_delta(_now_local() - parsed) if parsed else "at an unknown time",
            "stamp": parsed.strftime("%A, %B %d, %Y at %I:%M %p") if parsed else "unknown time",
        }
        if score is not None:
            hit["score"] = round(float(score), 3)
        return hit

    def last_interaction(self):
        try:
            row = self._conn.execute(
                "SELECT ts_local FROM turns ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        parsed = _parse_ts(row["ts_local"])
        if parsed is None:
            return None
        delta = _now_local() - parsed
        return {
            "seconds": max(0.0, delta.total_seconds()),
            "human": humanize_delta(delta),
            "stamp": parsed.strftime("%A, %B %d, %Y at %I:%M %p"),
            "timestamp": parsed,
        }

    def recent_turns(self, count=RECENT_TURNS_CONTEXT):
        rows = self._recent_rows(count)
        rows.reverse()
        return [self._row_to_hit(row) for row in rows]

    def search(self, query, limit=MEMORY_SEARCH_LIMIT, min_score=MEMORY_MIN_SCORE):
        """Keyword overlap fused with exponential recency decay (soft forgetting)."""
        terms = [token for token in _tokenize(query) if token not in STOPWORDS]
        if not terms:
            return []
        rows = self._recent_rows(SCAN_ROW_LIMIT)
        if not rows:
            return []
        now = _now_local()
        term_set = set(terms)
        scored = []
        for row in rows:
            query_tokens = set(_tokenize(row.get("query")))
            answer_tokens = set(_tokenize(row.get("answer")))
            matched_query = len(term_set & query_tokens)
            matched_answer = len(term_set & answer_tokens)
            if matched_query + matched_answer == 0:
                continue
            overlap = min(1.0, (matched_query * 1.6 + matched_answer) / (len(terms) * 1.6))
            parsed = row.get("parsed_ts")
            if parsed is not None:
                age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
                recency = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
            else:
                recency = 0.0
            score = 0.72 * overlap + 0.28 * recency
            if score >= min_score:
                scored.append((score, row))
        if not scored:
            return []
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[: int(limit)]
        self._mark_accessed([row for _score, row in top])
        return [self._row_to_hit(row, score=score) for score, row in top]

    def recall(self, query):
        """Recall past interactions for a memory-directed query.

        Date expressions (yesterday, last tuesday, 3 weeks ago, in january) narrow
        the window first; anything without a usable window, or an empty window,
        falls back to the keyword + recency search.
        """
        window = self._parse_when(query)
        if window is not None:
            rows = self._rows_between(window[0], window[1])
            if rows:
                self._mark_accessed(rows)
                return [self._row_to_hit(row) for row in rows]
        return self.search(query, limit=max(MEMORY_SEARCH_LIMIT, 8))

    def _parse_when(self, query):
        """Extract a (start_date, end_date) window from natural language."""
        text = " ".join((query or "").lower().split())
        if not text:
            return None
        today = _now_local().date()

        if re.search(r"\btoday\b", text):
            return today, today
        if re.search(r"\byesterday\b", text):
            return today - _dt.timedelta(days=1), today - _dt.timedelta(days=1)
        if re.search(r"\b(?:this|current)\s+week\b", text):
            monday = today - _dt.timedelta(days=today.weekday())
            return monday, today
        if re.search(r"\blast\s+week\b", text):
            this_monday = today - _dt.timedelta(days=today.weekday())
            last_monday = this_monday - _dt.timedelta(days=7)
            return last_monday, last_monday + _dt.timedelta(days=6)
        if re.search(r"\bthis\s+month\b", text):
            return today.replace(day=1), today
        if re.search(r"\blast\s+month\b", text):
            first_this_month = today.replace(day=1)
            end = first_this_month - _dt.timedelta(days=1)
            return end.replace(day=1), end

        match = _AGO_RE.search(text)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
            days = value * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
            target = today - _dt.timedelta(days=days)
            return target, target

        match = _WEEKDAY_RE.search(text)
        if match:
            target_index = WEEKDAY_NAMES.index(match.group(1).lower())
            back = (today.weekday() - target_index) % 7
            if back == 0:
                back = 7
            target = today - _dt.timedelta(days=back)
            return target, target

        match = _MONTH_RE.search(text)
        if match:
            month = MONTH_NAMES.index(match.group(1).lower()) + 1
            year = int(match.group(2)) if match.group(2) else None
            if year is None:
                year = today.year if month <= today.month else today.year - 1
            start = _dt.date(year, month, 1)
            if month == 12:
                end = _dt.date(year, 12, 31)
            else:
                end = _dt.date(year, month + 1, 1) - _dt.timedelta(days=1)
            return start, end
        return None

    # --------------------------------------------------------------- formatting

    def format_recent(self, turns):
        lines = []
        for turn in turns or []:
            when = turn.get("when") or "earlier"
            lines.append(f"- [{when}] User: {turn.get('query', '')}")
            answer = turn.get("answer")
            if answer:
                lines.append(f"  IRIS: {answer}")
        return "\n".join(lines) if lines else ""

    def format_hits(self, hits):
        blocks = []
        for hit in hits or []:
            lines = [
                f"PAST INTERACTION - {hit.get('stamp', 'unknown time')} ({hit.get('when', '')})",
                f"User asked: {hit.get('query', '')}",
            ]
            answer = hit.get("answer")
            if answer:
                lines.append(f"IRIS answered: {answer}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) if blocks else ""

    def build_temporal_context(self):
        """The temporal injector: current time plus elapsed-time awareness."""
        now = _now_local()
        lines = [f"CURRENT TIME: {now:%A, %B %d, %Y at %I:%M %p} ({_time_of_day(now)})"]
        last = self.last_interaction()
        if last is None:
            lines.append("LAST INTERACTION: none recorded; this is the first interaction with this user")
        elif last["seconds"] < 120:
            lines.append(f"LAST INTERACTION: {last['human']} (the conversation is ongoing)")
        else:
            lines.append(
                f"LAST INTERACTION: {last['human']} ({last['stamp']}); a new session starts now"
            )
        started = self._session_started or now
        lines.append(
            f"SESSION: started {started:%I:%M %p}, "
            f"{self._session_turns} recorded interaction(s) before this one"
        )
        return "\n".join(lines)

    def stats(self):
        turns = 0
        sessions = 0
        oldest = None
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total, COUNT(DISTINCT session_id) AS sessions, "
                "MIN(ts_local) AS oldest FROM turns"
            ).fetchone()
            if row:
                turns = int(row["total"] or 0)
                sessions = int(row["sessions"] or 0)
                oldest = row["oldest"]
        except Exception:
            pass
        db_bytes = 0
        try:
            db_bytes = os.path.getsize(self.db_path)
        except OSError:
            pass
        visual_files = self._archive.count_files() if self._archive else 0
        return {
            "turns": turns,
            "sessions": sessions,
            "oldest": oldest,
            "db_bytes": db_bytes,
            "visual_files": visual_files,
        }


# ------------------------------------------------------- session summarization


def _summary_session_rows(temporal, session_id):
    """Read this session's turns directly from the episodic log (never mutate it)."""
    try:
        cursor = temporal._conn.execute(
            "SELECT turn_no, ts_local, query, answer FROM turns "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []


def _summary_transcript(rows, max_chars=SESSION_SUMMARY_TRANSCRIPT_CHARS):
    """Compact, bounded transcript for the summarizer prompt (never printed)."""
    lines = []
    used = 0
    for row in rows:
        query = _collapse(row.get("query"))[:800]
        answer = _collapse(row.get("answer"))[:600]
        turn_no = row.get("turn_no")
        block = f"[{turn_no}] User: {query}"
        if answer:
            block += f"\n    IRIS: {answer}"
        if used + len(block) + 1 > max_chars:
            break
        lines.append(block)
        used += len(block) + 1
    return "\n".join(lines)


def _summary_already_done(memory, session_id):
    """Idempotency check against memory.db, with no LLM involvement."""
    try:
        row = memory.db.execute(
            "SELECT 1 FROM memories WHERE kind = ? AND tags LIKE ? LIMIT 1",
            ("session_summary_done", f"%{session_id}%"),
        ).fetchone()
        return bool(row)
    except Exception:
        return False


def _summary_safe_add(memory, **kwargs):
    """memory.add() that degrades to an embedding-free write and never raises."""
    try:
        return memory.add(**kwargs)
    except Exception:
        if kwargs.get("embed") is False:
            return None
        retry = dict(kwargs)
        retry["embed"] = False
        try:
            return memory.add(**retry)
        except Exception:
            return None


def _summary_mark_done(memory, session_id):
    return _summary_safe_add(
        memory,
        content=f"Session {session_id} summarized.",
        speaker="user",
        source="session_auto",
        kind="session_summary_done",
        importance=0.1,
        tags=f"session_id:{session_id}",
        embed=False,
    )


def _summary_parse_json(raw):
    """Parse the strict JSON response, tolerating code fences around it."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _summary_importance(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return max(0.1, min(1.0, number))


def _summary_normalize(data):
    """Return ([(topic, fact, importance)], narrative) from a JSON object."""
    facts = []
    entries = data.get("facts") if isinstance(data, dict) else None
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            content = entry.get("fact") or entry.get("content") or ""
            if not isinstance(content, str):
                continue
            content = _collapse(content)
            if not content:
                continue
            topic = entry.get("topic")
            topic = _collapse(topic) if isinstance(topic, str) else ""
            facts.append((topic, content, _summary_importance(entry.get("importance"))))
    narrative = data.get("narrative") if isinstance(data, dict) else ""
    narrative = _collapse(narrative) if isinstance(narrative, str) else ""
    return facts, narrative


def _summary_parse_lines(raw):
    """Parse the plain-text fallback. Returns facts, or None if malformed."""
    text = (raw or "").strip()
    if not text:
        return None
    facts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-* ").strip()
        if not line:
            continue
        if line.lower() == "none":
            continue
        if "|" not in line:
            return None
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2 or not parts[1]:
            return None
        topic = parts[0]
        importance = parts[2] if len(parts) > 2 else "0.5"
        facts.append((topic, _collapse(parts[1]), _summary_importance(importance)))
    return facts


def _summary_control_call(messages, json_mode, num_predict, deadline):
    """Call the shared LLM transport in a daemon thread with a wall-clock cap.

    Returns (raw_text, error). A timed-out call leaves its daemon thread behind
    but never blocks the caller (shutdown path) beyond the deadline.
    """
    import evaluator_router

    call_kwargs = {"json_mode": json_mode, "num_predict": num_predict}
    # The shared transport predates this feature and pins temperature 0.0.
    # Forward the requested sampling temperature only when a transport accepts
    # it; never modify evaluator_router itself.
    try:
        import inspect

        if "temperature" in inspect.signature(
            evaluator_router._control_complete
        ).parameters:
            call_kwargs["temperature"] = SESSION_SUMMARY_TEMPERATURE
    except (TypeError, ValueError):
        pass

    outcome = {}

    def worker():
        try:
            outcome["raw"] = evaluator_router._control_complete(
                messages, **call_kwargs
            )
        except Exception as error:  # never let a transport error escape the thread
            outcome["error"] = error

    thread = threading.Thread(
        target=worker, daemon=True, name="session-summary-llm"
    )
    thread.start()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "", TimeoutError("session summary deadline already elapsed")
    thread.join(remaining)
    if thread.is_alive():
        return "", TimeoutError(
            f"LLM call exceeded the {max(0.0, remaining):.1f}s remaining cap"
        )
    error = outcome.get("error")
    if error is not None:
        return "", error
    return outcome.get("raw") or "", None


def summarize_session(temporal, memory, min_turns=SESSION_SUMMARY_MIN_TURNS,
                      force=False, timeout=SESSION_SUMMARY_TIMEOUT):
    """Summarize one session into durable facts + one narrative row.

    Extractive by contract: the LLM is instructed to quote the user's own words
    or named entities, but this function only persists what it returns. Facts go
    to memory.db with source="session_auto", kind="session_summary"; the narrative
    uses kind="session_narrative". A session_summary_done bookkeeping row makes
    the pass idempotent per session_id. Every failure degrades to a status dict;
    this function never raises into main.py.
    """
    started = time.monotonic()
    session_id = str(getattr(temporal, "session_id", "") or "")

    def finish(status, facts_saved=0, narrative_saved=False, error=None):
        result = {
            "status": status,
            "facts_saved": int(facts_saved),
            "narrative_saved": bool(narrative_saved),
            "session_id": session_id,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "error": error,
        }
        if status != "ok":
            print(f"[crag] session summary: {status} ({error or 'no reason'})")
        return result

    if not SESSION_SUMMARY_ENABLED:
        return finish("skipped", error="disabled by CRAG_SESSION_SUMMARY_ENABLED")
    if temporal is None or not session_id:
        return finish("no_session", error="no active session")

    try:
        minimum = max(1, int(min_turns))
    except (TypeError, ValueError):
        minimum = SESSION_SUMMARY_MIN_TURNS

    rows = _summary_session_rows(temporal, session_id)
    if not rows:
        return finish("no_session", error="no turns in the current session")
    if len(rows) < minimum:
        return finish(
            "skipped", error=f"{len(rows)} turn(s) < min_turns={minimum}"
        )
    if not force and memory is not None and _summary_already_done(memory, session_id):
        return finish("already_summarized", error=f"session {session_id} already summarized")

    transcript = _summary_transcript(rows)
    if not transcript:
        return finish("skipped", error="empty transcript")

    try:
        cap = max(0.1, float(timeout))
    except (TypeError, ValueError):
        cap = SESSION_SUMMARY_TIMEOUT
    deadline = started + cap

    messages = [
        ("system", SESSION_SUMMARY_SYSTEM_PROMPT),
        (
            "user",
            "Transcript of the current session:\n\n"
            f"{transcript}\n\n"
            "Extract the durable facts and the narrative. JSON only.",
        ),
    ]
    try:
        raw, error = _summary_control_call(
            messages, json_mode=True, num_predict=SESSION_SUMMARY_MAX_TOKENS,
            deadline=deadline,
        )
    except Exception as failure:
        return finish("llm_failed", error=f"{type(failure).__name__}: {failure}")
    if error is not None:
        return finish(
            "llm_failed", error=f"{type(error).__name__}: {error}"
        )
    if not (raw or "").strip():
        return finish("llm_failed", error="empty response from LLM")

    data = _summary_parse_json(raw)
    if data is None:
        # One text fallback, same deadline: ask for plain pipe-delimited lines.
        fallback_messages = [
            ("system", SESSION_SUMMARY_TEXT_SYSTEM_PROMPT),
            (
                "user",
                "Transcript of the current session:\n\n"
                f"{transcript}\n\n"
                "Return one durable fact per line as: topic | fact | importance",
            ),
        ]
        try:
            raw_two, error_two = _summary_control_call(
                fallback_messages, json_mode=False,
                num_predict=SESSION_SUMMARY_MAX_TOKENS, deadline=deadline,
            )
        except Exception as failure:
            return finish(
                "llm_failed",
                error=f"text fallback failed: {type(failure).__name__}: {failure}",
            )
        if error_two is not None:
            return finish(
                "llm_failed",
                error=f"text fallback failed: {type(error_two).__name__}: {error_two}",
            )
        fallback_facts = _summary_parse_lines(raw_two)
        if fallback_facts is None:
            return finish(
                "llm_failed",
                error="LLM did not return valid JSON or plain fact lines",
            )
        facts, narrative = fallback_facts, ""
    else:
        facts, narrative = _summary_normalize(data)

    saved = 0
    for _topic, content, importance in facts:
        row_id = _summary_safe_add(
            memory,
            content=content,
            speaker="user",
            source="session_auto",
            kind="session_summary",
            importance=importance,
        )
        if row_id is not None:
            saved += 1

    narrative_saved = False
    if narrative:
        row_id = _summary_safe_add(
            memory,
            content=narrative,
            speaker="user",
            source="session_auto",
            kind="session_narrative",
            importance=0.6,
        )
        narrative_saved = row_id is not None

    if memory is not None:
        _summary_mark_done(memory, session_id)

    return finish("ok", facts_saved=saved, narrative_saved=narrative_saved)
