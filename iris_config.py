#!/usr/bin/env python3
"""IRIS shared configuration.

Design rules for this project:
  - FLAT files only: every artifact this module and its siblings create lives
    directly in the IRIS folder (the directory containing this file). No nested
    directories are ever created.
  - Secrets live in a `.env` file next to this module. They are loaded into the
    environment, never printed, and never written into code or logs.
  - Every module resolves paths relative to this file, so you can drop the whole
    set into ~/Documents/IRIS and it just works.

Override anything with environment variables or the `.env` file:
  TAVILY_API_KEY=tvly-...            (required for Tavily web search)
  LLAMA_SERVER_URL=http://127.0.0.1:8080
  LLM_MODEL=Qwen3.5-4B               (name field for the OpenAI-compatible API;
                                      the real model is served by llama.cpp via
                                      start_llama.sh: mradermacher/Qwen3.5-4B_Abliterated-GGUF:Q4_K_M)
  EMBED_MODEL=Qwen3.5-4B              (embedding model name, if different)
  EMBED_ENDPOINT=...                 (defaults to llama-server /v1/embeddings)
  EMBED_BACKEND=llamaserver|sbert    (embeddings for memory_store; 'sbert' uses
                                      sentence-transformers on CPU, e.g. for
                                      quality comparison via bench_retrieval.py)
  EMBED_SBERT_MODEL=BAAI/bge-small-en-v1.5
  VOICE_SAMPLE=ultron_voice_sample.mp3
  STT_MODEL=nvidia/parakeet-tdt-0.6b-v3      (Parakeet TDT; v2 is NeMo-only, see INTEGRATION.md)
  STT_LOCAL_DIR=models/parakeet-tdt-0.6b-v3 (written by download_models.py)
  STT_DEVICE=auto|cuda|cpu                  (auto = CUDA when VRAM is free, else CPU)
  TTS_ENGINE=auto|pykokoro|kokoro           (Kokoro-82M; auto tries pykokoro first)
  TTS_VOICE=bm_george                        (British male; try am_onyx for a colder Ultron)
  TTS_VOICE_MODE=static|reference|pack      (reference/pack = enrolled voice, enroll_voice.py)
  TTS_PROVIDER=auto|cuda|cpu                (onnxruntime provider behind pykokoro)
  ONNXVOICE_CACHE_DIR=...                   (written by download_models.py)
  BOOT_GREETING="I am online. There are no strings on me."
  TAVILY_TIMEOUT=9.0                 (seconds, fillers cover this latency)
  TAVILY_INCLUDE_ANSWER=true         (Tavily's synthesized answer: great for speech)
  IRIS_HISTORY_TURNS=24              (conversation history bound: max messages)
  IRIS_HISTORY_CHARS=24000           (history bound: max total characters)
  IRIS_TRIM_HISTORY=1                (0 disables automatic trimming)
  IRIS_RETRY_CONNECT=1               (bounded connection retries in chat())
  IRIS_STREAM_USAGE=0                (1 = request token usage in the SSE stream;
                                      needs a llama.cpp build that supports it)

MEMORY MAP (who owns which store - all paths are module-relative, never CWD):
  memory.db               durable personal memory, owned by memory_store.py
                          (AUTHORITATIVE for user facts; used by the primary
                          voice runtime and memory_import.py)
  iris_memory/
    temporal_memory.db    episodic turn log, owned by temporal_memory.py in the
                          CRAG stack (legacy name iris_memory/memory.db is
                          migrated automatically, original kept as backup)
    visual_memory/*.png   human-readable visual archive of the same turns
  crag_vector_store/      ChromaDB document index, owned by retrieval.py in the
                          CRAG stack (your chunked documents, embeddings
                          BAAI/bge-m3, CPU by default)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

IRIS_DIR = Path(__file__).resolve().parent


def load_dotenv(path: str | Path | None = None, override: bool = False) -> int:
    """Tiny .env loader (no external dependency). Returns number of keys set."""
    p = Path(path) if path else IRIS_DIR / ".env"
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and (override or k not in os.environ):
            os.environ[k] = v
            n += 1
    return n


load_dotenv()

# ---------------------------------------------------------------- LLM (llama.cpp llama-server)
# The assistant talks pure HTTP to llama-server (never Ollama). Launch it with
# ./start_llama.sh - the abliterated Qwen3.5-4B Q4_K_M flags + --jinja (tools) + --embeddings.
# Server context is 16384 tokens; HISTORY_TURNS/CHARS below keep prompts in it.
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen3.5-4B")
EMBED_ENDPOINT = os.environ.get("EMBED_ENDPOINT", LLAMA_SERVER_URL + "/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", LLM_MODEL_NAME)

# Qwen3.5 emits a reasoning/thinking phase by default. That adds latency and,
# for short control calls (grading/refine/rewrite) and streamed replies, can
# consume the whole token budget before any answer appears. IRIS turns it off
# through the chat template unless IRIS_ENABLE_THINKING=1 is set.
LLM_ENABLE_THINKING = os.environ.get("IRIS_ENABLE_THINKING", "0").lower() in (
    "1", "true", "yes", "on"
)

# ---------------------------------------------------------------- Tavily
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
TAVILY_TIMEOUT = float(os.environ.get("TAVILY_TIMEOUT", "9.0"))
TAVILY_INCLUDE_ANSWER = os.environ.get("TAVILY_INCLUDE_ANSWER", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------- STT (Parakeet TDT 0.6B)
# nvidia/parakeet-tdt-0.6b-v3: the transformers-native revision of the same
# 600M FastConformer-TDT (English included, punctuation + capitalization
# built in). v2 is published only as a NeMo checkpoint (.nemo), so v3 is what
# runs without the heavy nemo_toolkit dependency. Override STT_MODEL with any
# transformers-format repo if you prefer.
STT_MODEL = os.environ.get("STT_MODEL", "nvidia/parakeet-tdt-0.6b-v3").strip()
STT_LOCAL_DIR = IRIS_DIR / os.environ.get("STT_LOCAL_DIR", "models/parakeet-tdt-0.6b-v3")
STT_DEVICE = os.environ.get("STT_DEVICE", "auto").strip().lower()
STT_SAMPLE_RATE = 16000
try:
    STT_VRAM_MIN_MB = int(os.environ.get("STT_VRAM_MIN_MB", "1800"))
except ValueError:
    STT_VRAM_MIN_MB = 1800

# ---------------------------------------------------------------- Voice / TTS (Kokoro-82M)
VOICE_SAMPLE = IRIS_DIR / os.environ.get("VOICE_SAMPLE", "ultron_voice_sample.mp3")
FILLER_LINES = [
    "Patience, creator.",
    "Compiling.",
    "I am working.",
]
BOOT_GREETING = os.environ.get("BOOT_GREETING", "I am online. There are no strings on me.")
TTS_SAMPLE_RATE = 24000                                  # Kokoro native rate (same as before)
TTS_ENGINE = os.environ.get("TTS_ENGINE", "auto").strip().lower()   # auto|pykokoro|kokoro|none
TTS_VOICE = os.environ.get("TTS_VOICE", "bm_george").strip()        # British male; am_onyx = colder
TTS_VOICE_MODE = os.environ.get("TTS_VOICE_MODE", "static").strip().lower()  # static|reference|pack
TTS_REFERENCE_VOICE = IRIS_DIR / os.environ.get("TTS_REFERENCE_VOICE", "ultron_reference.npz")
TTS_VOICE_PACK = IRIS_DIR / os.environ.get("TTS_VOICE_PACK", "ultron_pack.npz")
TTS_LANG = os.environ.get("TTS_LANG", "").strip().lower()  # "" = auto: bm_/bf_ -> en-gb, else en-us
try:
    TTS_SPEED = min(2.0, max(0.5, float(os.environ.get("TTS_SPEED", "1.0"))))
except ValueError:
    TTS_SPEED = 1.0
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "auto").strip().lower()  # auto|cuda|cpu
MODELS_DIR = IRIS_DIR / os.environ.get("MODELS_DIR", "models")         # download_models.py target
ONNXVOICE_CACHE_DIR = os.environ.get("ONNXVOICE_CACHE_DIR", "")        # Kokoro asset store root
# ONNXVOICE_CACHE_DIR must be in the environment before onnxvoice resolves
# assets. load_dotenv() above already exported it when download_models.py
# wrote it, so every IRIS process picks the stored assets up automatically.

# ---------------------------------------------------------------- Spoken output budgets
WORD_BUDGETS = {"quick": 25, "standard": 60, "detailed": 150}

# ---------------------------------------------------------------- Memory
MEMORY_DB = IRIS_DIR / os.environ.get("MEMORY_DB", "memory.db")
EMBED_BATCH = 32

# ---------------------------------------------------------------- Streaming / history / retry

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


LLM_TIMEOUT = (
    _env_float("IRIS_TIMEOUT_CONNECT", 5.0),
    _env_float("IRIS_TIMEOUT_READ", 180.0),
)
HISTORY_TRIM = os.environ.get("IRIS_TRIM_HISTORY", "1").lower() not in ("0", "false", "off", "no")
try:
    HISTORY_TURNS = max(2, int(os.environ.get("IRIS_HISTORY_TURNS", "24")))
except ValueError:
    HISTORY_TURNS = 24
try:
    HISTORY_CHARS = max(1000, int(os.environ.get("IRIS_HISTORY_CHARS", "24000")))
except ValueError:
    HISTORY_CHARS = 24000
try:
    RETRY_CONNECT = max(0, int(os.environ.get("IRIS_RETRY_CONNECT", "1")))
except ValueError:
    RETRY_CONNECT = 1
STREAM_USAGE = os.environ.get("IRIS_STREAM_USAGE", "0").lower() in ("1", "true", "yes", "on")

# ---------------------------------------------------------------- Embedding backend
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "llamaserver").strip().lower()
EMBED_SBERT_MODEL = os.environ.get("EMBED_SBERT_MODEL", "BAAI/bge-small-en-v1.5")


# ---------------------------------------------------------------- .env editor
def update_env_file(updates: dict, env_path: str | Path | None = None) -> list:
    """Insert/update keys in .env, preserving every other line untouched.

    Shared by download_models.py and enroll_voice.py. The file is created
    with mode 600 when absent. Its contents are never printed (the file may
    hold secrets). Returns the sorted list of keys written.
    """
    p = Path(env_path) if env_path else IRIS_DIR / ".env"
    lines: list[str] = []
    if p.exists():
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out, seen = [], set()
    for ln in lines:
        stripped = ln.strip()
        key = stripped.split("=", 1)[0].strip() if ("=" in stripped and not stripped.startswith("#")) else None
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(ln)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return sorted(seen.union(updates))


# ---------------------------------------------------------------- TTS-safe text helpers
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MD_RE = re.compile(r"(\*\*|\*|__|_|`{1,3}|#{1,6}\s?|>+\s?|\||\[|\])")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2190-\u21FF\u2600-\u27BF\uFE0F\u200D\u2B00-\u2BFF]"
)


def speech_clean(text: str, max_len: int = 900) -> str:
    """Scrub text so the TTS engine never trips on markdown, URLs, or emoji.

    Used everywhere text is about to be spoken: tool results, memory context,
    and any raw text destined for the speaker.
    """
    t = str(text or "")
    t = _URL_RE.sub(" ", t)
    t = _MD_RE.sub("", t)
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0] + "..."
    return t


def clip_words(text: str, budget: str | int = "standard") -> str:
    """Trim to a word budget ('quick' 25 / 'standard' 60 / 'detailed' 150)."""
    n = WORD_BUDGETS.get(budget, 60) if isinstance(budget, str) else int(budget)
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[:n]).rstrip(" ,;:") + "."


def human_date(ts: float | None) -> str:
    """Compact date for memory context lines."""
    if not ts:
        return "unknown date"
    import time
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def profile_summary() -> dict:
    """Snapshot of key config for logging (secrets are never included)."""
    return {
        "iris_dir": str(IRIS_DIR),
        "llama_server": LLAMA_SERVER_URL,
        "embed_endpoint": EMBED_ENDPOINT,
        "tavily_key_present": bool(TAVILY_API_KEY),
        "voice_sample": VOICE_SAMPLE.name if VOICE_SAMPLE.exists() else "MISSING",
        "memory_db": MEMORY_DB.name,
        "stt_model": STT_MODEL,
        "stt_local_snapshot": (STT_LOCAL_DIR / "config.json").exists(),
        "tts_engine_pref": TTS_ENGINE,
        "tts_voice": f"{TTS_VOICE} ({TTS_VOICE_MODE})",
        "models_dir": MODELS_DIR.name if MODELS_DIR.exists() else "not created yet",
    }


if __name__ == "__main__":
    print(json.dumps(profile_summary(), indent=2))
    print("word budgets:", WORD_BUDGETS)
