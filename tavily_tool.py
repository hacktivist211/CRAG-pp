#!/usr/bin/env python3
"""Tavily web search for IRIS (live web data for the assistant).

Integration approach (per Tavily build-path guidance):
  - Uses the official `tavily-python` SDK when installed, and falls back to a
    direct REST call so the module has zero hard dependencies.
  - Reads TAVILY_API_KEY from the environment / .env (via iris_config).
    The key is never printed, logged, or embedded in error messages.
  - If no key is configured it automatically uses Tavily keyless access
    (X-Tavily-Access-Mode: keyless). Keyless is rate-limited but proves the
    pipeline works and is fine as an emergency fallback.

Tool surface for llama-server function calling:
  TOOL_SCHEMA is the OpenAI-style tool definition. `run_tool(args)` executes a
  search and returns a compact JSON payload for the LLM to speak from.

CLI:
  python tavily_tool.py --check              # key configured? endpoint reachable?
  python tavily_tool.py --smoke "query"      # one real search, speech-safe digest
"""
from __future__ import annotations

import json
import sys
import time
from urllib.parse import urlparse

import requests

import iris_config as cfg


class TavilyError(RuntimeError):
    pass


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for current information. Use this for news, prices, "
            "versions, weather, sports, or anything that may have changed after your "
            "training data. Ask a short focused query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search query, three to eight words.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "How many sources to return. Default 3.",
                },
            },
            "required": ["query"],
        },
    },
}

API_BASE = "https://api.tavily.com"


def search(
    query: str,
    max_results: int = 3,
    topic: str = "general",
    search_depth: str = "basic",
    include_answer: bool | None = None,
    timeout: float | None = None,
) -> dict:
    """Run a Tavily search and return the normalized response dict.

    The answer field (Tavily's synthesized digest) is enabled by default
    because it is ideal for spoken replies. Raises TavilyError on failure.
    """
    query = str(query or "").strip()
    if not query:
        raise TavilyError("empty query")
    max_results = max(1, min(int(max_results), 5))
    include_answer = cfg.TAVILY_INCLUDE_ANSWER if include_answer is None else bool(include_answer)
    timeout = timeout or cfg.TAVILY_TIMEOUT

    # Preferred path: official SDK (pip install tavily-python).
    if cfg.TAVILY_API_KEY:
        try:
            from tavily import TavilyClient  # type: ignore

            res = TavilyClient(api_key=cfg.TAVILY_API_KEY).search(
                query=query,
                max_results=max_results,
                topic=topic,
                search_depth=search_depth,
                include_answer=include_answer,
            )
            return _normalize(dict(res))
        except ImportError:
            pass  # fall through to REST
        except Exception as e:  # SDK failed: retry once over REST below
            print(f"[tavily] SDK path failed ({type(e).__name__}), trying REST", file=sys.stderr)

    body = {
        "query": query,
        "max_results": max_results,
        "topic": topic,
        "search_depth": search_depth,
        "include_answer": include_answer,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.TAVILY_API_KEY:
        headers["Authorization"] = f"Bearer {cfg.TAVILY_API_KEY}"
    else:
        # Keyless access: no account needed, rate limited.
        headers["X-Tavily-Access-Mode"] = "keyless"

    try:
        r = requests.post(f"{API_BASE}/search", json=body, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise TavilyError(f"network error: {e}") from None

    if r.status_code != 200:
        detail = ""
        try:
            j = r.json()
            detail = j.get("detail") or j.get("message") or json.dumps(j)[:200]
        except Exception:
            detail = r.text[:200]
        mode = "key" if cfg.TAVILY_API_KEY else "keyless"
        raise TavilyError(f"HTTP {r.status_code} ({mode} mode): {detail}")
    try:
        return _normalize(r.json())
    except Exception as e:
        raise TavilyError(f"bad response shape: {e}") from None


def _normalize(res: dict) -> dict:
    """Bound and normalize a Tavily response.

    The payload stays small (a few hundred tokens) so tool results never
    crowd the model context. Titles/URLs/snippets are preserved verbatim -
    the model is instructed to ground claims in these and to never invent
    sources. Empty results are reported as such, never silently.
    """
    results = []
    for item in (res.get("results") or [])[:5]:
        results.append(
            {
                "title": str(item.get("title", "") or "")[:200],
                "url": str(item.get("url", "") or "")[:300],
                "snippet": str(item.get("content", "") or "")[:400],
            }
        )
    return {
        "query": str(res.get("query", "") or "")[:200],
        "answer": str(res.get("answer") or "")[:600],
        "results": results,
        "empty": not results and not (res.get("answer") or ""),
    }


def run_tool(args: dict) -> dict:
    """Execute the web_search tool for llama-server. Returns a JSON payload.

    Shape (bounded, evidence-first):
      ok, elapsed_ms, answer (search-engine synthesis - a clue, not a source),
      results[{title,url,snippet}] (the actual evidence), guidance, speak_hint.
    Failures and empty result sets are honest `ok:false`/`empty` responses so
    the model says it could not verify instead of guessing.
    """
    try:
        query = str((args or {}).get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        t0 = time.time()
        res = search(query, max_results=(args or {}).get("max_results", 3))
        payload = {
            "ok": True,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "answer": res["answer"],
            "results": res["results"],
        }
        # compute emptiness from the payload itself (robust to any shape)
        is_empty = bool(res.get("empty")) or (
            not res.get("results") and not (res.get("answer") or "").strip()
        )
        if is_empty:
            payload["empty"] = True
            payload["guidance"] = (
                "No results were returned. Tell the user you could not find or "
                "verify this; do not answer from memory as if it were checked."
            )
        else:
            payload["guidance"] = (
                "Ground every factual claim in the results below. The 'answer' "
                "field is the search engine's synthesized digest, not a source. "
                "Only mention sites that appear in the results."
            )
        # Guidance the LLM can use verbatim when speaking.
        payload["speak_hint"] = speakable(payload, budget="standard")
        return payload
    except TavilyError as e:
        return {"ok": False, "error": f"web search unavailable: {e}"}
    except Exception as e:  # never leak anything unexpected
        return {"ok": False, "error": f"web search failed: {type(e).__name__}"}


def speakable(result: dict, budget: str = "standard") -> str:
    """Turn a search result into a TTS-safe spoken digest.

    Plain short sentences, no markdown, no URLs (site names only), clipped to
    the word budget. This is the fallback text if you want to bypass the LLM
    and speak the search result directly.
    """
    parts: list[str] = []
    answer = str(result.get("answer") or "").strip()
    if answer:
        parts.append(cfg.speech_clean(answer))
    if not parts or len(" ".join(parts).split()) < 10:
        for r in (result.get("results") or [])[:2]:
            domain = urlparse(r.get("url", "")).netloc.replace("www.", "")
            first = cfg.speech_clean(r.get("snippet", "")).split(". ")[0]
            if domain and first:
                parts.append(f"According to {domain}, {first}.")
    text = " ".join(p for p in parts if p).strip()
    if not text:
        text = "The network has nothing useful on that."
    return cfg.clip_words(text, budget)


def check() -> dict:
    """Connectivity check. Never prints the key itself."""
    info = {"key_configured": bool(cfg.TAVILY_API_KEY), "endpoint": f"{API_BASE}/search"}
    try:
        res = search("ping test", max_results=1, include_answer=False, timeout=6.0)
        info["reachable"] = True
        info["mode"] = "key" if cfg.TAVILY_API_KEY else "keyless"
        info["sample_titles"] = [r["title"][:60] for r in res["results"][:2]]
    except TavilyError as e:
        info["reachable"] = False
        info["error"] = str(e)
    return info


if __name__ == "__main__":
    if "--check" in sys.argv:
        print(json.dumps(check(), indent=2))
    elif "--smoke" in sys.argv:
        q = sys.argv[sys.argv.index("--smoke") + 1] if len(sys.argv) > sys.argv.index("--smoke") + 1 else "current stable Python version"
        t0 = time.time()
        try:
            res = search(q)
            dt = (time.time() - t0) * 1000
            print(f"query: {q!r}  ({int(dt)} ms, {'key' if cfg.TAVILY_API_KEY else 'keyless'} mode)")
            print("answer:", res["answer"][:300] or "(none)")
            for r in res["results"]:
                print(f"  - {r['title'][:70]}  [{urlparse(r['url']).netloc}]")
            print("\nspoken digest (TTS-safe):")
            print(" ", speakable(res))
        except TavilyError as e:
            print("FAILED:", e)
            sys.exit(1)
    else:
        print(__doc__)
