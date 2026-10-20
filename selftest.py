#!/usr/bin/env python3
"""Offline self-test for CRAG-pp.

The repository did not ship a selftest.py; this harness follows the sibling
IRIS/ULTRON selftest style (plain check() lines, tempfile.mkdtemp(), monkey-
patched transports). It runs entirely offline: no llama-server, no Tavily.

    python selftest.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))


HAPPY_FACTS = {
    "facts": [
        {
            "topic": "gpu_setup",
            "fact": "The user's GPU setup is an RTX 4050 with 6 GB of VRAM.",
            "importance": 0.8,
        },
        {
            "topic": "keyboard",
            "fact": "The user uses a Keychron K2 keyboard with brown switches.",
            "importance": 0.7,
        },
        {
            "topic": "project_decision",
            "fact": "The user decided to use Parakeet TDT v3 for STT, not v2.",
            "importance": 0.9,
        },
        {
            "topic": "location",
            "fact": "The user lives in Kyoto and likes ramen near the station.",
            "importance": 0.6,
        },
    ],
    "narrative": (
        "The session covered the user's RTX 4050 GPU setup, a Keychron K2 "
        "keyboard, and the decision to use Parakeet TDT v3 for STT."
    ),
}

FAKE_TURNS = [
    "I run an RTX 4050 with 6 GB of VRAM in my laptop.",
    "My keyboard is a Keychron K2 and I prefer brown switches.",
    "I decided to use Parakeet TDT v3 for STT instead of v2.",
    "I live in Kyoto and my favorite ramen shop is near the station.",
    "I prefer dark mode in every editor.",
    "My next project is a local voice assistant called ULTRON.",
]


def test_crag_session_summary():
    print("\n== CRAG session summarization (offline, stub LLM) ==")
    import evaluator_router
    import memory_store
    import temporal_memory

    tmp = Path(tempfile.mkdtemp(prefix="crag_session_summary_"))
    original_control = evaluator_router._control_complete
    pairs: list[tuple] = []

    def new_pair(name, turns):
        """Fresh temporal log + memory store with embeddings disabled."""
        temporal = temporal_memory.TemporalMemory(
            db_path=str(tmp / f"{name}_temporal.db"), visual=False
        )
        store = memory_store.MemoryStore(db_path=str(tmp / f"{name}_memory.db"))
        store._embed_batch = lambda texts: None
        for query in turns:
            temporal.begin_turn()
            temporal.save_turn(query=query, answer="noted", verdict="VOICE")
        pairs.append((temporal, store))
        return temporal, store

    calls: list[dict] = []

    def stub_ok(messages, json_mode, num_predict):
        calls.append({"json_mode": json_mode, "num_predict": num_predict})
        return json.dumps(HAPPY_FACTS)

    def row_count(store):
        return store.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    try:
        # 1. Happy path, 6-turn fake session.
        temporal, store = new_pair("happy", FAKE_TURNS)
        evaluator_router._control_complete = stub_ok
        result = temporal_memory.summarize_session(temporal, store)
        check(
            "happy path saves facts + narrative",
            result["status"] == "ok"
            and result["facts_saved"] >= 2
            and result["narrative_saved"] is True,
            f"result={result}",
        )

        # 2. Below minimum: 3 turns are skipped before any write.
        short_temporal, short_store = new_pair("short", FAKE_TURNS[:3])
        short_result = temporal_memory.summarize_session(short_temporal, short_store)
        check(
            "below-minimum session skipped with no writes",
            short_result["status"] == "skipped" and row_count(short_store) == 0,
            f"result={short_result} rows={row_count(short_store)}",
        )

        # 3. Idempotency: second pass is a no-op.
        before = row_count(store)
        second = temporal_memory.summarize_session(temporal, store)
        check(
            "second call is idempotent and adds no rows",
            second["status"] == "already_summarized"
            and row_count(store) == before,
            f"result={second} before={before} after={row_count(store)}",
        )

        # 4. force=True bypasses the idempotency check.
        call_count = len(calls)
        forced = temporal_memory.summarize_session(temporal, store, force=True)
        check(
            "force=True runs the summarizer again",
            forced["status"] == "ok" and len(calls) > call_count,
            f"result={forced} calls={call_count}->{len(calls)}",
        )

        # 5. Extractive check: named entities appear verbatim in the turns.
        fact_rows = store.db.execute(
            "SELECT content FROM memories WHERE kind = 'session_summary'"
        ).fetchall()
        fact_text = " ".join(row[0] for row in fact_rows)
        check(
            "stored facts contain a named entity from the transcript",
            any(entity in fact_text for entity in ("Keychron", "RTX 4050", "Kyoto")),
            f"facts={fact_text!r}",
        )

        # 6. Correct provenance tags.
        tags = store.db.execute(
            "SELECT source, kind FROM memories WHERE kind IN "
            "('session_summary', 'session_narrative')"
        ).fetchall()
        facts_ok = any(
            source == "session_auto" and kind == "session_summary"
            for source, kind in tags
        )
        narrative_ok = any(
            source == "session_auto" and kind == "session_narrative"
            for source, kind in tags
        )
        check(
            "facts/narrative use session_auto + correct kinds",
            facts_ok and narrative_ok,
            f"tags={tags}",
        )

        # 7. Retrievable in a later session/store against the same DB.
        later = memory_store.MemoryStore(db_path=store.db_path)
        later._embed_batch = lambda texts: None
        pairs.append((None, later))
        hits = later.search("gpu setup", k=5)
        check(
            "session_summary is retrievable in a later store",
            any(
                hit.kind == "session_summary" and hit.score > 0
                for hit in hits
            ),
            f"hits={[(h.kind, h.score, h.content[:40]) for h in hits]}",
        )

        # 8. Empty facts + empty narrative is a clean ok.
        empty_temporal, empty_store = new_pair("empty", FAKE_TURNS)
        evaluator_router._control_complete = (
            lambda messages, json_mode, num_predict:
            json.dumps({"facts": [], "narrative": ""})
        )
        empty_result = temporal_memory.summarize_session(empty_temporal, empty_store)
        empty_rows = empty_store.db.execute(
            "SELECT COUNT(*) FROM memories WHERE kind IN "
            "('session_summary', 'session_narrative')"
        ).fetchone()[0]
        check(
            "empty facts + empty narrative return ok with no fact rows",
            empty_result["status"] == "ok"
            and empty_result["facts_saved"] == 0
            and empty_result["narrative_saved"] is False
            and empty_rows == 0,
            f"result={empty_result} rows={empty_rows}",
        )

        # 9. Malformed JSON on both attempts -> llm_failed, no writes.
        bad_temporal, bad_store = new_pair("badjson", FAKE_TURNS)
        evaluator_router._control_complete = (
            lambda messages, json_mode, num_predict: "not json at all"
        )
        bad_result = temporal_memory.summarize_session(bad_temporal, bad_store)
        check(
            "malformed JSON returns llm_failed without writing rows",
            bad_result["status"] == "llm_failed" and row_count(bad_store) == 0,
            f"result={bad_result} rows={row_count(bad_store)}",
        )

        # 10. Transport ConnectionError -> llm_failed, no writes, no raise.
        error_temporal, error_store = new_pair("connerr", FAKE_TURNS)

        def boom(messages, json_mode, num_predict):
            import requests

            raise requests.ConnectionError("simulated transport failure")

        evaluator_router._control_complete = boom
        try:
            error_result = temporal_memory.summarize_session(
                error_temporal, error_store
            )
            raised = False
        except Exception:
            error_result = {}
            raised = True
        check(
            "transport ConnectionError returns llm_failed without raising",
            not raised
            and error_result.get("status") == "llm_failed"
            and row_count(error_store) == 0,
            f"result={error_result} raised={raised} rows={row_count(error_store)}",
        )

        # 11. Zero-turn session -> no_session.
        evaluator_router._control_complete = stub_ok
        zero_temporal, zero_store = new_pair("zero", [])
        zero_result = temporal_memory.summarize_session(zero_temporal, zero_store)
        check(
            "zero-turn session returns no_session",
            zero_result["status"] == "no_session"
            and zero_result["session_id"] == zero_temporal.session_id
            and row_count(zero_store) == 0,
            f"result={zero_result}",
        )

        # 12. Embedding backend down -> facts still stored with embedding=NULL.
        noemb_temporal, noemb_store = new_pair("noembed", FAKE_TURNS)

        def embed_down(texts):
            raise RuntimeError("embedding backend down")

        noemb_store._embed_batch = embed_down
        noemb_result = temporal_memory.summarize_session(noemb_temporal, noemb_store)
        null_rows = noemb_store.db.execute(
            "SELECT COUNT(*) FROM memories WHERE kind = 'session_summary' "
            "AND embedding IS NULL"
        ).fetchone()[0]
        check(
            "embedding failure still saves facts with NULL vectors",
            noemb_result["status"] == "ok"
            and noemb_result["facts_saved"] >= 2
            and null_rows >= 2,
            f"result={noemb_result} null_rows={null_rows}",
        )

        # 13. Slow LLM is capped on the synchronous exit path.
        slow_temporal, slow_store = new_pair("slow", FAKE_TURNS)

        def slow(messages, json_mode, num_predict):
            time.sleep(5.0)
            return json.dumps(HAPPY_FACTS)

        evaluator_router._control_complete = slow
        started = time.time()
        slow_result = temporal_memory.summarize_session(
            slow_temporal, slow_store, timeout=3.0
        )
        elapsed = time.time() - started
        check(
            "3-second cap aborts a slow LLM call cleanly",
            slow_result["status"] == "llm_failed" and elapsed < 3.5,
            f"result={slow_result} elapsed={elapsed:.2f}s",
        )

        # 14. Auto-flush fires exactly once at turn 20 (daemon thread).
        import contextlib
        import io

        import main

        auto_temporal = temporal_memory.TemporalMemory(
            db_path=str(tmp / "auto_temporal.db"), visual=False
        )
        pairs.append((auto_temporal, None))
        flush_calls: list[tuple] = []
        original_generate = main.generate_answer
        original_get_store = main.get_store
        original_summarize = main.summarize_session

        def spy_summarize(*args, **kwargs):
            flush_calls.append((args, kwargs))
            return {"status": "ok"}

        def fake_generate(*args, **kwargs):
            yield "ok"

        main.generate_answer = fake_generate
        main.get_store = lambda: object()
        main.summarize_session = spy_summarize
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for _index in range(20):
                    main.run_crag_pipeline(
                        "hi", engine=None, memory=auto_temporal, verbose=False
                    )
                for _ in range(300):
                    if flush_calls:
                        break
                    time.sleep(0.01)
        finally:
            main.generate_answer = original_generate
            main.get_store = original_get_store
            main.summarize_session = original_summarize

        check(
            "auto-flush calls summarize_session exactly once at turn 20",
            len(flush_calls) == 1,
            f"calls={len(flush_calls)} turns={auto_temporal._session_turns}",
        )
    finally:
        evaluator_router._control_complete = original_control
        for temporal, store in pairs:
            try:
                if store is not None:
                    store.close()
            except Exception:
                pass
            try:
                if temporal is not None:
                    temporal._conn.close()
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_crag_session_summary()
    failed = [result for result in RESULTS if not result[1]]
    print(
        f"\n{'=' * 50}\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed"
        + (f", {len(failed)} FAILED" if failed else "")
    )
    for name, _ok, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
