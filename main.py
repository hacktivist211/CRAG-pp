import argparse
import os
import sys
import time

from evaluator_router import (
    LOWER_THRESHOLD,
    MAX_CONTEXT_CHUNKS,
    WEB_RESULT_LIMIT,
    classify_query,
    grade_documents,
    refine_knowledge,
    rewrite_query,
    route_verdict,
    web_search_ddg,
)
from evaluator_router import CRAG_BACKEND, LLAMA_SERVER_URL
from generator import backend_info, format_sources, generate_answer
from retrieval import HYBRID_RETRIEVAL, PERSIST_DIRECTORY, SEARCH_EF, VectorEngine
from temporal_memory import MEMORY_DB_PATH, TemporalMemory

TOP_K = 8

BANNER = """
================================================================
  Self-Corrective RAG (CRAG) with Temporal Memory
  llm backend : {backend}
                (the same llama-server as the primary IRIS runtime;
                 no second model is ever loaded; Ollama stays off)
  embeddings  : BAAI/bge-m3, cpu by default (CRAG_EMBED_DEVICE=cuda
                for standalone GPU sessions with the LLM server stopped)
  vector db   : chromadb persistent store, hnsw cosine + bm25 hybrid
  fallback    : duckduckgo web search, optional tavily
  evaluator   : llama-server relevance grading at temperature 0
  memory      : sqlite episodic store (wal) + background png archive
  routing     : simple / memory / full crag control plane
================================================================
""".format(backend=backend_info())


def _llm_health_hint():
    """One quick, silent health probe so a dead backend is obvious early."""
    if CRAG_BACKEND != "llamaserver":
        return
    try:
        import urllib.request

        urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=2)
    except Exception:
        print(
            f"[crag] WARNING: llama-server is not answering at {LLAMA_SERVER_URL}.\n"
            "       start it with ./start_llama.sh; CRAG will not (and must not)\n"
            "       start any model service itself."
        )

MENU = """----------------------------------------------------------------
  KNOWLEDGE BASE OPTIONS
  1. chunk a directory (recursive .txt .md .pdf)
  2. continue with the existing knowledge base
  3. chunk only new or changed documents from the last directory
----------------------------------------------------------------"""

WEB_FALLBACK_NOTICE = (
    "Live web search is unavailable or rate-limited right now. This answer relies on "
    "the local knowledge base, whose content was only partially relevant, so treat it with care."
)
PARAMETRIC_NOTICE = (
    "Neither the local knowledge base nor live web search returned usable information. "
    "The answer below comes from the model's general knowledge and is unverified."
)


def _log(message):
    print(f"[crag] {message}")


def _enable_ansi():
    if os.name == "nt":
        os.system("")


def _prompt_path(message):
    try:
        return input(message).strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        return ""


def _choose_ingestion(engine):
    print(MENU)
    try:
        choice = input("select an option (1/2/3): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if choice == "1":
        path = _prompt_path("enter the directory path to chunk: ")
        if path:
            engine.ingest_directory(path)
        else:
            print("[crag] no directory given, continuing")
    elif choice == "3":
        last = engine.last_directory()
        if not last:
            print("[crag] no previous directory on record")
            last = _prompt_path("enter the directory path to chunk: ")
        if last:
            print(f"[crag] scanning {last}")
            engine.ingest_directory(last)
    else:
        print("[crag] continuing with the existing knowledge base")


def run_crag_pipeline(query, engine, memory, top_k=TOP_K, ef=None, verbose=True):
    started = time.time()
    log = _log if verbose else (lambda message: None)

    # 1. temporal injector: session handling plus current-time awareness
    turn_no = memory.begin_turn()
    temporal_note = memory.build_temporal_context()
    recent = memory.recent_turns()

    # 2. adaptive complexity router
    route = classify_query(query)
    kind = route["kind"]
    log(f"query route: {kind} ({route['reason']})")

    context_blocks = []
    sources = []
    scores = []
    selected_chunks = []
    notice = None
    verdict = None
    mode = "retrieval"

    if recent:
        context_blocks.append(
            {
                "source": "recent conversation",
                "text": memory.format_recent(recent),
                "rank": 3.4,
            }
        )

    if kind == "simple":
        mode = "direct"
        verdict = "DIRECT"
        log("simple query, answering directly from time and memory context")

    elif kind == "memory":
        mode = "memory"
        verdict = "MEMORY"
        hits = memory.recall(query)
        log(f"memory recall: {len(hits)} matching interaction(s)")
        text = (
            memory.format_hits(hits)
            if hits
            else "(No matching past interactions are recorded for this request.)"
        )
        context_blocks.append({"source": "conversation memory", "text": text, "rank": 4.2})

    else:
        # ---- full CRAG control plane ----
        chunks = []
        try:
            chunks = engine.retrieve(query, k=top_k)
        except Exception as error:
            log(f"local retrieval failed: {error}")
        documents = [chunk["text"] for chunk in chunks]
        log(f"retrieved {len(documents)} local chunk(s) for grading")

        try:
            grading = grade_documents(query, documents)
        except Exception:
            grading = {"scores": [0.5 for _ in documents], "search_query": None}
        scores = grading.get("scores") or []
        if scores:
            log("chunk scores: " + ", ".join(f"{value:.2f}" for value in scores))

        scored = []
        for chunk, score in zip(chunks, scores):
            if score >= LOWER_THRESHOLD:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        scored = scored[:MAX_CONTEXT_CHUNKS]
        selected_chunks = [chunk for _score, chunk in scored]
        selected_documents = [chunk["text"] for chunk in selected_chunks]
        dropped = len(documents) - len(selected_chunks)
        log(
            f"chunk selection: kept {len(selected_chunks)}/{len(documents)} "
            f"(dropped {dropped} below {LOWER_THRESHOLD:.1f})"
        )

        verdict = route_verdict(scores)
        log(f"routing verdict: {verdict}")

        local_refined = ""

        if verdict in ("CORRECT", "AMBIGUOUS") and selected_documents:
            local_refined = refine_knowledge(query, selected_documents)
            if local_refined:
                context_blocks.append(
                    {
                        "source": "local documents",
                        "text": local_refined,
                        "rank": 3.0 if verdict == "CORRECT" else 2.4,
                    }
                )
                seen = set()
                for chunk in selected_chunks:
                    key = (chunk["source"], chunk.get("path", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    sources.append(
                        {"label": chunk["source"], "detail": chunk.get("path") or "local knowledge base"}
                    )
            else:
                log("knowledge refinement discarded every local sentence")

        if verdict in ("AMBIGUOUS", "INCORRECT"):
            rewritten = grading.get("search_query") or rewrite_query(query)
            log(f"rewritten web query: {rewritten}")
            try:
                web_results = web_search_ddg(rewritten, max_results=WEB_RESULT_LIMIT)
            except Exception:
                web_results = []
            log(f"web search returned {len(web_results)} result(s)")
            if web_results:
                web_texts = [f"{item['title']}. {item['snippet']}" for item in web_results]
                refined_web = refine_knowledge(query, web_texts)
                web_rank = 4.2 if verdict == "INCORRECT" else 2.8
                if refined_web:
                    context_blocks.append(
                        {"source": "live web results", "text": refined_web, "rank": web_rank}
                    )
                else:
                    context_blocks.append(
                        {
                            "source": "live web results (unfiltered)",
                            "text": " ".join(web_texts),
                            "rank": web_rank,
                        }
                    )
                for item in web_results:
                    sources.append({"label": item["title"] or "web result", "detail": item["url"] or "web search"})
            else:
                log("web search unavailable or empty, engaging circuit breaker")
                if local_refined:
                    notice = WEB_FALLBACK_NOTICE
                else:
                    notice = PARAMETRIC_NOTICE

        # semantic memory overlay: related past interactions for continuity
        overlay = memory.search(query)
        if overlay:
            log(f"memory overlay: {len(overlay)} related past interaction(s)")
            context_blocks.append(
                {"source": "conversation memory", "text": memory.format_hits(overlay), "rank": 3.6}
            )

        if not context_blocks and not notice:
            notice = PARAMETRIC_NOTICE

    if notice:
        log(f"notice: {notice}")
        if context_blocks:
            context_blocks.append({"source": "system notice", "text": notice, "rank": 1.0})

    # 6. generation with lost-in-the-middle packing and the TIME block
    log(f"generating answer from {len(context_blocks)} context block(s) [{mode}]")
    print("\nIRIS: ", end="", flush=True)
    collected = []
    for token in generate_answer(query, context_blocks, temporal_note=temporal_note, mode=mode):
        sys.stdout.write(token)
        sys.stdout.flush()
        collected.append(token)
    print()
    answer = "".join(collected)

    # 7. dual memory write: sqlite synchronously, png visual archive in background
    memory.save_turn(
        query=query,
        answer=answer,
        verdict=verdict,
        sources=sources,
        turn_no=turn_no,
        elapsed=time.time() - started,
    )

    elapsed = time.time() - started
    if verbose:
        print()
        print("[crag] sources:")
        print(format_sources(sources))
        print(f"[crag] route {kind}, verdict {verdict}, memory turn {turn_no} stored")
        print(f"[crag] finished in {elapsed:.1f}s")

    return {
        "route": kind,
        "verdict": verdict,
        "scores": scores,
        "selected": len(selected_chunks),
        "sources": sources,
        "notice": notice,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Corrective RAG with intelligent web fallback")
    parser.add_argument("--ingest-dir", dest="ingest_dir", default=None, metavar="DIR",
                        help="chunk a directory recursively before chatting")
    parser.add_argument("--sync", action="store_true",
                        help="chunk only new or changed documents from the last directory")
    parser.add_argument("--k", type=int, default=TOP_K, help="number of chunks to retrieve and grade per query")
    parser.add_argument("--ef", type=int, default=None,
                        help=f"hnsw search depth, default {SEARCH_EF} (16-32 fast, 128-256 max recall)")
    parser.add_argument("--reset", action="store_true",
                        help="wipe the chroma store and the manifest before starting")
    parser.add_argument("--quiet", action="store_true", help="hide pipeline trace and source footer")
    parser.add_argument("--no-menu", action="store_true", help="skip the startup menu")
    parser.add_argument("--vision", choices=["auto", "force", "off"], default="auto",
                        help="vision OCR mode: auto = only pages without a text layer, "
                             "force = every pdf page, off = never")
    args = parser.parse_args()

    _enable_ansi()

    if args.reset:
        manifest_path = os.path.join(PERSIST_DIRECTORY, "manifest.json")
        if os.path.isfile(manifest_path):
            os.remove(manifest_path)
            print("[crag] manifest removed")

    try:
        engine = VectorEngine(pool_kind="process", max_workers=4, drop_old=args.reset)
    except Exception as error:
        print(f"[fatal] could not initialize the vector engine: {error}")
        sys.exit(1)

    try:
        memory = TemporalMemory()
    except Exception as error:
        print(f"[fatal] could not initialize temporal memory: {error}")
        sys.exit(1)

    if args.ingest_dir:
        engine.ingest_directory(args.ingest_dir)
    elif args.sync:
        last = engine.last_directory()
        if last:
            print(f"[crag] scanning {last}")
            engine.ingest_directory(last)
        else:
            print("[crag] no previous directory on record, nothing to sync")

    print(BANNER)
    _llm_health_hint()
    print(f"local chunks in store : {engine.chunk_count()}")
    print(
        f"retrieval             : top-{args.k} chunks, ef {args.ef or SEARCH_EF}, "
        f"hybrid bm25 {'on' if HYBRID_RETRIEVAL else 'off'}"
    )
    if engine.last_directory():
        print(f"last chunked directory: {engine.last_directory()}")
    memory_info = memory.stats()
    since = f" since {memory_info['oldest'][:10]}" if memory_info["oldest"] else ""
    print(f"memory turns stored   : {memory_info['turns']}{since}")
    print(f"memory database       : {MEMORY_DB_PATH}")
    last_interaction = memory.last_interaction()
    if last_interaction:
        print(f"last interaction      : {last_interaction['human']}")
        if last_interaction["seconds"] > 3600:
            print(
                f"[iris] welcome back, it has been {last_interaction['human']} "
                "since your last interaction."
            )
    print(
        "controls              : /menu ingestion options, /stats store+memory stats, "
        "/memory recent turns, /forget last|all, exit to quit"
    )

    if not (args.ingest_dir or args.sync or args.no_menu):
        _choose_ingestion(engine)
        print(f"local chunks in store : {engine.chunk_count()}")

    verbose = not args.quiet
    while True:
        try:
            print()
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nIRIS: goodbye.")
            break
        if not query:
            continue
        lowered = query.lower()
        if lowered in {"exit", "quit", "/exit", "/quit", "bye"}:
            print("IRIS: goodbye.")
            break
        if lowered == "/menu":
            _choose_ingestion(engine)
            print(f"local chunks in store : {engine.chunk_count()}")
            continue
        if lowered == "/stats":
            print(f"local chunks in store : {engine.chunk_count()}")
            print(f"last chunked directory: {engine.last_directory() or 'none'}")
            info = memory.stats()
            print(f"memory turns stored   : {info['turns']} across {info['sessions']} session(s)")
            print(f"memory oldest turn    : {info['oldest'] or 'n/a'}")
            print(f"memory db size        : {info['db_bytes'] / 1024.0:.0f} kb")
            print(f"visual archive pngs   : {info['visual_files']}")
            continue
        if lowered == "/memory":
            turns = memory.recent_turns(10)
            if not turns:
                print("[iris] no interactions recorded yet")
            else:
                for turn in turns:
                    print(f"[{turn['when']}] You: {turn['query']}")
                    answer = turn.get("answer") or ""
                    if answer:
                        preview = answer[:70]
                        tail = "..." if len(answer) > 70 else ""
                        print(f"    IRIS: {preview}{tail}")
            continue
        if lowered.startswith("/forget"):
            parts = lowered.split()
            scope = parts[1] if len(parts) > 1 else "last"
            if scope not in ("last", "all"):
                print("usage: /forget last | /forget all")
            else:
                deleted = memory.forget(scope)
                print(f"[iris] forgot {deleted} turn(s)")
            continue
        try:
            run_crag_pipeline(query, engine, memory, top_k=args.k, ef=args.ef, verbose=verbose)
        except KeyboardInterrupt:
            print("\n[crag] interrupted")
        except Exception as error:
            print(f"\n[crag] pipeline error: {error}")
            print("IRIS: something went wrong while processing that question. Please try rephrasing it.")


if __name__ == "__main__":
    main()
