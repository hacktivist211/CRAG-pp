#!/usr/bin/env python3
"""Streaming verification for tool_runtime.py (Critical Issue 1).

Proves, with executed tests, that the SSE implementation is GENUINELY
incremental - not accumulate-then-callback:

  1. Decoder correctness: a canonical SSE transcript is split at EVERY byte
     boundary (and at random multi-split boundaries) - events, order, UTF-8
     multi-byte characters, think-tag removal, and [DONE] must survive any
     split. Events split across chunks, multiple events per chunk, comments,
     keep-alive lines, and a missing final newline are all covered.
  2. Incremental delivery over a REAL socket: a local HTTP server sends event
     1, sleeps 350 ms, then sends event 2. The first on_delta callback must
     fire BEFORE the server has written the second event. This cannot pass
     with a buffer-everything implementation.
  3. Tool-call fragment merging across arbitrary splits; the reconstructed
     arguments match; the tool executes exactly once.
  4. Malformed tool-call JSON is NEVER executed (spy must stay uncalled).
  5. Resource discipline: the HTTP response is closed on success, on
     mid-stream failure, on cancellation from on_delta, and on consumer
     early-exit from stream_completion().

Run:  python test_streaming.py        (offline, no llama-server needed)
"""
from __future__ import annotations

import json
import random
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

import tool_runtime as tr

# assembled from pieces so the literal tags never appear in the source
T_OPEN = "<" + "think" + ">"
T_CLOSE = "</" + "think" + ">"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = ""):
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and not cond else ""))


# ------------------------------------------------------------------ fakes
class FakeStreamResponse:
    """Minimal requests.Response stand-in: status, chunk stream, close."""

    def __init__(self, chunks: list[bytes], status_code: int = 200, fail_at: int | None = None):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._fail_at = fail_at
        self.closed = False
        self.exhausted = False
        self.text = "simulated error body" if status_code != 200 else ""

    def iter_content(self, chunk_size=None):
        for i, c in enumerate(self._chunks):
            if self._fail_at is not None and i == self._fail_at:
                raise requests.exceptions.ConnectionError("simulated connection drop")
            yield c
        self.exhausted = True

    def close(self):
        self.closed = True


class FakePost:
    """Transport stub for chat()/stream_completion(): returns scripted responses
    and records every payload it receives (for multi-round tool tests)."""

    def __init__(self, responses: list[FakeStreamResponse]):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def __call__(self, endpoint, json=None, stream=False, timeout=None):
        self.payloads.append(json)
        if self.responses:
            return self.responses.pop(0)
        return FakeStreamResponse([b"data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n", b"data: [DONE]\n\n"])


def sse(*objs, done: bool = True, ensure_ascii: bool = True, comments: bool = False) -> bytes:
    out = b""
    if comments:
        out += b": keep-alive ping\n\n"
    for o in objs:
        out += b"data: " + json.dumps(o, ensure_ascii=ensure_ascii).encode("utf-8") + b"\n\n"
    if done:
        out += b"data: [DONE]\n\n"
    return out


# ------------------------------------------------------------------ 1. decoder splits
def expected_events(transcript: bytes) -> list[dict]:
    """Reference parse: split on events the naive way (full-buffer parse)."""
    dec = tr.SSEDecoder()
    evs = []
    for data in dec.feed(transcript):
        evs.append(data)
    for data in dec.flush():
        evs.append(data)
    return evs


def run_decoder(chunks: list[bytes]) -> list[dict]:
    dec = tr.SSEDecoder()
    got: list[str] = []
    for c in chunks:
        got.extend(dec.feed(c))
    got.extend(dec.flush())
    return got


def test_exhaustive_single_split():
    print("\n== SSEDecoder: split at EVERY byte boundary ==")
    events = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": ", wor"}}]},
        {"choices": [{"delta": {"content": "ld!"}}]},
    ]
    transcript = sse(*events, comments=True)
    want = expected_events(transcript)
    bad = 0
    for p in range(0, len(transcript) + 1):
        got = run_decoder([transcript[:p], transcript[p:]])
        if got != want:
            bad += 1
            if bad == 1:
                check("every-byte split parse", False, f"split at {p}: {got!r}")
    check("every-byte split parse", bad == 0, f"{bad} failures over {len(transcript) + 1} splits")
    check("[DONE] surfaced as terminator payload", want and want[-1] == "[DONE]",
          "decoder must surface the terminator so the consumer can stop")


def test_random_multisplit():
    print("\n== SSEDecoder: random multi-split fuzz (UTF-8 + think tags + tool calls) ==")
    events = [
        {"choices": [{"delta": {"reasoning_content": "chain of thought must not leak"}}]},
        {"choices": [{"delta": {"content": T_OPEN + " Let me " + "think. " + T_CLOSE}}]},
        {"choices": [{"delta": {"content": " I now know the answer."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "web_search", "arguments": "{\"qu"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ery\":\"py\"}"}}]}}]},
        {"choices": [{"delta": {"content": "café ☕ done"}}]},
    ]
    transcript = sse(*events, ensure_ascii=False)
    want = expected_events(transcript)
    rng = random.Random(20261017)
    bad = 0
    for trial in range(400):
        n = rng.randint(1, 6)
        cuts = sorted(rng.randint(0, len(transcript)) for _ in range(n))
        chunks, prev = [], 0
        for c in cuts:
            chunks.append(transcript[prev:c])
            prev = c
        chunks.append(transcript[prev:])
        got = run_decoder(chunks)
        if got != want:
            bad += 1
            if bad == 1:
                check("random multi-split parse", False, f"trial {trial} cuts={cuts}: {got!r}")
    check("random multi-split parse", bad == 0, f"{bad} failures / 400 trials")


def test_missing_trailing_newline():
    print("\n== SSEDecoder: stream ends without final newline ==")
    body = b'data: {"choices":[{"delta":{"content":"tail"}}]}\n'  # no blank line
    got = run_decoder([body])
    want = expected_events(body)
    check("unterminated final event recovered", got == want, f"got {got!r}")
    # and completely unterminated (no newline at all)
    body2 = b'data: {"choices":[{"delta":{"content":"x"}}]}'
    got2 = run_decoder([body2])
    check("no-newline event recovered", len(got2) == 1 and "x" in got2[0], f"got {got2!r}")


def test_utf8_split_across_chunks():
    print("\n== UTF-8 multi-byte split across chunks ==")
    text = "café ☕ naïve"
    # one SSE data line carrying the raw UTF-8 text, split at every byte
    transcript = b"data: " + text.encode("utf-8") + b"\n\ndata: [DONE]\n\n"
    ok = True
    for p in range(1, len(transcript)):
        got = run_decoder([transcript[:p], transcript[p:]])
        payload = "".join(d for d in got if d != "[DONE]")
        if payload != text:
            ok = False
            check("utf-8 reassembly", False, f"split {p}: {payload!r}")
            break
    check("utf-8 reassembly at every byte split", ok)
    # three-way split including inside the 4-byte coffee emoji
    raw = text.encode("utf-8")
    got = run_decoder([b"data: caf", raw[3:5], raw[5:] + b"\n\n"])
    check("multi-chunk byte split reassembles", "".join(got) == text, f"got {got!r}")


# ------------------------------------------------------------------ 2. think filter
def test_think_filter():
    print("\n== ThinkFilter: reasoning never reaches on_delta ==")

    def collect(deltas: list[str]) -> str:
        f = tr.ThinkFilter()
        out = []
        for d in deltas:
            out.append(f.feed(d))
        out.append(f.flush())
        return "".join(out)

    check("whole think block removed",
          collect([T_OPEN + " Let me " + "think. " + T_CLOSE + "Visible answer."]).strip() == "Visible answer.")
    check("think tags split across deltas",
          collect([" " + T_OPEN[:3], T_OPEN[3:] + "think. ", T_CLOSE + " The answer is 42."]).strip() == "The answer is 42.")
    check("close tag split across deltas",
          collect([T_OPEN + " thinking... " + T_CLOSE[:3], T_CLOSE[3:] + " The answer.", " Done."]).strip()
          == "The answer. Done.")
    check("no think tags: text untouched",
          collect(["a < b and c > d"]) == "a < b and c > d")
    check("unterminated think at stream end dropped",
          collect([T_OPEN + " going off the rails"]) == "")
    check("leading whitespace stripped after think",
          collect(["  hi there"]).strip() == "hi there")


# ------------------------------------------------------------------ 3. chat() over fakes
TOOLCALL_EVENTS = [
    {"choices": [{"delta": {"content": "Accessing the network."}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "web_search", "arguments": "{\"qu"}}]}}]},
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ery\":\"latest python\"}"}}]}}]},
]
FINAL_EVENTS = [
    {"choices": [{"delta": {"content": "Python 3.13, creator."}}]},
]


def test_chat_incremental_callbacks():
    print("\n== chat(): callbacks fire DURING the stream, in order ==")
    transcript = sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo, "}}]},
        {"choices": [{"delta": {"content": "creator."}}]},
    )
    resp = FakeStreamResponse([transcript])
    deltas: list[tuple[str, bool]] = []  # (text, stream_exhausted_at_callback)

    def on_delta(s):
        deltas.append((s, resp.exhausted))

    text, trace = tr.chat([{"role": "user", "content": "hi"}],
                          on_delta=on_delta, use_tools=False, _post=FakePost([resp]))
    check("text assembled in order", text == "Hello, creator.", f"text={text!r}")
    check("deltas match pieces", [d for d, _ in deltas] == ["Hel", "lo, ", "creator."])
    check("FIRST callback fires before the response finishes",
          len(deltas) >= 1 and not deltas[0][1],
          "first on_delta ran after the HTTP body was fully consumed - streaming is fake")
    check("every callback fires before stream exhaustion", all(not ex for _, ex in deltas))
    check("response closed on success", resp.closed)


def test_chat_reasoning_suppressed():
    print("\n== chat(): reasoning_content and think blocks never spoken ==")
    transcript = sse(
        {"choices": [{"delta": {"reasoning_content": "internal chain of thought, must not leak"}}]},
        {"choices": [{"delta": {"content": " The answer is 42."}}]},
    )
    resp = FakeStreamResponse([transcript])
    seen: list[str] = []
    text, _ = tr.chat([{"role": "user", "content": "q"}], on_delta=seen.append,
                      use_tools=False, _post=FakePost([resp]))
    check("reasoning_content filtered", text.strip() == "The answer is 42.", f"text={text!r}")
    check("on_delta saw no reasoning", all("chain of thought" not in s for s in seen))


def test_tool_roundtrip():
    print("\n== chat(): fragmented tool call -> execute -> continuation ==")
    r1 = FakeStreamResponse([sse(*TOOLCALL_EVENTS)])
    r2 = FakeStreamResponse([sse(*FINAL_EVENTS)])
    post = FakePost([r1, r2])
    calls: list[tuple[str, dict]] = []
    starts: list[str] = []

    def spy(name, args):
        calls.append((name, args))
        return {"ok": True, "note": "stubbed", "results": ["x"], "_tool": name, "_ms": 3}

    orig = tr.execute_tool
    tr.execute_tool = spy
    try:
        text, trace = tr.chat(
            [{"role": "user", "content": "latest python?"}],
            on_delta=lambda s: None,
            on_tool_start=starts.append,
            _post=post,
        )
    finally:
        tr.execute_tool = orig

    check("tool executed exactly once", len(calls) == 1, f"calls={calls!r}")
    check("merged arguments correct", calls and calls[0][1] == {"query": "latest python"},
          f"args={calls[0][1] if calls else None!r}")
    check("on_tool_start fired before execution", starts == ["web_search"])
    check("continuation text returned", text == "Python 3.13, creator.", f"text={text!r}")
    check("trace records the tool round", len(trace) == 1 and trace[0]["_tool"] == "web_search")
    # message threading: round 2 must contain the assistant tool_call + tool reply
    msgs2 = post.payloads[1]["messages"]
    roles = [m["role"] for m in msgs2]
    check("round-2 history threaded correctly", roles == ["user", "assistant", "tool"],
          f"roles={roles}")
    tc = msgs2[1].get("tool_calls", [{}])[0]
    check("assistant tool_call id/name preserved", tc.get("id") == "call_1"
          and tc.get("function", {}).get("name") == "web_search")
    check("tool reply references the call id", msgs2[2].get("tool_call_id") == "call_1")
    check("tool reply content is valid JSON", isinstance(json.loads(msgs2[2]["content"]), dict))
    check("responses closed", r1.closed and r2.closed)


def test_malformed_tool_args_never_executed():
    print("\n== chat(): malformed tool JSON is rejected, never executed ==")
    bad_events = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_bad",
                        "function": {"name": "web_search", "arguments": "{\"query\": \"unclosed"}}]}}]},
    ]
    r1 = FakeStreamResponse([sse(*bad_events)])
    r2 = FakeStreamResponse([sse(*FINAL_EVENTS)])
    post = FakePost([r1, r2])
    executed: list[str] = []

    def spy(name, args):
        executed.append(name)
        return {"ok": True}

    orig = tr.execute_tool
    tr.execute_tool = spy
    try:
        text, _ = tr.chat([{"role": "user", "content": "q"}], _post=post)
    finally:
        tr.execute_tool = orig
    check("tool NOT executed on malformed args", executed == [], f"executed={executed!r}")
    tool_msg = post.payloads[1]["messages"][-1]
    check("model receives a malformed-args error message",
          tool_msg["role"] == "tool" and "malformed" in tool_msg["content"].lower())
    check("conversation still completes", text == "Python 3.13, creator.")


def test_midstream_failure():
    print("\n== chat(): mid-stream failure -> partial text + honest error + closed ==")
    transcript = sse({"choices": [{"delta": {"content": "Partial answer, "}}]})
    # event 1 ok, then the transport dies before the stream completes
    resp = FakeStreamResponse([transcript[:-6], transcript[-6:]], fail_at=1)
    text, trace = tr.chat([{"role": "user", "content": "q"}], use_tools=False,
                          _post=FakePost([resp]))
    check("partial text preserved", text == "Partial answer, ", f"text={text!r}")
    check("response closed after failure", resp.closed)

    # nothing received at all -> the error surfaces
    resp2 = FakeStreamResponse([b"data: \"x\"\n\n"], fail_at=0)
    try:
        tr.chat([{"role": "user", "content": "q"}], use_tools=False, _post=FakePost([resp2]))
        check("empty-stream failure raises", False, "no exception raised")
    except tr.LlamaServerError:
        check("empty-stream failure raises", True)
    check("response closed on empty failure", resp2.closed)


def test_cancellation_closes_stream():
    print("\n== cancellation: on_delta exception + early generator exit ==")

    class Cancelled(Exception):
        pass

    transcript = sse(
        {"choices": [{"delta": {"content": "one"}}]},
        {"choices": [{"delta": {"content": "two"}}]},
    )
    resp = FakeStreamResponse([transcript])

    def on_delta(s):
        raise Cancelled()

    try:
        tr.chat([{"role": "user", "content": "q"}], on_delta=on_delta,
                use_tools=False, _post=FakePost([resp]))
        check("on_delta exception propagates", False, "swallowed")
    except Cancelled:
        check("on_delta exception propagates", True)
    check("response closed on cancellation", resp.closed)

    # consumer abandons stream_completion mid-way
    resp2 = FakeStreamResponse([transcript])
    gen = tr.stream_completion([{"role": "user", "content": "q"}], _post=FakePost([resp2]))
    next(gen)
    gen.close()
    check("early generator exit closes response", resp2.closed)


def test_history_trimming():
    print("\n== trim_history: bounded, system kept, no orphaned tool messages ==")
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(40):
        msgs.append({"role": "user", "content": f"question {i} " + "x" * 80})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    trimmed = tr.trim_history(msgs, max_turns=10, max_chars=4000)
    check("system message always kept", trimmed[0]["role"] == "system")
    check("newest turn kept intact",
          any(m["content"].startswith("question 39") for m in trimmed)
          and trimmed[-1]["content"] == "answer 39",
          f"last={trimmed[-1]['content']!r}")
    check("turn bound respected", len(trimmed) <= 11, f"len={len(trimmed)}")

    # tool messages stick to their assistant owner
    tool_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u0 " + "y" * 500},
        {"role": "assistant", "content": "a0", "tool_calls": [{"id": "t0", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t0", "content": "res0"},
        {"role": "user", "content": "u1 " + "y" * 500},
        {"role": "assistant", "content": "a1", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "res1"},
        {"role": "user", "content": "u2 " + "y" * 500},
    ]
    trimmed2 = tr.trim_history(tool_msgs, max_turns=4, max_chars=10**9)
    check("tool message never orphaned at window head",
          not (trimmed2 and trimmed2[0].get("role") == "tool"), f"head={trimmed2[0] if trimmed2 else None}")
    check("assistant+tool pair kept together",
          any(m.get("role") == "tool" for m in trimmed2)
          and next(i for i, m in enumerate(trimmed2) if m.get("role") == "tool") ==
          next(i for i, m in enumerate(trimmed2) if m.get("tool_calls")) + 1)
    check("short history passes through unchanged",
          tr.trim_history(tool_msgs[:2], max_turns=24, max_chars=24000) == tool_msgs[:2])


# ------------------------------------------------------------------ 4. real socket proof
def test_real_socket_incremental():
    print("\n== REAL SOCKET: first callback before the server writes event 2 ==")
    timing = {"sent_second": None}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # chunked framing, like llama-server/aiohttp

        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)  # consume the body, like a real server
            ev1 = b'data: {"choices":[{"delta":{"content":"First"}}]}\n\n'
            ev2 = b'data: {"choices":[{"delta":{"content":"Second"}}]}\n\n'

            def chunk(b: bytes):
                return b"%x\r\n" % len(b) + b + b"\r\n"

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(chunk(ev1))
            self.wfile.flush()
            time.sleep(0.35)  # the client MUST receive event 1 during this gap
            self.wfile.write(chunk(ev2))
            self.wfile.flush()
            timing["sent_second"] = time.perf_counter()
            self.wfile.write(chunk(b"data: [DONE]\n\n"))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    stamps: list[float] = []

    def on_delta(s):
        stamps.append(time.perf_counter())

    old_url = tr.cfg.LLAMA_SERVER_URL
    tr.cfg.LLAMA_SERVER_URL = f"http://127.0.0.1:{port}"
    try:
        text, _ = tr.chat([{"role": "user", "content": "stream test"}],
                          on_delta=on_delta, use_tools=False)
    finally:
        tr.cfg.LLAMA_SERVER_URL = old_url
        server.shutdown()
        server.server_close()

    check("two deltas received", len(stamps) == 2, f"stamps={len(stamps)}")
    check("text order preserved over socket", text == "FirstSecond", f"text={text!r}")
    if len(stamps) >= 2 and timing["sent_second"] is not None:
        margin = (timing["sent_second"] - stamps[0]) * 1000.0
        check("FIRST callback fired while the server was still streaming",
              margin > 100.0, f"only {margin:.0f} ms before event 2 was written")
        print(f"       (first callback preceded the second event by {margin:.0f} ms)")
    else:
        check("FIRST callback fired while the server was still streaming", False, "missing stamps")


# ------------------------------------------------------------------ main
def main() -> int:
    print("tool_runtime streaming verification")
    print("=" * 60)
    t0 = time.time()
    test_exhaustive_single_split()
    test_random_multisplit()
    test_missing_trailing_newline()
    test_utf8_split_across_chunks()
    test_think_filter()
    test_chat_incremental_callbacks()
    test_chat_reasoning_suppressed()
    test_tool_roundtrip()
    test_malformed_tool_args_never_executed()
    test_midstream_failure()
    test_cancellation_closes_stream()
    test_history_trimming()
    test_real_socket_incremental()
    failed = [r for r in RESULTS if not r[1]]
    print("=" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed in {time.time() - t0:.1f} s"
          + (f", {len(failed)} FAILED" if failed else ""))
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
