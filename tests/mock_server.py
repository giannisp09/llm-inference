"""A tiny stdlib OpenAI-compatible mock of the vLLM server — no GPU, no model.

It implements just enough of the contract (`/v1/models`, `/v1/chat/completions` with streaming
and logprobs, `/health`, and a Prometheus `/metrics` endpoint with `vllm:` counters that increment
per request) to exercise *our* code — the client, metrics scraper, examples, and benchmark — on any
machine, including macOS and CI.

It validates the integration glue, NOT the engine. Real inference correctness/performance is a
property of vLLM and is demonstrated separately on a GPU (see the Colab demo).

Run standalone:  python tests/mock_server.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "mock/qwen3-0.6b"

# A canned reply, tokenized crudely so streaming has multiple chunks and usage looks real.
_REPLY = (
    "PagedAttention stores the KV cache in fixed-size blocks so memory is used efficiently "
    "with almost no waste."
)


class _State:
    """Process-wide counters surfaced via /metrics (mimics vLLM's gauges/counters)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = 0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.prefix_cache_queries = 0
        self.prefix_cache_hits = 0
        self._seen_prefixes: set[int] = set()

    def on_request(self, messages: list[dict], completion_tokens: int) -> None:
        with self.lock:
            self.prompt_tokens += sum(len(m.get("content", "").split()) for m in messages)
            self.generation_tokens += completion_tokens
            # Model prefix caching: a system prompt seen before counts as a hit.
            system = next((m["content"] for m in messages if m["role"] == "system"), None)
            if system is not None:
                self.prefix_cache_queries += 1
                h = hash(system)
                if h in self._seen_prefixes:
                    self.prefix_cache_hits += 1
                else:
                    self._seen_prefixes.add(h)


STATE = _State()


def _tokens() -> list[str]:
    return [t + " " for t in _REPLY.split()]


def _logprobs_payload(tokens: list[str]) -> dict:
    """Minimal but client-parseable logprobs structure."""
    content = []
    for t in tokens:
        content.append(
            {
                "token": t,
                "logprob": -0.12,
                "top_logprobs": [
                    {"token": t, "logprob": -0.12},
                    {"token": " the", "logprob": -2.30},
                ],
            }
        )
    return {"content": content}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence per-request logging
        pass

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/v1/models":
            self._send_json({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        elif self.path == "/metrics":
            self._send_metrics()
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        messages = req.get("messages", [])
        max_tokens = int(req.get("max_tokens", 64))
        tokens = _tokens()[:max_tokens]

        STATE.on_request(messages, len(tokens))
        if req.get("stream"):
            self._send_stream(tokens)
        else:
            self._send_completion(tokens, want_logprobs=bool(req.get("logprobs")))

    def _send_completion(self, tokens: list[str], want_logprobs: bool) -> None:
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": "".join(tokens).strip()},
            "finish_reason": "stop",
        }
        if want_logprobs:
            choice["logprobs"] = _logprobs_payload(tokens)
        self._send_json(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": MODEL_ID,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": len(tokens),
                    "total_tokens": 21 + len(tokens),
                },
            }
        )

    def _send_stream(self, tokens: list[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        for tok in tokens:
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.002)  # tiny delay so TTFT/streaming timing is measurable
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_metrics(self) -> None:
        with STATE.lock:
            text = (
                f"# vLLM mock metrics\n"
                f"vllm:num_requests_running {STATE.running}\n"
                f"vllm:num_requests_waiting 0\n"
                f"vllm:gpu_cache_usage_perc 0.0\n"
                f"vllm:cpu_cache_usage_perc 0.0\n"
                f"vllm:prompt_tokens_total {STATE.prompt_tokens}\n"
                f"vllm:generation_tokens_total {STATE.generation_tokens}\n"
                f"vllm:prefix_cache_queries_total {STATE.prefix_cache_queries}\n"
                f"vllm:prefix_cache_hits_total {STATE.prefix_cache_hits}\n"
            )
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_in_thread(port: int = 8000) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the mock in a daemon thread; return (server, thread). server.shutdown() stops it."""
    server = make_server(port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI-compatible mock vLLM server.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"Mock vLLM server on http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
