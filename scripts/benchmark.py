#!/usr/bin/env python
"""Load generator for a running vLLM server.

Fires N chat requests at a target concurrency and reports throughput (tokens/s), TTFT
(time-to-first-token, via streaming), and end-to-end latency percentiles.

    python scripts/benchmark.py --url http://localhost:8000 --concurrency 32 --num-requests 256
"""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference.client import wait_for_server  # noqa: E402
from openai import OpenAI  # noqa: E402

PROMPT = "Explain why LLM inference is memory-bound, in two sentences."


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100 * (len(s) - 1)))))
    return s[k]


def _one_request(client: OpenAI, model: str, max_tokens: int) -> tuple[float, float, int]:
    """Return (end_to_end_seconds, ttft_seconds, completion_tokens)."""
    start = time.time()
    ttft = None
    tokens = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.time() - start
            tokens += 1
    return time.time() - start, (ttft or 0.0), tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a vLLM server.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    model = wait_for_server(args.url)
    client = OpenAI(base_url=f"{args.url}/v1", api_key="unused")

    print(f"\nFiring {args.num_requests} requests at concurrency {args.concurrency} ...")
    latencies: list[float] = []
    ttfts: list[float] = []
    total_tokens = 0

    wall_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(_one_request, client, model, args.max_tokens)
            for _ in range(args.num_requests)
        ]
        for fut in concurrent.futures.as_completed(futures):
            e2e, ttft, tokens = fut.result()
            latencies.append(e2e)
            ttfts.append(ttft)
            total_tokens += tokens
    wall = time.time() - wall_start

    print("\n--- Results ---")
    print(f"  Requests:          {args.num_requests}")
    print(f"  Wall time:         {wall:.2f}s")
    print(f"  Throughput:        {total_tokens / wall:.1f} tokens/s")
    print(f"  Mean latency:      {statistics.mean(latencies):.2f}s")
    print(f"  p50 / p95 / p99:   "
          f"{_percentile(latencies, 50):.2f} / "
          f"{_percentile(latencies, 95):.2f} / "
          f"{_percentile(latencies, 99):.2f}s")
    print(f"  TTFT p50 / p95:    "
          f"{_percentile(ttfts, 50):.3f} / {_percentile(ttfts, 95):.3f}s")


if __name__ == "__main__":
    main()
