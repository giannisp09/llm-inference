#!/usr/bin/env python
"""Continuous batching in action.

Send several requests concurrently and watch the live /metrics. vLLM schedules at the token
level: when a request finishes, its slot is immediately filled by the next one — no waiting
for the slowest request in a fixed batch.

Note: the mid-flight metric is sampled once, a fraction of a second after dispatch, so it can
read 0 if the requests haven't reached the scheduler yet. Run again to catch the true peak.
"""

import concurrent.futures
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI  # noqa: E402

from llm_inference.client import wait_for_server  # noqa: E402
from llm_inference.metrics import get_vllm_metrics  # noqa: E402

BASE_URL = "http://localhost:8000"

PROMPTS = [
    "What is quantization?",
    "Explain KV caching briefly.",
    "What is continuous batching?",
    "Why is LLM inference memory-bound?",
    "What is PagedAttention?",
]


def main() -> None:
    model = wait_for_server(BASE_URL)
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="unused")

    def ask(prompt: str):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    before = get_vllm_metrics(BASE_URL)
    print(f"\nSending {len(PROMPTS)} concurrent requests...\n")
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        futures = {pool.submit(ask, p): p for p in PROMPTS}
        time.sleep(0.5)
        during = get_vllm_metrics(BASE_URL)
        print(
            f"  [mid-flight]  running: {during.get('vllm:num_requests_running', '--')}  |  "
            f"waiting: {during.get('vllm:num_requests_waiting', '--')}"
        )
        for fut in concurrent.futures.as_completed(futures):
            resp = fut.result()
            print(f'  done: "{futures[fut][:40]}" -> {resp.usage.completion_tokens} tokens')

    elapsed = time.time() - start
    after = get_vllm_metrics(BASE_URL)
    tokens = after.get("vllm:generation_tokens_total", 0) - before.get(
        "vllm:generation_tokens_total", 0
    )
    print(f"\nAll {len(PROMPTS)} completed in {elapsed:.2f}s")
    if tokens > 0:
        print(f"Tokens generated: {tokens:g}  |  ~{tokens / elapsed:.1f} tokens/s")


if __name__ == "__main__":
    main()
