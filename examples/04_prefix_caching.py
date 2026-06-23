#!/usr/bin/env python
"""Prefix caching: shared prompt prefixes reuse cached KV blocks.

Many apps reuse a long system prompt across requests. With prefix caching, the first request
pays the full prefill cost for the shared prefix; later requests skip it. The
`prefix_cache_queries_total` metric climbing confirms vLLM is checking/reusing cached blocks.

With a short system prompt the wall-clock savings are small, but in production (thousands of
tokens of instructions / few-shot examples) it eliminates a large chunk of per-request prefill.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference.client import wait_for_server  # noqa: E402
from llm_inference.metrics import get_vllm_metrics  # noqa: E402
from openai import OpenAI  # noqa: E402

BASE_URL = "http://localhost:8000"

SYSTEM_PROMPT = (
    "You are a helpful AI teaching assistant for a course on LLM optimization. You specialize "
    "in explaining concepts like quantization, inference optimization, and model serving. Keep "
    "answers concise -- one or two sentences."
)

QUESTIONS = [
    "What is weight quantization?",
    "How does vLLM handle memory?",
    "What is continuous batching?",
    "Why use prefix caching?",
    "What is GPTQ?",
]


def main() -> None:
    model = wait_for_server(BASE_URL)
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="unused")

    before = get_vllm_metrics(BASE_URL)
    print("\nSending 5 requests with the SAME system prompt...\n")
    for i, q in enumerate(QUESTIONS):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            max_tokens=60,
            temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        dt = time.time() - t0
        tokens = resp.usage.completion_tokens
        ms_per_tok = (dt / tokens * 1000) if tokens else 0.0
        print(f"  [{i + 1}] {q:<35} {dt:.2f}s  ({tokens} tok, {ms_per_tok:.0f} ms/tok)")

    after = get_vllm_metrics(BASE_URL)
    q_before = before.get("vllm:prefix_cache_queries_total", 0)
    q_after = after.get("vllm:prefix_cache_queries_total", 0)
    h_before = before.get("vllm:prefix_cache_hits_total", 0)
    h_after = after.get("vllm:prefix_cache_hits_total", 0)

    print(f"\nPrefix cache queries: {q_before:g} -> {q_after:g}  (+{q_after - q_before:g})")
    print(f"Prefix cache hits:    {h_before:g} -> {h_after:g}  (+{h_after - h_before:g})")
    print("\nThe rising query/hit counts confirm vLLM is reusing cached KV blocks for the")
    print("shared system prompt.")


if __name__ == "__main__":
    main()
