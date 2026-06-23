#!/usr/bin/env python
"""Qwen3 thinking mode, streamed.

Qwen3 can emit internal chain-of-thought (<think>...</think>) before the visible answer.
Thinking ON tends to produce better answers but uses many more tokens (more KV cache, more
compute, longer latency). This streams both modes on the same prompt for comparison.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI  # noqa: E402

from llm_inference.client import wait_for_server  # noqa: E402

BASE_URL = "http://localhost:8000"
PROMPT = "What makes continuous batching better than static batching?"


def main() -> None:
    model = wait_for_server(BASE_URL)
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="unused")

    for label, thinking, max_tok in [("Thinking OFF", False, 80), ("Thinking ON", True, 200)]:
        print(f"\n=== {label} ===\n")
        start = time.time()
        tokens = 0
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tok,
            temperature=0.7,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                sys.stdout.write(chunk.choices[0].delta.content)
                sys.stdout.flush()
                tokens += 1
        print(f"\n  [{tokens} tokens, {time.time() - start:.2f}s]")


if __name__ == "__main__":
    main()
