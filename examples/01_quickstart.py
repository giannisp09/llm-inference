#!/usr/bin/env python
"""Your first local LLM request against the vLLM server.

Start a server first:  ./scripts/serve.sh qwen3-0.6b
Then run:              python examples/01_quickstart.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI  # noqa: E402

from llm_inference.client import wait_for_server  # noqa: E402

BASE_URL = "http://localhost:8000"


def main() -> None:
    model = wait_for_server(BASE_URL)
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="unused")

    # Qwen3 supports a "thinking" mode (chain-of-thought). Disable it for short answers.
    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is PagedAttention in one sentence?"}],
        max_tokens=80,
        temperature=0.7,
        top_p=0.8,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed = time.time() - start

    print(f"\nResponse ({elapsed:.2f}s, {resp.usage.completion_tokens} tokens):")
    print(resp.choices[0].message.content)
    print(
        f"\nUsage: {resp.usage.prompt_tokens} prompt + "
        f"{resp.usage.completion_tokens} completion = {resp.usage.total_tokens} total"
    )


if __name__ == "__main__":
    main()
