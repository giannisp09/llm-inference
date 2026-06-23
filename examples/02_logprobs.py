#!/usr/bin/env python
"""Inspect the model's confidence with logprobs.

`logprobs=True` + `top_logprobs=N` returns, for each generated token, the chosen token's
log-probability and the top-N alternatives it considered.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI  # noqa: E402

from llm_inference.client import wait_for_server  # noqa: E402

BASE_URL = "http://localhost:8000"


def main() -> None:
    model = wait_for_server(BASE_URL)
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="unused")

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "The capital of France is"}],
        max_tokens=15,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    print(f"\nResponse: {resp.choices[0].message.content.strip()}\n")
    print("Token-by-token probabilities:\n")
    for tok in resp.choices[0].logprobs.content[:8]:
        print(f"  Chosen: '{tok.token}'  (logprob {tok.logprob:.2f})")
        for alt in (tok.top_logprobs or [])[:5]:
            pct = math.exp(alt.logprob) * 100
            bar = "█" * min(20, max(1, int(pct / 5)))
            print(f"    {pct:5.1f}%  {bar}  '{alt.token}'")
        print()


if __name__ == "__main__":
    main()
