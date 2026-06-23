"""Thin convenience wrapper over the OpenAI SDK pointed at a vLLM server.

vLLM is OpenAI-compatible, so the stock `openai` client works directly. This adds two things
worth not rewriting in every script: a `wait_for_server` readiness poll and a `chat` helper
that injects the Qwen-style `enable_thinking` toggle.
"""

from __future__ import annotations

import time

import requests
from openai import OpenAI


def wait_for_server(base_url: str = "http://localhost:8000", timeout: float = 300.0) -> str:
    """Block until the server answers /v1/models; return the served model id.

    Raises RuntimeError if the server is not reachable within `timeout` seconds.
    """
    deadline = time.time() + timeout
    print(f"Waiting for vLLM server at {base_url} ...")
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                model_id = r.json()["data"][0]["id"]
                print(f"Connected — model: {model_id}")
                return model_id
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(
        f"vLLM server not reachable at {base_url} after {timeout:.0f}s. "
        f"Start one with: ./scripts/serve.sh qwen3-0.6b"
    )


class InferenceClient:
    """Minimal client: discovers the model id and exposes a `chat` helper."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "unused"):
        self.base_url = base_url.rstrip("/")
        self.openai = OpenAI(base_url=f"{self.base_url}/v1", api_key=api_key)
        self.model: str | None = None

    def ready(self, timeout: float = 300.0) -> InferenceClient:
        self.model = wait_for_server(self.base_url, timeout=timeout)
        return self

    def chat(self, prompt: str, *, thinking: bool = False, **kwargs):
        """One-shot chat completion. Extra kwargs pass straight to the OpenAI call."""
        if self.model is None:
            self.ready()
        extra_body = kwargs.pop("extra_body", {})
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = thinking
        return self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            extra_body=extra_body,
            **kwargs,
        )
