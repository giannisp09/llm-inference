"""Smoke test against a live server. Skipped unless one is reachable.

Run with a server up:  ./scripts/serve.sh qwen3-0.6b  (in another terminal)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

BASE_URL = "http://localhost:8000"


def _server_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/v1/models", timeout=2).status_code == 200
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason="no vLLM server on :8000")


def test_chat_completion_returns_text():
    from llm_inference.client import InferenceClient

    client = InferenceClient(BASE_URL).ready(timeout=10)
    resp = client.chat("Say hello in one word.", max_tokens=10, temperature=0.0)
    assert resp.choices[0].message.content.strip()


def test_metrics_endpoint_parses():
    from llm_inference.metrics import get_vllm_metrics

    metrics = get_vllm_metrics(BASE_URL)
    assert any(k.startswith("vllm:") for k in metrics)
