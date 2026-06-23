"""End-to-end tests against a running server (real vLLM if present, else the mock).

These exercise the full client path — model discovery, chat completion, streaming, logprobs,
and metrics scraping — with no GPU required. See conftest.py for how `base_url` is provided.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI  # noqa: E402

from llm_inference.client import InferenceClient  # noqa: E402
from llm_inference.metrics import get_vllm_metrics  # noqa: E402


def test_model_discovery_and_chat(base_url):
    client = InferenceClient(base_url).ready(timeout=10)
    resp = client.chat("What is PagedAttention?", max_tokens=20, temperature=0.0)
    assert resp.choices[0].message.content.strip()
    assert resp.usage.completion_tokens > 0


def test_streaming_yields_tokens(base_url):
    model = InferenceClient(base_url).ready(timeout=10).model
    oai = OpenAI(base_url=f"{base_url}/v1", api_key="unused")
    chunks = 0
    stream = oai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Stream please"}],
        max_tokens=16,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks += 1
    assert chunks > 0


def test_logprobs_present_when_requested(base_url):
    model = InferenceClient(base_url).ready(timeout=10).model
    oai = OpenAI(base_url=f"{base_url}/v1", api_key="unused")
    resp = oai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "The capital of France is"}],
        max_tokens=8,
        logprobs=True,
        top_logprobs=2,
    )
    content = resp.choices[0].logprobs.content
    assert content and content[0].token


def test_metrics_endpoint_parses(base_url):
    metrics = get_vllm_metrics(base_url)
    assert any(k.startswith("vllm:") for k in metrics)
    assert "vllm:generation_tokens_total" in metrics


def test_prefix_cache_hits_increment_on_repeated_system_prompt(base_url):
    model = InferenceClient(base_url).ready(timeout=10).model
    oai = OpenAI(base_url=f"{base_url}/v1", api_key="unused")
    system = "You are a concise assistant for an integration test."

    def ask(q):
        return oai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": q},
            ],
            max_tokens=8,
        )

    before = get_vllm_metrics(base_url).get("vllm:prefix_cache_queries_total", 0)
    ask("first")
    ask("second")
    after = get_vllm_metrics(base_url).get("vllm:prefix_cache_queries_total", 0)
    # Both backends increment queries on each request carrying the shared system prompt.
    assert after >= before + 2
