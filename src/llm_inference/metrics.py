"""Scrape and parse vLLM's Prometheus `/metrics` endpoint.

vLLM exposes a Prometheus text-format endpoint. This is a dependency-light parser that
returns a flat {metric_name: value} dict — enough to watch KV cache pressure, queue depth,
and throughput without pulling in a full Prometheus client.
"""

from __future__ import annotations

import requests

# The metrics worth watching at a glance.
KEY_METRICS = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
]


def get_vllm_metrics(
    base_url: str = "http://localhost:8000", timeout: float = 5.0
) -> dict[str, float]:
    """Scrape `/metrics` and return {name: value}.

    For metrics with labels, the last-seen sample for a given name wins. That is fine for the
    aggregate gauges/counters we care about; use a real Prometheus client for label-aware queries.
    """
    resp = requests.get(f"{base_url}/metrics", timeout=timeout)
    resp.raise_for_status()
    metrics: dict[str, float] = {}
    for line in resp.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{")[0].split()[0]
        try:
            metrics[name] = float(line.split()[-1])
        except (ValueError, IndexError):
            continue
    return metrics


def print_key_metrics(base_url: str = "http://localhost:8000") -> dict[str, float]:
    """Print the headline serving metrics and return the full dict."""
    metrics = get_vllm_metrics(base_url)
    print("Current vLLM metrics:")
    for key in KEY_METRICS:
        if key in metrics:
            print(f"  {key.replace('vllm:', ''):<28} {metrics[key]:g}")
    return metrics
