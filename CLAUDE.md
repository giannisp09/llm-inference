# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this is

A production-ready **LLM inference serving** template built on [vLLM](https://github.com/vllm-project/vllm).
The goal is to host *any* open-weight model behind a fast, OpenAI-compatible API the way large labs and
products do it — with continuous batching, PagedAttention, prefix caching, quantization support,
containerization, autoscaling, and observability.

This is a **serving** repo, not a training repo. Weights are downloaded/loaded, not trained here.
(Sibling projects `../nanochat` and `../llm-scaling-ladder` cover training/scaling.)

## Architecture in one breath

`configs/<model>.yaml` → `scripts/serve.sh` builds the `vllm serve` command → vLLM exposes the
OpenAI HTTP API on `:8000` + Prometheus `/metrics`. Clients use the stock `openai` SDK (or the thin
wrapper in `src/llm_inference/client.py`). Docker Compose adds Prometheus + Grafana for observability.
Kubernetes manifests in `deploy/k8s/` run it for real with GPU scheduling and an HPA.

## Layout & where things live

- `configs/` — one YAML per servable model. **To add a model, add a config file** (do not hardcode
  model names in scripts). `src/llm_inference/config.py` loads these into the `vllm serve` argv.
- `src/llm_inference/` — small Python package:
  - `config.py` — load a model YAML, turn it into engine CLI args.
  - `client.py` — convenience wrapper over the OpenAI SDK + a `wait_for_server` helper.
  - `metrics.py` — scrape and parse the Prometheus `/metrics` endpoint.
- `scripts/serve.sh` — the canonical way to start a server from a config.
- `scripts/benchmark.py` — load generator reporting throughput / TTFT / latency percentiles.
- `examples/` — runnable, self-contained demos. Keep them dependency-light (just `openai` + stdlib).
- `docker/` — `Dockerfile` (CUDA base) + `docker-compose.yml` (vLLM + Prometheus + Grafana).
- `deploy/` — Kubernetes manifests + monitoring config.

## Conventions

- **Engine flags live in configs, not in code.** If you need a new vLLM option, surface it in the
  YAML schema (`config.py`) rather than special-casing it in `serve.sh`.
- **OpenAI-compatible only.** Clients should never depend on vLLM-internal request shapes; route
  everything through `/v1/*`. vLLM-specific knobs go in `extra_body` (e.g. `top_k`,
  `chat_template_kwargs`).
- **Examples must run against a live server on `localhost:8000`** and degrade gracefully if it's
  down (print a clear "start the server first" message — see `client.wait_for_server`).
- Keep examples thin and readable; they double as documentation.
- Python 3.10–3.12 (vLLM has no 3.13+ wheels yet). The host here runs 3.14 — use a 3.12 venv for
  anything that imports vLLM.

## Common commands

```bash
./scripts/serve.sh qwen3-0.6b              # serve a model from its config
python examples/01_quickstart.py           # smoke-test the API
python scripts/benchmark.py --concurrency 32
cd docker && docker compose up             # full stack with monitoring
pytest -q                                  # tests (smoke tests need a live server)
ruff check . && ruff format .              # lint + format
```

## Key serving metrics (from `/metrics`)

- `vllm:num_requests_running` / `vllm:num_requests_waiting` — active vs queued.
- `vllm:gpu_cache_usage_perc` / `vllm:cpu_cache_usage_perc` — KV cache pressure.
- `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` — cumulative throughput.
- `vllm:prefix_cache_queries_total` / `vllm:prefix_cache_hits_total` — prefix-cache effectiveness.
- `vllm:time_to_first_token_seconds` — TTFT histogram.

The HPA in `deploy/k8s/hpa.yaml` scales on `num_requests_waiting`.

## When making changes

- Adding a model → new `configs/<name>.yaml`, nothing else required.
- New example → add to `examples/`, list it in the README table, keep it runnable standalone.
- Changing engine behavior → update the config schema + `serve.sh`, and note GPU/CPU implications.
- Don't commit model weights, `outputs/`, or `~/.cache/huggingface` artifacts (see `.gitignore`).
