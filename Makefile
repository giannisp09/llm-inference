.PHONY: help install serve quickstart bench test lint fmt docker-up docker-down clean

CONFIG ?= qwen3-0.6b

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the uv env (Python 3.12) with dev extras
	uv sync --extra dev

install-serve:  ## Also install the vLLM engine (Linux + GPU only)
	uv sync --extra dev --extra serve

serve:  ## Serve a model: make serve CONFIG=qwen3-0.6b
	./scripts/serve.sh $(CONFIG)

quickstart:  ## Run the quickstart example against a running server
	uv run examples/01_quickstart.py

bench:  ## Benchmark a running server
	uv run scripts/benchmark.py --concurrency 32 --num-requests 256

test:  ## Run tests (smoke tests need a live server)
	uv run pytest

lint:  ## Lint with ruff
	uv run ruff check .

fmt:  ## Format with ruff
	uv run ruff format .

docker-up:  ## Bring up vLLM + Prometheus + Grafana
	cd docker && docker compose up

docker-down:  ## Tear down the stack
	cd docker && docker compose down

clean:  ## Remove caches and outputs
	rm -rf outputs .pytest_cache .ruff_cache **/__pycache__
