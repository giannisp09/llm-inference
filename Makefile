.PHONY: help install serve quickstart bench test lint fmt docker-up docker-down clean

CONFIG ?= qwen3-0.6b

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with dev extras
	pip install -e ".[dev]"

serve:  ## Serve a model: make serve CONFIG=qwen3-0.6b
	./scripts/serve.sh $(CONFIG)

quickstart:  ## Run the quickstart example against a running server
	python examples/01_quickstart.py

bench:  ## Benchmark a running server
	python scripts/benchmark.py --concurrency 32 --num-requests 256

test:  ## Run tests (smoke tests need a live server)
	pytest

lint:  ## Lint with ruff
	ruff check .

fmt:  ## Format with ruff
	ruff format .

docker-up:  ## Bring up vLLM + Prometheus + Grafana
	cd docker && docker compose up

docker-down:  ## Tear down the stack
	cd docker && docker compose down

clean:  ## Remove caches and outputs
	rm -rf outputs .pytest_cache .ruff_cache **/__pycache__
