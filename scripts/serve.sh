#!/usr/bin/env bash
# Serve a model from configs/<name>.yaml with vLLM.
#
# Usage:
#   ./scripts/serve.sh qwen3-0.6b            # serve configs/qwen3-0.6b.yaml
#   ./scripts/serve.sh llama3-8b --port 8001
#
# Engine flags come from the YAML; this script just translates it into a `vllm serve` call.
# Uses `uv run` when available (respects .python-version → 3.12), else falls back to python.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${MODEL_CONFIG:-qwen3-0.6b}}"
shift || true

HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

# Pick a runner: uv (preferred) keeps us on the pinned interpreter + project env.
if command -v uv >/dev/null 2>&1 && [ -f "$REPO_ROOT/pyproject.toml" ]; then
  RUN=(uv run --project "$REPO_ROOT")
else
  RUN=(env PYTHONPATH="$REPO_ROOT/src" python)
fi

# Build the `vllm serve ...` command from the config (config.py knows the YAML->CLI mapping).
CMD="$(cd "$REPO_ROOT" && "${RUN[@]}" -m llm_inference.config "$CONFIG" --host "$HOST" --port "$PORT")"

echo ">>> $CMD $*"
# Run vllm through the same runner so it resolves from the project env.
exec env ${VLLM_API_KEY:+VLLM_API_KEY="$VLLM_API_KEY"} "${RUN[@]}" $CMD "$@"
