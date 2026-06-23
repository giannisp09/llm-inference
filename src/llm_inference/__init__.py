"""llm-inference: production-ready LLM serving with vLLM (OpenAI-compatible API).

Heavy imports (openai, requests) are loaded lazily so that the config path used by
scripts/serve.sh works with only pyyaml installed.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "InferenceClient",
    "wait_for_server",
    "ModelConfig",
    "load_config",
    "get_vllm_metrics",
]

__version__ = "0.1.0"

# attribute -> submodule it lives in
_LAZY = {
    "InferenceClient": "client",
    "wait_for_server": "client",
    "ModelConfig": "config",
    "load_config": "config",
    "get_vllm_metrics": "metrics",
}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{mod}", __name__), name)


if TYPE_CHECKING:  # for type checkers / IDEs only
    from .client import InferenceClient, wait_for_server
    from .config import ModelConfig, load_config
    from .metrics import get_vllm_metrics
