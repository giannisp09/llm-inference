"""Load a model YAML config and turn it into a `vllm serve` command line.

A config file describes one servable model: the HF model id, the served name, vLLM engine
flags, and default sampling params. Keeping engine flags in YAML (not hardcoded in scripts)
means adding a model is just dropping in a new file under configs/.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

# Maps engine YAML keys to vLLM CLI flags. Booleans become `--flag` (when true) and are
# omitted when false; everything else becomes `--flag value`.
_ENGINE_FLAGS = {
    "dtype": "--dtype",
    "max_model_len": "--max-model-len",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "tensor_parallel_size": "--tensor-parallel-size",
    "pipeline_parallel_size": "--pipeline-parallel-size",
    "quantization": "--quantization",
    "enable_prefix_caching": "--enable-prefix-caching",
    "enable_chunked_prefill": "--enable-chunked-prefill",
    "max_num_seqs": "--max-num-seqs",
    "kv_cache_dtype": "--kv-cache-dtype",
}


@dataclass
class ModelConfig:
    model: str
    served_model_name: str
    engine: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)

    def serve_command(self, host: str = "0.0.0.0", port: int = 8000) -> list[str]:
        """Build the argv for `vllm serve`."""
        argv = ["vllm", "serve", self.model]
        if self.served_model_name:
            argv += ["--served-model-name", self.served_model_name]
        argv += ["--host", host, "--port", str(port)]

        for key, value in self.engine.items():
            flag = _ENGINE_FLAGS.get(key)
            if flag is None:
                # Unknown keys pass through as --kebab-case-key value.
                flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            else:
                argv += [flag, str(value)]
        return argv


def load_config(name_or_path: str) -> ModelConfig:
    """Load a config by name (configs/<name>.yaml) or explicit path."""
    path = Path(name_or_path)
    if not path.exists():
        path = CONFIGS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"No config '{name_or_path}'. Available: {available or '(none)'}")
    data = yaml.safe_load(path.read_text())
    return ModelConfig(
        model=data["model"],
        served_model_name=data.get("served_model_name", data["model"]),
        engine=data.get("engine", {}),
        sampling=data.get("sampling", {}),
    )


def main() -> None:
    """`llm-serve <config>` — print the vllm serve command (used by scripts/serve.sh)."""
    parser = argparse.ArgumentParser(description="Print the vllm serve command for a model config.")
    parser.add_argument("config", help="config name (configs/<name>.yaml) or path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(shlex.join(cfg.serve_command(host=args.host, port=args.port)))


if __name__ == "__main__":
    main()
