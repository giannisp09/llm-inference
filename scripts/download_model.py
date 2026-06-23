#!/usr/bin/env python
"""Pre-download a model's weights into the HF cache (so the first request isn't slow).

Useful for baking weights into a container image or warming a node before it serves traffic.

    python scripts/download_model.py Qwen/Qwen3-0.6B
    python scripts/download_model.py --config qwen3-0.6b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights into the HF cache.")
    parser.add_argument("model", nargs="?", help="HF model id (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--config", help="resolve the model id from configs/<name>.yaml")
    args = parser.parse_args()

    if args.config:
        from llm_inference.config import load_config

        model_id = load_config(args.config).model
    elif args.model:
        model_id = args.model
    else:
        parser.error("provide a model id or --config")

    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id} ...")
    path = snapshot_download(repo_id=model_id)
    print(f"Done: {path}")


if __name__ == "__main__":
    main()
