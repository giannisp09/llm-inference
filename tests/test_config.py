"""Unit tests for the config loader — no server required."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference.config import load_config  # noqa: E402


def test_qwen_config_loads():
    cfg = load_config("qwen3-0.6b")
    assert cfg.model == "Qwen/Qwen3-0.6B"
    assert cfg.served_model_name == "qwen3-0.6b"
    assert cfg.engine["max_model_len"] == 4096


def test_serve_command_builds_expected_flags():
    cfg = load_config("qwen3-0.6b")
    cmd = cfg.serve_command(host="0.0.0.0", port=8000)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-0.6B"]
    assert "--max-model-len" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"
    # Boolean true flag is present with no value.
    assert "--enable-prefix-caching" in cmd
    assert "--served-model-name" in cmd


def test_boolean_false_flag_is_omitted():
    cfg = load_config("qwen3-0.6b")
    cfg.engine["enable_prefix_caching"] = False
    assert "--enable-prefix-caching" not in cfg.serve_command()


def test_unknown_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist")
