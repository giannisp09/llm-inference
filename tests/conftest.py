"""Shared fixtures. Provides a `base_url` pointing at a running OpenAI-compatible server.

If a real vLLM server is up on :8000 we use it; otherwise we spin up the stdlib mock
(tests/mock_server.py) on an ephemeral port. Either way the integration tests run — no GPU
required — so the client, metrics scraper, and example logic are always exercised.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _real_server_up() -> bool:
    try:
        return requests.get("http://localhost:8000/v1/models", timeout=1).status_code == 200
    except requests.RequestException:
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def base_url():
    if _real_server_up():
        yield "http://localhost:8000"
        return

    from mock_server import serve_in_thread

    port = _free_port()
    server, _ = serve_in_thread(port)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
