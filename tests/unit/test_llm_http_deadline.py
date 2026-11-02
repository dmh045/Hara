from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from unittest.mock import patch
import io
import json
import time

import pytest

from hara_agent.infrastructure.llm.openai_compatible import (
    OpenAICompatibleClient, TransientLLMError, LLMTransportError,
)


@contextmanager
def response_server(body, *, chunked, interval=0):
    stopped = Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                for byte in body:
                    part = bytes([byte])
                    self.wfile.write(b"1\r\n" + part + b"\r\n" if chunked else part)
                    self.wfile.flush()
                    if stopped.wait(interval):
                        return
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("chunked", [False, True])
def test_continuous_keepalive_bytes_do_not_extend_request_deadline(chunked):
    # The server keeps producing bytes more frequently than the socket timeout.
    # A buffered read(n) used to wait for the entire 1.5-second body here.
    with response_server(b" " * 150 + b"{}", chunked=chunked, interval=.01) as url:
        started = time.monotonic()
        with pytest.raises(TransientLLMError) as raised:
            OpenAICompatibleClient._http_transport(url, {}, b"{}", .1)
        elapsed = time.monotonic() - started
    assert raised.value.category == "timeout"
    assert elapsed < .6


@pytest.mark.parametrize("chunked", [False, True])
def test_incremental_body_read_preserves_complete_unicode_json(chunked):
    expected = {"choices": [{"message": {"content": "保留完整响应"}}], "usage": {"total_tokens": 7}}
    body = json.dumps(expected, ensure_ascii=False).encode("utf-8")
    with response_server(body, chunked=chunked) as url:
        assert OpenAICompatibleClient._http_transport(url, {}, b"{}", 1) == expected


@pytest.mark.parametrize("recovers", [False, True])
def test_invalid_http_json_uses_bounded_retry_and_preserves_rejected_body(tmp_path, recovers):
    from hara_agent.config import LLMConfig
    from hara_agent.infrastructure.llm.protocol import LLMRequest
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget
    from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint

    rejected = b'{"choices":['
    valid = json.dumps({"choices": [{"message": {"content": '{"ok":true}'},
                                    "finish_reason": "stop"}], "model": "fake"}).encode()
    budget = ProviderAttemptBudget(tmp_path / "attempts.jsonl", run_id="test", limit=2)
    client = OpenAICompatibleClient(LLMConfig(
        provider="openai-compatible", base_url="https://example.invalid/v1",
        model="fake", api_key="not-a-real-key", max_retries=1, retry_backoff_seconds=0,
    ), attempt_budget=budget)
    request = LLMRequest(task="test", system_prompt="system", user_prompt="user",
                         schema_name="Test", prompt_version="test")
    cache = ResponseCheckpoint(tmp_path / "cache")
    with patch("urllib.request.urlopen", side_effect=[io.BytesIO(rejected),
               io.BytesIO(valid if recovers else rejected)]) as send:
        if recovers:
            assert cache.complete(client, request).data == {"ok": True}
            assert cache.complete(client, request).data == {"ok": True}
        else:
            with pytest.raises(LLMTransportError) as error:
                cache.complete(client, request)
            assert error.value.category == "invalid_http_json"
            assert error.value.attempts == 2
            assert not list((tmp_path / "cache").glob("*.json"))
        assert send.call_count == budget.attempts == 2
    first = json.loads((tmp_path / "attempts.responses/000001.json").read_text())
    assert first["response"]["transport_error"]["raw_body"] == rejected.decode()
    assert "choices" not in first["response"]
    ledger = [json.loads(line) for line in budget.path.read_text().splitlines()]
    assert [e["outcome"] for e in ledger if e["event"] == "attempt_finished"] == [
        "error", "response" if recovers else "error"]
    assert "not-a-real-key" not in json.dumps(first)


def test_invalid_model_content_is_not_an_http_retry():
    from hara_agent.config import LLMConfig
    from hara_agent.infrastructure.llm.protocol import LLMRequest

    client = OpenAICompatibleClient(LLMConfig(
        provider="openai-compatible", base_url="https://example.invalid/v1",
        model="fake", api_key="fake", max_retries=1, retry_backoff_seconds=0,
    ))
    response = json.dumps({"choices": [{"message": {"content": "invalid model JSON"},
                                       "finish_reason": "stop"}]}).encode()
    with patch("urllib.request.urlopen", return_value=io.BytesIO(response)) as send:
        with pytest.raises(ValueError):
            client.complete_json(LLMRequest(task="test", system_prompt="system",
                user_prompt="user", schema_name="Test", prompt_version="test"))
        assert send.call_count == 1
