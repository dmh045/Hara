from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
import json
import time

import pytest

from hara_agent.infrastructure.llm.openai_compatible import (
    OpenAICompatibleClient, TransientLLMError,
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
