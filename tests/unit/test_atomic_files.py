"""Audit-file replacement stays complete under failures and concurrent writers."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event

import pytest

from hara_agent.infrastructure.atomic_files import write_json_atomic, write_text_atomic


def test_json_keeps_unicode_precision_and_trailing_newline(tmp_path):
    target = tmp_path / "nested" / "trace.json"
    payload = {"场景": "倒车", "value": 0.154286, "unknown": None,
               "status": "PENDING", "sources": ["item:paragraph[2]"]}
    write_json_atomic(target, payload)
    assert target.read_bytes() == (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    assert not list(target.parent.glob(".*.tmp"))


def test_text_keeps_exact_content_without_adding_newline(tmp_path):
    target = tmp_path / "audit.md"
    write_text_atomic(target, "第一行\n第二行")
    assert target.read_bytes() == "第一行\n第二行".encode()


@pytest.mark.parametrize("writer,payload", [(write_json_atomic, {"value": 1}),
                                           (write_text_atomic, "new")])
def test_replace_failure_preserves_previous_file_and_cleans_temporary(tmp_path, monkeypatch, writer, payload):
    import hara_agent.infrastructure.atomic_files as module
    target = tmp_path / "audit.json"
    target.write_bytes(b"previous-complete-file")

    def fail_replace(source, destination):
        assert Path(source).parent == target.parent
        assert Path(source).is_file()
        assert destination == target
        raise OSError("replacement unavailable")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement unavailable"):
        writer(target, payload)
    assert target.read_bytes() == b"previous-complete-file"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audit.json"]


def test_concurrent_writes_never_expose_partial_json(tmp_path):
    target = tmp_path / "shared.json"
    payloads = [{"writer": i, "body": str(i) * 10000} for i in range(8)]
    write_json_atomic(target, payloads[0])
    stopped = Event()

    def read_during_writes():
        while not stopped.is_set():
            assert json.loads(target.read_text()) in payloads

    def write_many(payload):
        for _ in range(12):
            write_json_atomic(target, payload)

    with ThreadPoolExecutor(max_workers=9) as pool:
        reader = pool.submit(read_during_writes)
        try:
            for task in [pool.submit(write_many, payload) for payload in payloads]:
                task.result()
        finally:
            stopped.set()
        reader.result()
    assert json.loads(target.read_text()) in payloads
    assert not list(tmp_path.glob(".*.tmp"))
