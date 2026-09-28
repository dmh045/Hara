"""Durable, per-run limit on actual Provider transport attempts.

The ledger deliberately contains request metadata only.  Prompts, response
content, endpoints, and credentials must never be written here.
"""

from __future__ import annotations

import json
import fcntl
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import LLMRequest


class ProviderAttemptBudgetExceeded(RuntimeError):
    def __init__(self, *, limit: int, attempts: int, ledger_path: Path):
        super().__init__(
            f"Provider attempt budget exhausted: {attempts}/{limit}; "
            f"ledger={ledger_path}"
        )
        self.limit = limit
        self.attempts = attempts
        self.ledger_path = ledger_path


class ProviderAttemptBudget:
    """Reserve each HTTP attempt before sending it, including retries.

    A reserved attempt counts even if the process exits before a response.
    This favors a hard upper bound over optimistic accounting on resume.
    """

    def __init__(self, path: str | Path, *, run_id: str, limit: int):
        if limit < 1:
            raise ValueError("Provider attempt limit must be positive")
        self.path = Path(path).expanduser().resolve()
        self.run_id = run_id
        self.limit = limit
        self._lock = threading.Lock()
        self._attempts = 0
        if self.path.is_file():
            for raw in self.path.read_text(encoding="utf-8").splitlines():
                event = json.loads(raw)
                if event.get("run_id") != run_id:
                    raise ValueError("Provider attempt ledger is bound to a different run")
                if event.get("event") == "attempt_started":
                    self._attempts += 1
        if self._attempts > limit:
            raise ValueError("Existing Provider attempts exceed the requested limit")

    @property
    def attempts(self) -> int:
        with self._lock:
            return self._attempts

    @staticmethod
    def _write_locked(descriptor: int, event: dict[str, Any]) -> None:
        payload = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        os.lseek(descriptor, 0, os.SEEK_END)
        if os.write(descriptor, payload) != len(payload):
            raise OSError("Incomplete Provider attempt ledger write")
        os.fsync(descriptor)

    def _append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._write_locked(descriptor, event)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def begin(self, request: LLMRequest, *, model: str) -> int:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                size = os.fstat(descriptor).st_size
                os.lseek(descriptor, 0, os.SEEK_SET)
                raw = bytearray()
                while len(raw) < size:
                    chunk = os.read(descriptor, size - len(raw))
                    if not chunk:
                        raise OSError("Provider attempt ledger changed during read")
                    raw.extend(chunk)
                attempts = 0
                for line in raw.decode("utf-8").splitlines():
                    event = json.loads(line)
                    if event.get("run_id") != self.run_id:
                        raise ValueError("Provider attempt ledger is bound to a different run")
                    attempts += event.get("event") == "attempt_started"
                self._attempts = attempts
                if attempts >= self.limit:
                    raise ProviderAttemptBudgetExceeded(
                        limit=self.limit, attempts=attempts,
                        ledger_path=self.path,
                    )
                attempt_id = attempts + 1
                self._write_locked(descriptor, {
                    "event": "attempt_started",
                    "run_id": self.run_id,
                    "attempt_id": attempt_id,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "task": request.task,
                    "schema_name": request.schema_name,
                    "model": model,
                })
                self._attempts = attempt_id
                return attempt_id
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def finish(self, attempt_id: int, *, response: dict[str, Any] | None = None,
               error: BaseException | None = None) -> None:
        choice = (response or {}).get("choices", [{}])
        first = choice[0] if isinstance(choice, list) and choice else {}
        with self._lock:
            self._append({
                "event": "attempt_finished",
                "run_id": self.run_id,
                "attempt_id": attempt_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "outcome": "error" if error is not None else "response",
                "error_type": type(error).__name__ if error is not None else "",
                "finish_reason": str(first.get("finish_reason", "")) if isinstance(first, dict) else "",
                "model": str((response or {}).get("model", "")),
            })
