"""Run-local response persistence; restored JSON still goes through semantic validation."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock

from .protocol import LLMResponse

_locks_guard = Lock()
_request_locks = {}


def _request_lock(path):
    with _locks_guard:
        return _request_locks.setdefault(str(path.resolve()), Lock())


class ResponseCheckpoint:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def complete(self, client, request):
        config = getattr(client, "config", None)
        # Full prompt/schema identity includes source facts, method and scenario
        # IDs. Credentials are neither part of the key nor written to disk.
        effective_request = asdict(request)
        # Batch position is transport tracing, not analysis input. A completed
        # singleton remains valid when another work item is added on resume.
        effective_request["metadata"] = {key: value for key, value in effective_request["metadata"].items()
            if key not in {"parent_batch", "split_path", "split_depth"}}
        material = {"request": effective_request, "route": {
            key: getattr(config, key, "") for key in ("provider", "base_url", "model", "scenario_thinking")
        }}
        key = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":"), default=str).encode()).hexdigest()
        path = self.directory / f"{key}.json"
        # Same in-flight request in concurrent workers must not pay twice on a
        # cache miss. Atomic rename below still protects interrupted writes.
        with _request_lock(path):
            return self._complete_locked(client, request, path, key)

    def _complete_locked(self, client, request, path, key):
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["request_hash"] != key:
                raise ValueError("Scenario response checkpoint identity mismatch")
            response = LLMResponse(**saved["response"])
            return replace(response, usage={"response_checkpoint_hit": True,
                                            "prompt_tokens": 0, "completion_tokens": 0})
        response = client.complete_json(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"request_hash": key, "task": request.task,
                           "metadata": request.metadata, "response": asdict(response)},
                          stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return response
