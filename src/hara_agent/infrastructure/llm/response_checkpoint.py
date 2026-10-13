"""Run-local response persistence; restored JSON still goes through semantic validation."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .protocol import LLMResponse


class ResponseCheckpoint:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def complete(self, client, request):
        config = getattr(client, "config", None)
        # Full prompt/schema identity includes source facts, method and scenario
        # IDs. Credentials are neither part of the key nor written to disk.
        material = {"request": asdict(request), "route": {
            key: getattr(config, key, "") for key in ("provider", "base_url", "model", "scenario_thinking")
        }}
        key = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":"), default=str).encode()).hexdigest()
        path = self.directory / f"{key}.json"
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
