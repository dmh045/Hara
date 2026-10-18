from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from .state import HARAState


class CheckpointRepository:
    """Persist workflow state atomically for resume and audit."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        safe_id = "".join(char for char in run_id if char.isalnum() or char in "-_")
        if not safe_id or safe_id != run_id:
            raise ValueError("run_id只能包含字母、数字、连字符和下划线")
        return self.run_dir / f"{safe_id}.checkpoint.json"

    def save(self, state: HARAState) -> Path:
        target = self.path_for(state.run_id)
        # Large full-run states need the fast encoder; whitespace is not state.
        payload = json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":"))
        handle, temp_name = tempfile.mkstemp(
            prefix=f".{state.run_id}.", suffix=".tmp", dir=str(self.run_dir), text=True
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(5):
                try:
                    os.replace(temp_name, target)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return target

    def load(self, run_id: str) -> HARAState:
        path = self.path_for(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint不存在: {path}")
        return HARAState.from_dict(json.loads(path.read_text(encoding="utf-8")))
