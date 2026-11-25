"""Atomic UTF-8 audit files shared by the CLI and child-run tools.

These writers retain the existing audit format and replacement behavior.
Checkpoint, review-journal and contract persistence keep their own durability
and serialization requirements.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.stem}-", suffix=".tmp", delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.stem}-", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
