from __future__ import annotations

import sys
import time

from hara_agent.services.extraction import DocumentReader, select_source_blocks
from hara_agent.workflow.state import HARAState, WorkflowStage


def read_item_document(state: HARAState, path: str,
                       reader: DocumentReader | None = None,
                       project_analysis_policy: dict | None = None) -> HARAState:
    started = time.monotonic()
    artifact = (reader or DocumentReader()).read(path)
    selection = select_source_blocks(artifact.blocks, project_analysis_policy)
    elapsed_seconds = time.monotonic() - started

    def serialize(block):
        return {
            "block_id": block.block_id,
            "kind": block.kind,
            "location": block.location,
            "text": block.text,
            "section_path": list(block.section_path),
        }

    state.item_definition = {
        "source_id": artifact.source_id,
        "source_path": str(artifact.source_path),
        "raw_blocks": [serialize(block) for block in artifact.blocks],
        "blocks": [serialize(block) for block in selection.selected_blocks],
        "text": "\n".join(block.text for block in selection.selected_blocks if block.text),
        "source_selection": selection.to_dict(),
    }
    state.stage = WorkflowStage.EXTRACT
    state.record(
        "item_document_read",
        source_id=artifact.source_id,
        block_count=len(selection.selected_blocks),
        raw_block_count=len(artifact.blocks),
        excluded_block_count=len(selection.excluded_blocks),
        source_selection_fingerprint=selection.fingerprint,
        character_count=len(state.item_definition["text"]),
        elapsed_seconds=round(elapsed_seconds, 3),
    )
    print(
        "[HARA] document parse completed "
        f"blocks={len(selection.selected_blocks)} input_chars={len(state.item_definition['text'])} "
        f"elapsed={elapsed_seconds:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return state
