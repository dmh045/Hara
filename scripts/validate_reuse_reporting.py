"""Re-export immutable samples through the production rebuilder, without Provider."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copy2, copytree

from hara_agent.services.reporting.offline_rebuild import OfflineReportRebuilder


def run(source: Path, output: Path, previous_reports: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for kind in ("point", "range", "symbolic"):
        folder = source / kind
        checkpoint = next((folder / "checkpoints").glob("*.checkpoint.json"))
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        target = output / kind
        target.mkdir()
        original = previous_reports / kind / "HARA_Review.xlsx" if previous_reports else next(folder.glob("*.xlsx"))
        copy2(original, target / "Before.xlsx")
        # Keep trace links portable together with this newly exported workbook.
        copytree(folder / "review", target / "review")
        rebuilt = OfflineReportRebuilder().rebuild(
            checkpoint_path=checkpoint,
            method_baseline_path="method_assets/fusa_baseline_v1/manifest.yaml",
            report_style_template_path="references/HARA_Template_AI_20260327.xlsx",
            output_path=target / "HARA_Review.xlsx", review_root=target / "review",
            audit_output_dir=target / "audit",
        )
        after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert before == after, "Rebuild mutated the immutable checkpoint"
        audit = json.loads((target / "audit/p2-report-rebuild/report_projection_audit.json").read_text())
        records.append({"sample": kind, "checkpoint": str(checkpoint),
            "checkpoint_before_sha256": before, "checkpoint_after_sha256": after,
            "unchanged_fields": "entire checkpoint byte-identical: values, provenance, IDs, states",
            "provider_calls": 0, "workbook": str(rebuilt), "main_rows": audit["rows"],
            "detail_rows": audit["scenario_detail_rows"], "grouping": audit["grouping"]})
        print(kind, "main", audit["rows"], "details", audit["scenario_detail_rows"], flush=True)
    (output / "comparison.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-reports", type=Path, help="Exact prior exported workbooks for before/after comparison")
    args = parser.parse_args()
    run(args.source, args.output, args.previous_reports)
