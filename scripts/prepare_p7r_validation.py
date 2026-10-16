"""Select unmodified production candidates for the bounded P7R validation.

Consumes benchmark_p6_offline artifacts. It does not create physical fields,
approve facts, or implement a scoring path.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(checkpoint: Path, generation: Path, output: Path, smoke: bool = False):
    if output.exists():
        raise ValueError("Use a new plan path")
    saved = json.loads(checkpoint.read_text())
    summary = json.loads((generation / "summary.json").read_text())
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert summary["source_checkpoint_sha256"] == source_hash
    previews = json.loads((generation / "candidate_previews.json").read_text())
    audits = {row["malfunction_id"]: row for row in json.loads((generation / "generation_audit.json").read_text())}
    functions = {row["function_id"]: row for row in saved["functions"]}
    malfunctions = {row["malfunction_id"]: row for row in saved["malfunctions"]}
    cases = []
    for identity, condition_limit in (("MF-F01-001", 3), ("MF-F01-003", 3), ("MF-F02-002", 1)):
        signatures, selected = [], []
        for row in previews:
            if row["malfunction_id"] != identity:
                continue
            candidate = row["source_candidate"]
            facts = candidate["facts"]
            signature = (facts.get("road_user_type"), facts.get("collision_type"),
                         facts.get("ego_longitudinal_direction"), facts.get("object_position"))
            if signature not in signatures and len(signatures) < condition_limit:
                signatures.append(signature)
            if signature in signatures and not any(
                (c["facts"].get("road_user_type"), c["facts"].get("collision_type"),
                 c["facts"].get("ego_longitudinal_direction"), c["facts"].get("object_position")) == signature
                and c["facts"].get("driver_position") == facts.get("driver_position") for c in selected
            ):
                selected.append(candidate)
        mf = malfunctions[identity]
        cases.append({"malfunction": mf, "function": functions[mf["function_id"]],
                      "candidates": selected, "generation": audits[identity]["generation"]})
    typed = dict(saved["item_definition"]["typed"])
    if smoke:
        cases = [{**cases[0], "candidates": cases[0]["candidates"][:2]}]
    typed["risk_facts"] = [f for f in typed["risk_facts"] if "scenario_id" not in f.get("context", {})]
    fact_ids = {f["fact_id"] for f in typed["risk_facts"]}
    typed["method_risk_fact_bindings"] = [b for b in typed["method_risk_fact_bindings"] if b["source_fact_id"] in fact_ids]
    plan = {"source_checkpoint_sha256": source_hash, "method_hash": summary["method_hash"],
            "generation_artifact": str(generation), "project": typed, "cases": cases,
            "provider_attempt_limit": 20, "selection": "source candidates copied unchanged, first object/direction families"}
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"plan": str(output), "pairs": sum(len(c["candidates"]) for c in cases),
                      "cases": [c["malfunction"]["malfunction_id"] for c in cases]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prepare(args.checkpoint, args.generation, args.output, args.smoke)
