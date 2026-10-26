# Offline risk rescoring

`rescore-risk` recomputes the committed eligible queue with the same scoring,
ASIL and fact-binding services as production orchestration. It creates no
semantic agents or Provider clients. It writes a new checkpoint, risk trace
and template-preserving draft workbook; it refuses existing target paths.

```powershell
$env:PYTHONPATH='src'
python -m hara_agent rescore-risk --source-run-id hara-p1-final-validation-r2 --target-run-id hara-r2-risk-rescore-r3 --baseline method_assets/fusa_baseline_v1/manifest.yaml --risk-input-supplement output/HARA_R2_Risk_Input_Supplement.json --output output/HARA_R2_Risk_Recomputed_r3.xlsx
```

`--checkpoint` selects an explicit committed snapshot. When omitted,
`CheckpointRepository.path_for(source_run_id)` determines the path, which is
included in the execution summary. The source run ID, typed causal contract,
method hash/version, scenario identities/provenance and exact eligible queue
must agree before scoring. No Excel or reduced trace is used as input truth.

Committed snapshot decoding is separate from workflow resume invalidation.
Rescoring retains the stored scenario-generation version and fingerprints:
it consumes existing candidates, not the current generator. A causal-contract
or method mismatch fails closed unless `--source-baseline` proves one of the
supported C-only routing, applicability or conditional-reference changes.
All other assets and generation/causal contracts must remain unchanged.
The command never restarts upstream stages or upgrades a version label.

The optional JSON supplement is checkpoint-hash bound. `scopes` maps group
IDs to explicit `{malfunction_id, scenario_id}` pairs. Each `entries` item
contains `scope_id`, `field`, `value` (or an unchanged range), `unit` and
`provenance`. Null values document missing inputs. Source-supported unchanged
facts can be reasserted; new or changed conditions are queued for differential
validation. A citation or FINALIZED label alone cannot establish that an old
causal conclusion applies to a new condition. The command does not apply such
proposals to old HE results. `excluded_risk_facts` lists an existing `fact_id`,
`reason` and `source_locator`; these facts are demoted only in the target copy.
Supplemented analytical template options remain proposed new instances and
do not inherit old causal approval.

The result includes original/new value counts, every original association's
disposition in `risk_execution_trace.json`, source hashes, runtime, and
`business_scoring_complete`. Exit code 0 means execution succeeded, not that
all engineering inputs are available. Missing E never becomes QM and does
not suppress ready S/C. Old risk/Harm results are invalidated. SG membership
and highest known ASIL are recomputed by the same deterministic aggregator
used by `analyze`, preserving recorded original text and scoped Safe State
proposals. This does not call a semantic Provider or grant engineering approval.
QM members still require no SG; unresolved members remain visible. FTTI remains
not evaluated. A source-selected sample may have an empty guideword log when
its committed malfunction identities and causal evidence are already present.

The workbook's audit sheet contains the new trace reference and execution ID.
`rescore_summary.json` is the execution handoff; `summary.json` remains the
existing review-artifact summary. Generated reports and runtime artifacts
remain local under the repository's existing ignore rules.
