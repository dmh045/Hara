# HARA V13

Template-driven, evidence-grounded HARA analysis with one production runtime.

```text
Template workbook -> TemplateRoleContract -> MethodContract
Item document     -> ProjectFacts + neutral RiskFacts
MethodContract + facts -> HARAState -> ReportContract renderer
```

The template owns the analysis method and report structure. The Item owns project facts. Python supplies generic discovery, validation, rule execution, provenance, quality gates, and rendering; it does not contain a second set of HARA business rules.

## Run

Python 3.12 is the validated baseline.
The CLI loads a repository-local `.env` when present; existing shell or CI
environment variables take precedence, and secret values are never logged.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"

.\.venv\Scripts\python.exe -m hara_agent doctor `
  --template .\references\HARA_Template_AI_20260327.xlsx

.\.venv\Scripts\python.exe -m hara_agent analyze `
  --item .\input\ItemDef.docx `
  --template .\references\HARA_Template_AI_20260327.xlsx `
  --output .\output\HARA_Report.xlsx `
  --operating-mode "<mode from Item Definition>" `
  --allow-draft
```

`doctor` is offline for template compilation and separately reports LLM configuration readiness. `analyze` fails closed when the method, facts, or approvals are incomplete. `--allow-draft` permits a visibly watermarked report without bypassing review state.

### P7: physical input and bounded validation closure

P7 code and evidence are documented in [the unified issue ledger](docs/P7_Closure_20261014.md).
Normal operating speed, stationary initial speed and impact speed are separate.
Explicit template maneuvers now supply missing direction; fact binding precedes
physics derivation. Driver pairs share requests while retaining independent
judgments. Approved zero ratings can establish QM through the active matrix.

The fixed historical population produces `output/p7_closure/HARA_P7_Revised_Draft_20261014.xlsx`:
928 risk records, E finalized for 922 and C for 180. S/ASIL remain pending because
the historical 7 km/h operating limit does not establish impact speed. The old
report is preserved. Two live samples used 15 of 20 approved requests, with
response recovery verified. This is a revised engineering draft, not release
approval or authorization for another full Provider run.

### P6: ordinary analyze generates method-bound scenarios

The normal `analyze` path now combines Function/Malfunction context, the confirmed
ODD, the compiled Atom catalog, and the existing bounded coverage planner before
causal validation. Atom membership and physical constraints are deterministic;
the Provider validates causality. Selected conditions remain `SCENARIO_DEFINED`,
and Exposure uses the existing FUSA evaluator.

Identical structured candidates are coalesced with an identity map. Driver
variants share a causal result only when explicit dependency evidence proves
noninterference and the child evidence validates again. Each retained branch is
scored separately. Report grouping requires matching engineering conditions,
causal evidence, harm and finalized ratings; details retain all risk records.

The P6 review correction preserves source conditions before Atom projection.
WHERE uses source locations; road/parking-space types remain explicit permitted
alternatives in the source scope, without being multiplied into site locations.
Location semantics apply to every Atom dimension. Mechanism variants are covered
once per physical input, with a representative of each additional source context
and any novel primary variants retained. Uphill/downhill source boundaries are
sampled at one location per physical option. Unselected combinations are recorded
as unevaluated, never as proven equivalent.

The current offline replay in `output/p6_review_bounded/` has 613 representatives
and 1,226 driver instances. E preview is ready for 1,200; 26 remain pending. The
old 510-candidate merge proof and the intermediate 7,650-candidate expansion are
superseded. Geometry equivalence, joint ODD coverage and fresh causal/S/C/ASIL
validation remain incomplete; see `docs/P6_Review_Correction_Report.md`.

Follow-up validation (`docs/P6_Targeted_Validation_20261012.md`) found and fixed
missing compiled slope bounds for an existing Atom. The population remains
613 / 1,226; E is now ready for 1,224 with 2 unresolved traffic contexts.
The 18-case live sample completed in 19 requests and proved response recovery;
driver causal reuse was 0/9 groups and all 16 retained records still lacked TTC.
The new full run is `full-p6-20261012-002`, with its independent log at
`runtime/agent/full-p6-20261012-002.monitor.log`. It completed a full-scope draft
on 2026-10-13: 1,270 causal judgments, 964 risk records, 253 exclusions and
53 cases pending causal evidence. E is finalized for 958 records, S for 396,
and C/ASIL for 82. The report is `output/HARA_full_p6_20261012_002.xlsx`;
it is not for release. Driver causal reuse and report grouping saved no further
calls/rows in this run. A small eligibility-classification fix keeps unresolved
causal evidence pending instead of displaying it as an exclusion.

After representative causal and coverage validation, use a new run for the Item.
Resuming an already completed P5-O checkpoint does not regenerate its scenarios.

```bash
set -o pipefail
.venv/bin/python -u -m hara_agent analyze \
  --item ./input/ItemDef.docx \
  --method-baseline ./method_assets/fusa_baseline_v1/manifest.yaml \
  --report-template ./references/HARA_Template_AI_20260327.xlsx \
  --operating-mode "parking vehicle control" \
  --run-id full-p6-20261012-001 \
  --output ./output/HARA_full_p6_20261012.xlsx \
  --allow-draft \
  2>&1 | tee runtime/agent/full-p6-20261012-001.monitor.log
```

The offline replay uses production generation and Exposure services without a
Provider, and distinguishes candidate previews from committed risk results:

```bash
.venv/bin/python scripts/benchmark_p6_offline.py \
  --checkpoint runtime/agent/full-p5o-20261010-001.checkpoint.json \
  --corrected-report output/p5o_review_20261012/HARA_full_p5o_20261010_review_fixed.xlsx \
  --output output/p6_review_bounded
```

Audit the new source coverage and the previous merges without a Provider:

```bash
.venv/bin/python scripts/audit_p6_source_coverage.py --before output/p6 --after output/p6_review_bounded
```

`scripts/validate_p6_sample.py` prepares an eight-candidate sample offline by
default; `--live` explicitly enables the existing Provider with a durable request
cap. The completed attempt ledger for `p6-review-sample-20261012-001` is already
8/8. Do not reuse or overwrite it as a fresh budget. That initial live sample
stopped after one schema repair consumed the remaining allowance. The later
18-case validation and full-run results above supersede that initial outcome.

### macOS: resume the historical parking-control run

Run from the repository root, after the existing analysis process has stopped.
This command resumes `full-p5o-20261010-001`, using the confirmed parking-control
scope (0–7 km/h). Its extracted speed-envelope name is `parking vehicle control`.
The shorter name `parking` is ambiguous: the same Item also defines parking-space
search at 0–30 km/h. For other runs, use the applicable name in the checkpoint's
`item_definition.typed.speed_envelopes`; these names are project inputs.

```bash
set -o pipefail
.venv/bin/python -u -m hara_agent analyze \
  --item ./input/ItemDef.docx \
  --method-baseline ./method_assets/fusa_baseline_v1/manifest.yaml \
  --report-template ./references/HARA_Template_AI_20260327.xlsx \
  --operating-mode "parking vehicle control" \
  --run-id full-p5o-20261010-001 \
  --output ./output/HARA_full_p5o_20261010.xlsx \
  --allow-draft --resume \
  2>&1 | tee -a runtime/agent/full-p5o-20261010-001.monitor.log
```

The explicit interpreter uses the project's virtual environment. `tee` displays
progress in the terminal and saves the same output. To watch the already running
analysis from another terminal:

```bash
tail -n 25 -F runtime/agent/full-p5o-20261010-001.monitor.log
```

Ctrl+C in this log-viewing terminal stops only `tail`. For a fresh analysis, omit
`--resume` and choose a new run ID, output path, and log path.

When `doctor` reports an ambiguous role, confirm only the listed role once; the system saves an auto-generated manifest keyed by the workbook hash:

```powershell
.\.venv\Scripts\python.exe -m hara_agent confirm-template-role `
  --template .\references\HARA_Template_AI_20260327.xlsx `
  --select "ASIL_MATRIX=ASIL_Table!A1:E20" `
  --confirmed-by "reviewer-id" `
  --rationale "selected the normative matrix"
```

No manually maintained template configuration is required. Automatic discoveries and human confirmations use `runtime/agent/template-role-manifests` by default; `HARA_TEMPLATE_ROLE_MANIFEST_DIR` can relocate that generated cache.

## Core contracts

- `MethodContract`: discovered workflow, guidewords, scenario dimensions, S/E/C rules, ASIL matrix, SG/Safe-State instructions, required facts, diagnostics, provenance, and report mappings.
- `ProjectFacts`: source-grounded Item/ODD facts independent of the template.
- `RiskFact`: source-grounded scenario/project fact independent of the template.
- `MethodRiskFactBinding`: template-hash-bound mapping from a neutral `RiskFact` to a method fact type.
- `ReportContract`: canonical output field to discovered sheet/header/column mapping.

Exact ontology-name bindings are automatic. Semantic interpretation is evidence-bounded and stays pending until approved. Human-confirmed mappings are reusable for the same template hash.

In YAML baseline mode, `normalized/malfunction_situation_selection.yaml` compiles
reference HARA situation guidance into the same `MethodContract`. Selection runs
once per malfunction and base scenario, before FM physical options and driver
branches. `INCLUDE` records supported situations; `UNRESOLVED` continues to the
existing causal assessment. The supplied reference table authorizes no exclusions.

Selection requires approved, source-linked `RiskFact` bindings for
`FUNCTIONAL_OUTPUT` and `FUNCTION_OPERATING_SCOPE`, scoped by `project_scope`,
`function_id`, and `malfunction_id`; the output binding also identifies the exact
`function_output`. Scope values are `WITHIN_FUNCTION` or `OUTSIDE_FUNCTION`.
Missing bindings remain unresolved without an additional review gate or model
request. Per-parent reasons, expansion counts, causal counts, and Z/F source
limitations are recorded in `malfunction_situation_selection.json` under the
existing review directory. Synthesis records a read-only applicability annotation.
Reference Z/F trigger definitions and assessment examples do not change the
active `fusa_v1` exposure policy, component-domain routing, or S/C/ASIL methods.

## Quality and safety behavior

- No fixed sheet names, coordinates, guideword cardinality, or copied S/E/C thresholds in runtime code.
- Template examples and references cannot override normative rules.
- Missing facts, conflicting values, unknown units, ambiguous ranges, incomplete ASIL matrices, and hash mismatches remain explicit and block release.
- Reports are produced by rewriting only target worksheet XML in a copied package, saving atomically, reopening, and verifying canonical IDs and ASIL values.

## Repository layout

```text
src/hara_agent/template/            template discovery and compilation
src/hara_agent/contracts/           typed contracts and rule IR
src/hara_agent/services/extraction/ project fact extraction
src/hara_agent/services/semantic/   bounded semantic interpretation
src/hara_agent/services/analysis/   generic method executors
src/hara_agent/services/reporting/  ReportContract renderer
src/hara_agent/workflow/            orchestration, checkpoints, quality gate
skills/                             stage procedures
tests/                              offline unit/evaluation tests
```

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
git diff --check
```
