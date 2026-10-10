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
