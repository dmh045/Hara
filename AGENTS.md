# HARA V13 Engineering Instructions

## Production entry points

```powershell
python -m hara_agent doctor --template TEMPLATE.xlsx
python -m hara_agent confirm-template-role --template TEMPLATE.xlsx --select "ROLE=SHEET!A1:B9" --confirmed-by REVIEWER
python -m hara_agent analyze --item ITEM.docx --template TEMPLATE.xlsx --output REPORT.xlsx --operating-mode MODE --allow-draft
python -m hara_agent doctor --method-baseline method_assets/fusa_baseline_v1/manifest.yaml --report-template references/HARA_Template_AI_20260327.xlsx
python -m hara_agent analyze --item ITEM.docx --method-baseline method_assets/fusa_baseline_v1/manifest.yaml --report-template references/HARA_Template_AI_20260327.xlsx --output REPORT.xlsx --operating-mode MODE --allow-draft
```

The installed console entry is `hara-agent`. There is one runtime and one downstream `MethodContract`; do not recreate old controllers, fallback engines, compatibility adapters, or alternate report generators. A governed YAML baseline is an additional method source, not a second runtime.

## Truth-source contract

- Explicit Template → `TemplateRoleContract` → `MethodContract`.
- No explicit Template → governed YAML baseline manifest → `MethodContract`.
- Item/source documents → typed `ProjectFacts` and neutral `RiskFacts`.
- `MethodContract + ProjectFacts/RiskFacts` → HARA evaluation.
- `HARAState + ReportContract` → template-preserving Excel report. Method-source and report-template hashes are independent in YAML baseline mode.
- Human confirmation is required only for unresolved semantic ambiguity or engineering approval.

Template role confirmations and explicit `MethodRiskFactBinding` records are bound to `template_hash`; repeat runs with the same template reuse them. They are generated governance artifacts, not a third manually maintained business input.

## Non-negotiable rules

- Never copy S/E/C, ASIL, scenario, SG, or Safe-State business rules into Python.
- Runtime evaluators never open YAML directly; only the YAML baseline compiler may load governed method assets.
- Never identify a method role by a fixed sheet name, sheet order, row, column, range, or workbook hash.
- Never infer missing project facts from template examples or defaults.
- LLM outputs must be schema-bounded, source-linked, and `PENDING` until approved when they contain semantic inference.
- Examples and references are non-executable unless the template explicitly marks them normative.
- Ambiguous rules, range gaps/overlaps, missing facts, hash mismatch, or conflicting facts fail closed.
- ASIL is determined only through the active `MethodContract.asil` matrix.
- Renderer coordinates come only from the compiled `ReportContract`; copy the template package, write atomically, reopen, and verify.
- Static HARA columns may be hierarchically merged only by canonical field identity. Dynamic S/E/C, ASIL, SG, and remark fields are never merged.

## Architecture boundaries

- `src/hara_agent/template/`: workbook scan, role discovery, role confirmation, rule parsing, MethodContract compilation. No project/domain runtime imports.
- `src/hara_agent/services/extraction/`: source-grounded project fact extraction and normalization.
- `src/hara_agent/services/semantic/`: bounded semantic agents; no engineering-rule authority.
- `src/hara_agent/services/analysis/`: generic MethodContract executors and fact binding.
- `src/hara_agent/services/reporting/`: ReportContract-driven rendering only.
- `src/hara_agent/workflow/`: typed state, checkpointing, quality gate, orchestration.

## Required validation

Before handoff run:

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

Also run `python -m hara_agent doctor` against the active template. Any skipped or unavailable live-provider test must be reported separately; offline compiler and unit tests must never require a Provider.

## Documentation authority

`SKILL.md`, stage skills, this file, the active input documents/template, compiled contracts, and deterministic services are current authority. Historical migration documents must not be restored as runtime guidance.

## GitHub delivery

After each project update, complete the required validation, commit the relevant project changes, and push to GitHub. This is the user’s standing instruction. Keep credentials, personal configuration, and historical runtime artifacts out of commits; do not force-push.
