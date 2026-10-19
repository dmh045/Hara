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
- Compatible analysis hypotheses may enter the existing deterministic calculator for conditional results; never present them as source facts or engineering approval, relax physical/source constraints, or require users to supply every generated dynamic parameter.
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

Samples and ordinary production share one analysis strategy. Sample entry points only select scope, budget and evidence recording; they must not use private semantic prompts or prefill scoring fields to create sample-only success.

## Required validation

Before handoff run:

```powershell
python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

Also run `python -m hara_agent doctor` against the active template. Any skipped or unavailable live-provider test must be reported separately; offline compiler and unit tests must never require a Provider.

Run affected tests locally, then the full gates and necessary CLI/report/resume checks before delivery. Do not add skips, delete failing counterexamples, or bulk-update snapshots to pass cleanup. Judge results by functional/environment coverage, mechanism results, request cost and maintainability, not lines changed, test counts or report row reduction.

## Documentation authority

`SKILL.md`, stage skills, this file, the active input documents/template, compiled contracts, and deterministic services are current authority. Historical migration documents must not be restored as runtime guidance.

## GitHub delivery

After each project update, complete the required validation, commit the relevant project changes, and push to GitHub. This is the user’s standing instruction. Keep credentials, personal configuration, and historical runtime artifacts out of commits; do not force-push.

## Targeted repairs

Start each round by inspecting current source, uncommitted work and applicable instructions. Find the authoritative implementation and its production callers before editing; explain why an existing responsibility cannot serve any proposed new service, wrapper, state or switch. Locate and fix routine defects autonomously with the smallest demonstrated repair. Preserve effective behavior and verify the affected production path. Report material method or behavior changes; new paid Provider usage requires a current, unspent, explicit authorization. Cleanup never authorizes another paid full run.

## Safe code convergence

- Before every delivery inspect touched modules and direct callers for duplicate responsibilities, obsolete paths, synonymous policies and extra side effects. After every three complete repair rounds, or before a full validation run (whichever comes first), inspect cross-module responsibilities. Count work rounds, not commits. A check does not require a deletion; record when no safe removal is proven.
- Static non-reference or coverage is only a candidate signal. Before removal check imports/re-exports, CLI, dynamic registration, configuration/template/skill strings, serialized type paths, old checkpoints, replay scripts, fixtures and documented commands. Preserve distinct responsibilities of method-source compilers, facts versus hypotheses, calculation versus display, and offline replay versus real requests.
- Keep `fix:` and `refactor:` commits separate: A is the recorded baseline, B the verified functional repair with explained behavior changes, C the cleanup of B. Do not require known wrong A behavior to survive a repair.
- Compare B and C with the same input, method, configuration, frozen responses and random seed. Equivalence includes numerical results, sources/approval, objects/stages/coordinates, identities, exclusions, exceptions, request contracts/counts, retries, budget/cache behavior, concurrent persistence and recovery. Old checkpoints must still read and re-export. Whitelist only explicit non-business differences such as timestamps/output paths.
- Prompt semantics, method policies, public interfaces and cache-key meaning are behavioral changes, not cleanup. Validate invalidation/compatibility separately. Roll back a cleanup with unexplained differences rather than weakening assertions or altering expected results.
- Migrate actual callers to the retained authoritative implementation and remove the replaced copy. Keep necessary compatibility entries thin, documenting consumers and removal conditions; do not introduce a parallel runtime or universal service to combine unlike duties.
- Reuse existing tools/tests. Preserve assertion coverage when merging equivalent fixtures. Historical inputs, method assets, responses, ledgers, reports, personal configuration and uncommitted work are never cleanup targets. No destructive reset/clean, unrelated mass formatting, unmapped directory moves or force-push.
- Use the same closure checklist for original responsibility, actual callers, redundancy evidence, retained implementation, migrations/removals, tests, differences and independently reversible commits. Do not accumulate permanent stage-specific repair scripts or duplicate reports in production directories.
