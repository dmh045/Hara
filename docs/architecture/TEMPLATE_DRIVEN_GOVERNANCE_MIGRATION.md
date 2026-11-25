# Template-Driven Governance — Cutover Record

> 历史设计与核验记录：本文中的“当前”、缺口和运行结论仅对应下述日期或提交。现行实现与使用方式见[开发交接手册](../handover/README.md)；历史记录索引见[文档目录](../README.md#历史开发与验证记录)。

## Final architecture

```text
WorkbookSnapshot
  -> automatic role discovery
  -> optional hash-bound role confirmation
  -> MethodContract

Item artifacts
  -> typed ProjectFacts
  -> bounded scenario interpretation
  -> neutral RiskFacts
  -> hash-bound MethodRiskFactBinding

MethodContract + facts
  -> constrained scenarios
  -> compiled S/E/C rules
  -> compiled ASIL matrix
  -> review-gated SG/Safe-State derivation
  -> ReportContract renderer
```

There is one runtime and one engineering truth path. Domain-policy scoring,
fixed template readers, old staged engines, compatibility adapters, alternate
report generators, and FTTI heuristics outside the active template method were
removed at cutover.

## Coupling boundary

`RiskFact` is deliberately template-independent: it contains a canonical
parameter, value, unit, context, provenance, approval, and exact evidence.
Only `MethodRiskFactBinding` contains `method_contract_hash`. This permits the
same extracted facts to be reused with another template while ensuring that a
semantic mapping confirmed for Template A cannot silently authorize Template B.

Automatic binding is limited to exact canonical ontology names. Ambiguous
mappings require human confirmation. Confirmations and Template Role manifests
are reusable by template hash and do not duplicate engineering rules.
They are generated governance records, not a third manually maintained business input.

## Fail-closed behavior

- Multiple credible role candidates require confirmation.
- Rule conflicts, gaps, overlaps, unresolved alternatives, missing provenance,
  or incomplete ASIL matrices block method readiness.
- Scenario/HUMAN_EVIDENCE facts are not guessed from examples or defaults.
- LLM-interpreted RiskFacts remain `PENDING` and retain exact source references.
- A formal workbook is blocked while required facts or approvals are pending.
- Renderer layout comes only from `ReportContract`; template identity is checked
  before writing, output is atomic, and the saved workbook is reopened and verified.

## Removed surfaces

- Domain registry/profile/policy runtime and copied AVP rule data.
- Three-phase and 17-step engines, experience-library routing, and legacy JSON adapters.
- Fixed ASIL/template readers and hard-coded report coordinate contract.
- Alternate `hara-analyze`, example, fallback, and controller entry points.
- Heuristic FTTI calculations not represented by the active MethodContract.

The supported entry point is `python -m hara_agent` (or installed `hara-agent`).
