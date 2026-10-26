# FUSA v1 Exposure parity authority

The direct source reviewed for the selected baseline is
`C:\Users\DengMohan(CN-TV-M4)\Desktop\fusa_agent\core\hara\evaluators\exposure_evaluator.py`.
Its `compute_exposure()` accepts only `severity`, resolved `e_dimension`,
Scenario atom identifiers and `atoms_coupling`.  It contains no
`FUNCTION_CONTEXT` coverage input and does not select required, optional or
not-applicable dimensions.

The original executable sequence is:

1. `S0` short-circuits to `E0`.
2. The resolved failure-mode Z/F domain is selected.
3. Values are collected only from supplied atoms in that domain.
4. If all supplied atoms lack the selected value, the complete Scenario moves
   to the other domain; individual atoms are never mixed across Z/F.
5. Both domains empty is pending.
6. All E4 yields E4; an E3/E4 mix yields E3; other unequal ranks yield the
   minimum.
7. Only equal ranks consume `atoms_coupling`; absent coupling is pending.

`method_assets/fusa_baseline_v1/normalized/exposure_policy.yaml` compiles this
source as `policy_id: fusa_v1`, while
`normalized/scenario_coverage_rules.yaml` contains no approved coverage rules.
Therefore `ExposureDimensionCoverageDecision` remains an audit diagnostic for
`fusa_v1`; it cannot block the native atom executor.  A differently named
future aggregation policy retains the formal coverage gate.

The baseline atom catalog is traced to
`C:\Users\DengMohan(CN-TV-M4)\Desktop\fusa_agent\skills\hara\vda702_atoms.yaml`
and is compiled into the MethodContract.  Runtime code reads the compiled
contract only.

## Current-project exact-set correction (2026-10-26)

The original implementation used a **contains** test: any supplied combination
containing E3 and E4 took that special branch, even if E1/E2 was also present.
The current project explicitly selects `mixed_high.match: EXACT_LEVEL_SET`:
only combinations whose distinct levels are exactly E3/E4 use that branch;
E1/E3/E4 and E2/E3/E4 use the existing unequal-level minimum policy. This is an
intentional governed behavior correction, not a claim of unchanged legacy
parity or a new ISO rule. Z/F routing, domain fallback, equal-level dependence,
atom levels, S/C and the ASIL matrix are unchanged.

Old contracts lacking this field retain `CONTAINS`. The original seven parity
cases and joint-completion counterexample remain tested against that explicit
legacy policy. The current policy has separate exact-set regression tests.
Both preview and scoring call the same compiled-policy match operation.
