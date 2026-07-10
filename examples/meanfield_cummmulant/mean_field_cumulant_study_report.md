# Mean-Field And Cumulant Reference-State Study

This study implements the blueprint from `Mean field higher body terms.md` as a reproducible diagnostic pass. It does not change the `PauliOperators.jl` public API. The goal is to compare the existing one-body mean-field truncation against the reference information needed for multireference states.

Run it from this folder with:

```bash
julia --project=. mean_field_cumulant_study.jl
```

This folder is its own Julia project. `Project.toml` references the local package source with:

```toml
[sources]
PauliOperators = {path = "../.."}
```

The study uses package APIs directly for expectations and evolution: `expectation_value`, `tr`, `trotterize`, `evolve!`, `truncate!`, and `MeanFieldTruncation`.

## What The Study Checks

- Product reference baseline: reproduces the XXZ-style single-reference setup and verifies connected cross-site cumulants vanish.
- Bell/GHZ-style multireference check: shows `<Z1> = <Z2> = 0` while `<Z1 Z2> = kappa(Z1,Z2) = 1`, so current single-site mean field drops a connected pair that a cumulant-aware method would preserve.
- Heisenberg eigenvector diagnostic: uses an XXX-chain eigenvector as a multireference `KetSum`, tracks mean-field expectation error, and compares it with weighted two-body and selected three-body cumulant signals from high-weight active Pauli strings.
- Infinite-temperature control: verifies non-identity Pauli moments and connected cumulants vanish, so cumulant-aware corrections should also do nothing.

## Outputs

The script writes results under:

```text
outputs/mean_field_cumulant_study/
```

Expected files:

- `mean_field_cumulant_study_results.md`
- `metrics.csv`
- `multireference_cumulant_diagnostics.csv`
- `product_reference_curves.png`
- `multireference_error_cumulant.png`

## Latest Verified Run

Verified on July 2, 2026 from this folder with:

```bash
JULIA_DEPOT_PATH="/private/tmp/julia_depot:$HOME/.julia" /Users/cshrikh/.julia/juliaup/julia-1.12.1+0.aarch64.apple.darwin14/bin/julia --project=. mean_field_cumulant_study.jl
```

Key results from `outputs/mean_field_cumulant_study/metrics.csv`:

- Product XXZ baseline, `N=10`, `k=3`: hard weight truncation max error `0.190489`; mean-field max error `0.000356569`.
- Bell/GHZ-style check: `kappa(Z1,Z2) = 1`; current single-site mean field produces `0` low-order `ZZ` terms at `k=1`.
- Heisenberg eigenvector diagnostic, `N=8`, `k=3`: final expectation error `0.651068`; pair-cumulant signal/error correlation `0.893413`.
- Selected triple-cumulant signal was below numerical tolerance in this run, so its correlation is reported as `NaN`.

## Interpretation

Ordinary mean field is the one-body cumulant baseline. It works best when the reference is product-like because cross-site connected cumulants vanish. For multireference states, one-body averages may be zero even when connected two-body structure is large. The Bell check is the minimal example: `ZZ` is invisible to single-site mean field at `k=1`, but it is fully visible to the two-body cumulant.

The Heisenberg eigenvector section is intentionally diagnostic rather than a new truncation rule. If the error grows with the weighted pair or triple cumulant signals, that is evidence that the next implementation should add a generalized cumulant/reference interface instead of further tuning one-body mean field.
