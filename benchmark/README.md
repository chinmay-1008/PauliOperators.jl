# Sparse mean-field experiment

This experiment branches from `origin/main` commit
`8c4221f93d950e2408ac0a98e3d68b107a3c987d`. It compares the existing
`PauliSum` implementation with the packed `SparsePauliVector` implementation
for `MeanFieldTruncation` and normalized `SingleSiteMeanFieldDecoupling`.

Run the reproducible matrix with:

```shell
julia --project=. benchmark/mean_field_sparse_vector.jl
```

The driver uses fixed random seeds and checks direct parity to `1e-12`,
evolution parity to `1e-10`, and the structural SPV invariants before timing.
Direct and steady-state measurements exclude input construction/copying;
end-to-end measurements include the one-time `PauliSum`-to-SPV conversion.

## Results

Measured on 2026-07-29 with Julia 1.12.6 and an Intel Core i7-8565U
(4 cores/8 threads). Values are medians of five samples.

| Strategy | N / terms / k / rotations | Direct ms (Dict → SPV) | Direct speedup | Evolution ms (Dict → SPV) | Evolution speedup | End-to-end ms (Dict → SPV) | End-to-end speedup | Conversion ms | Break-even rotations | Final terms | Max coefficient difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean field | 16 / 1,000 / 2 / 30 | 0.226 → 0.068 | 3.34× | 5.504 → 2.325 | 2.37× | 5.605 → 2.310 | 2.43× | 0.296 | 3 | 1,126 | 1.332e-15 |
| Normalized SMFD | 16 / 1,000 / 2 / 30 | 0.228 → 0.051 | 4.45× | 4.440 → 1.698 | 2.61× | 4.872 → 2.381 | 2.05× | 0.294 | 4 | 1,127 | 4.441e-16 |
| Mean field | 16 / 5,000 / 3 / 100 | 1.554 → 0.566 | 2.75× | 456.462 → 164.340 | 2.78× | 457.127 → 195.704 | 2.34× | 1.238 | 1 | 16,249 | 2.220e-15 |
| Normalized SMFD | 16 / 5,000 / 3 / 100 | 1.369 → 0.463 | 2.96× | 416.546 → 145.358 | 2.87× | 425.266 → 144.315 | 2.95× | 1.726 | 1 | 16,248 | 1.332e-15 |
| Mean field | 20 / 1,000 / 4 / 30 | 0.288 → 0.114 | 2.52× | 723.866 → 206.977 | 3.50× | 578.650 → 242.651 | 2.38× | 0.420 | 1 | 154,968 | 3.109e-15 |
| Normalized SMFD | 20 / 1,000 / 4 / 30 | 0.164 → 0.060 | 2.75× | 286.974 → 106.030 | 2.71× | 278.964 → 106.615 | 2.62× | 0.403 | 1 | 117,110 | 2.220e-16 |
| Mean field | 20 / 5,000 / 3 / 100 | 1.642 → 0.672 | 2.44× | 863.990 → 306.248 | 2.82× | 838.911 → 293.916 | 2.85× | 1.714 | 1 | 32,550 | 9.326e-15 |
| Normalized SMFD | 20 / 5,000 / 3 / 100 | 1.226 → 0.419 | 2.92× | 751.081 → 234.052 | 3.21× | 749.207 → 235.363 | 3.18× | 1.656 | 1 | 32,550 | 8.882e-16 |

### Allocations

The prepared-input sparse transforms allocated zero bytes directly. The nine
steady-state allocations in the bounded cases are evolution-driver overhead;
the larger counts for the N=20, k=4 cases come from capacity growth as the
result expands above 100,000 terms.

| Strategy | N / terms / k / rotations | Direct allocations / KiB (Dict → SPV) | Evolution allocations / MiB (Dict → SPV) | End-to-end allocations / MiB (Dict → SPV) | Conversion allocations / MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Mean field | 16 / 1,000 / 2 / 30 | 3,717 / 486.44 → 0 / 0 | 35,318 / 6.31 → 9 / 0.73 | 35,327 / 6.50 → 48 / 2.57 | 38 / 1.83 |
| Normalized SMFD | 16 / 1,000 / 2 / 30 | 3,450 / 584.24 → 0 / 0 | 31,001 / 6.72 → 9 / 0.73 | 31,010 / 6.91 → 48 / 2.57 | 38 / 1.83 |
| Mean field | 16 / 5,000 / 3 / 100 | 17,437 / 2,359.41 → 0 / 0 | 2,547,841 / 438.10 → 9 / 3.67 | 2,547,850 / 438.86 → 48 / 12.82 | 38 / 9.16 |
| Normalized SMFD | 16 / 5,000 / 3 / 100 | 14,150 / 2,276.00 → 0 / 0 | 2,021,534 / 421.89 → 9 / 3.67 | 2,021,543 / 422.65 → 48 / 12.82 | 38 / 9.16 |
| Mean field | 20 / 1,000 / 4 / 30 | 2,474 / 333.70 → 0 / 0 | 1,948,583 / 363.31 → 95 / 55.67 | 1,948,592 / 363.51 → 134 / 57.50 | 38 / 1.83 |
| Normalized SMFD | 20 / 1,000 / 4 / 30 | 1,504 / 252.81 → 0 / 0 | 1,250,501 / 261.88 → 87 / 38.09 | 1,250,510 / 262.08 → 126 / 39.92 | 38 / 1.83 |
| Mean field | 20 / 5,000 / 3 / 100 | 16,465 / 2,253.77 → 0 / 0 | 3,733,097 / 591.04 → 9 / 3.67 | 3,733,106 / 591.81 → 48 / 12.82 | 38 / 9.16 |
| Normalized SMFD | 20 / 5,000 / 3 / 100 | 13,494 / 2,184.76 → 0 / 0 | 3,078,760 / 580.44 → 9 / 3.67 | 3,078,769 / 581.21 → 48 / 12.82 | 38 / 9.16 |

These numbers are machine-specific. The measured direct speedup is
2.44–4.45× for most cases (with a 2.52–2.75× N=20/k=4 range), steady-state
evolution is 2.37–3.50× faster, and end-to-end execution is 2.05–3.18× faster.
No 10× target is assumed.
