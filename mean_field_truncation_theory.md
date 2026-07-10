# Mean-Field Truncation Theory

This note explains the mean-field truncation used in this package. It is based
on the derivation in `examples/concept.ipynb` and the implementation in
`src/mean_field.jl` and `src/truncation.jl`.

The main idea is simple:

- `WeightTruncation(k)` removes every Pauli string whose weight is larger than
  `k`.
- `MeanFieldTruncation(k, psi)` replaces each high-weight Pauli string by a
  sum of lower-weight Pauli strings, expanded around a reference computationalv
  basis state `psi`.

So mean-field truncation is not just deletion. It projects a high-weight string
onto the weight-`<= k` subspace while preserving its expectation value on the
chosen reference state.

## Why Truncation Is Needed

In Heisenberg-picture Pauli propagation, an observable is stored as a
`PauliSum`,

```julia
O = sum_j c_j P_j
```

where each `P_j` is a Pauli string. Under time evolution, commutators and
Trotter steps generate more strings, and the weights of those strings can grow.
Without truncation, the number of terms can become too large.

A hard weight cutoff,

```julia
truncate!(O, WeightTruncation(k))
```

keeps only strings with Pauli weight `<= k`. This is cheap, but it throws away
the entire contribution of each high-weight string. Mean-field truncation tries
to keep the part of a high-weight string that is visible from a chosen product
reference state.

## Reference State

The implementation currently uses a computational-basis reference

```julia
psi::Ket{N}
```

For a single qubit in a computational-basis state,

```math
\langle X_i \rangle_\psi = 0,\qquad
\langle Y_i \rangle_\psi = 0,\qquad
\langle Z_i \rangle_\psi = \epsilon_i \in \{+1,-1\}.
```

In code, for qubit `i`,

```julia
epsilon_i = bit_i(psi) == 0 ? +1 : -1
```

This means pure `Z` factors can be replaced by their mean value plus a
fluctuation. Pure `X` factors and `Y` factors have zero expectation value, so
they must remain as operator factors if the term is to survive.

## Fluctuation Decomposition

For any local operator `A`, write

```math
A = \langle A \rangle_\psi I + \delta A,
\qquad
\delta A = A - \langle A \rangle_\psi I.
```

For a two-body product `AB`,

```math
AB =
(\langle A \rangle I + \delta A)
(\langle B \rangle I + \delta B).
```

Expanding gives

```math
AB =
\delta A\delta B
+ \langle B \rangle \delta A
+ \langle A \rangle \delta B
+ \langle A \rangle \langle B \rangle I.
```

If we keep only terms with at most one fluctuation, we drop
`\delta A\delta B`. Substituting
`\delta A = A - \langle A \rangle I` and
`\delta B = B - \langle B \rangle I` gives the usual two-body mean-field
formula:

```math
AB \approx
\langle B \rangle A
+ \langle A \rangle B
- \langle A \rangle \langle B \rangle I.
```

The package generalizes this same idea to arbitrary Pauli strings and an
arbitrary target weight `k`.

## General Pauli String

Take one Pauli string

```math
cP = c \prod_{i=1}^N P_i.
```

Split its sites into two groups:

- `X/Y sites`: sites where the Pauli has an `x` bit set. These are `X` and `Y`
  sites. Their reference expectation value is zero.
- `pure-Z sites`: sites where the Pauli is `Z`, not `Y`. These have expectation
  value `epsilon_i = <Z_i>_psi = +/-1`.

The code calls the number of `X/Y` sites

```julia
n_xy = count_ones(pb.x)
```

Those `X/Y` sites are unavoidable fluctuations. If

```julia
n_xy > k
```

then the mean-field projection is zero, because even before keeping any pure
`Z` factors the term already exceeds the allowed fluctuation weight.

If `n_xy <= k`, the pure-`Z` factors receive the remaining budget

```julia
budget = k - n_xy
```

Each pure `Z_i` is expanded as

```math
Z_i = \epsilon_i I + \delta Z_i,
\qquad
\delta Z_i = Z_i - \epsilon_i I.
```

The exact expansion is a sum over all subsets of pure-`Z` fluctuation sites.
The order-`k` mean-field truncation keeps only terms with at most `budget`
pure-`Z` fluctuations.

## Formula Used By The Code

Let `Zset` be the set of pure-`Z` sites in the Pauli string, and let
`n_z = length(Zset)`. Let `Q` be the part of the Pauli string containing all
`X` and `Y` sites, including the `Z` bit carried by any `Y`.

The truncated result is a sum over subsets `T` of the pure-`Z` sites that remain
as actual `Z` operators after converting back from fluctuation variables:

```math
cP \mapsto
\sum_{T \subseteq Zset,\ |T| \le budget}
c\,
\left(\prod_{i \in Zset \setminus T} \epsilon_i\right)
A(n_z-|T|, budget-|T|)
\; Q\prod_{i \in T} Z_i.
```

The alternating binomial factor is

```math
A(n,r) = \sum_{m=0}^{r} {n \choose m}(-1)^m.
```

This is exactly what `_partial_alt_binom(n, r)` computes in
`src/mean_field.jl`.

The reason this factor appears is that a kept fluctuation product such as
`\delta Z_2\delta Z_3` must be converted back into ordinary Pauli strings:

```math
\delta Z_i = Z_i - \epsilon_i I.
```

Different fluctuation products can collapse onto the same ordinary Pauli
string. The alternating binomial sum collects all those contributions without
building the full intermediate expansion.

## Worked Example: `X1 Z2 Z3` With `k = 2`

Use the reference state `|000>`, so

```math
\langle Z_2 \rangle = \langle Z_3 \rangle = 1,
\qquad
\langle X_1 \rangle = 0.
```

Start from

```math
P = X_1 Z_2 Z_3.
```

Write the two pure-`Z` factors as fluctuations:

```math
Z_2 = I + \delta Z_2,\qquad
Z_3 = I + \delta Z_3.
```

Then

```math
P = X_1(I+\delta Z_2)(I+\delta Z_3).
```

Expanding,

```math
P =
X_1
+ X_1\delta Z_2
+ X_1\delta Z_3
+ X_1\delta Z_2\delta Z_3.
```

The `X_1` factor already counts as one fluctuation. Since `k = 2`, we can keep
at most one pure-`Z` fluctuation. Therefore we drop the last term:

```math
P^{MF}_{k=2}
=
X_1
+ X_1\delta Z_2
+ X_1\delta Z_3.
```

Convert back to standard Pauli strings:

```math
P^{MF}_{k=2}
=
X_1
+ X_1(Z_2-I)
+ X_1(Z_3-I).
```

So

```math
X_1Z_2Z_3
\mapsto
X_1Z_2 + X_1Z_3 - X_1.
```

A weight-3 string has become a sum of strings with weights 2, 2, and 1.

## Important Properties

The tests in `test/test_mean_field.jl` check the important invariants.

First, if `k` is at least the original Pauli weight, the factorization returns
the original string exactly:

```math
P^{MF}_{k \ge weight(P)} = P.
```

Second, every output string has weight `<= k`.

Third, the reference expectation value is preserved for every `k`:

```math
\langle \psi | P^{MF}_k | \psi \rangle
=
\langle \psi | P | \psi \rangle.
```

By linearity, the same holds for a full `PauliSum` at the instant where
truncation is applied:

```math
\langle \psi | O^{MF}_k | \psi \rangle
=
\langle \psi | O | \psi \rangle.
```

This is the key advantage over hard weight truncation. Weight truncation can
change the reference expectation value immediately. Mean-field truncation keeps
that expectation value fixed while still controlling operator weight.

## How `src/mean_field.jl` Implements It

The central function is

```julia
mean_field_factorize(pb::PauliBasis{N}, c, psi::Ket{N}, k::Int)
```

It acts on one Pauli basis term `c * pb`.

The implementation steps are:

1. Count the number of `X/Y` sites:

   ```julia
   n_xy = count_ones(pb.x)
   ```

   If `n_xy > k`, return an empty `PauliSum`.

2. Find the pure-`Z` sites:

   ```julia
   z_only = pb.z & ~pb.x
   z_pos = get_on_bits(z_only)
   ```

3. Compute the reference signs `epsilon_i = +/-1` for those pure-`Z` sites.

4. Enumerate subsets `T` of pure-`Z` sites with size `t <= k - n_xy`.

5. For each `T`, compute the coefficient

   ```julia
   c * prod(epsilon_i for i not in T) *
       _partial_alt_binom(n_z - t, budget - t)
   ```

6. Add the resulting lower-weight `PauliBasis` to the output.

The in-place version

```julia
mean_field_factorize!(O::PauliSum, psi::Ket, k::Int)
```

only replaces terms whose weight is larger than `k`. Terms already within the
weight cutoff are left untouched.

## How `src/truncation.jl` Hooks It Into Evolution

`MeanFieldTruncation` is a normal `TruncationStrategy`:

```julia
struct MeanFieldTruncation{N} <: TruncationStrategy
    max_weight::Int
    reference::Ket{N}
end
```

The dispatch rule is:

```julia
_apply!(O::PauliSum{N,T}, s::MeanFieldTruncation{N}) =
    mean_field_factorize!(O, s.reference, s.max_weight)
```

So it can be used anywhere the package expects a truncation strategy:

```julia
psi = Ket(repeat([0, 1], 5))
k = 4

strat = CompositeTruncation(
    MeanFieldTruncation(k, psi),
    CoeffTruncation(1e-5),
)

Ot = evolve(Ot, generators, angles; truncation=strat)
```

The coefficient truncation after mean-field truncation is often useful because
one high-weight string can expand into several lower-weight strings.

## Special Cases

### `k = 0`

If the Pauli string contains any `X` or `Y` site, then `n_xy > 0`, so the result
is zero.

If the Pauli string is pure `Z`, then it becomes a scalar identity term:

```math
P \mapsto \langle \psi | P | \psi \rangle I.
```

This behaves like an automatic zero-body energy correction.

### Pure `Z` Strings

For a pure `Z` string, the method is just a fluctuation expansion around the
classical spin configuration encoded by `psi`. A high-weight product of `Z`s is
replaced by a sum of lower-weight `Z` products plus possibly an identity term.

### Strings With Many `X/Y` Factors

If the number of `X/Y` factors is already larger than `k`, the term cannot be
represented within the target weight using this computational-basis mean field.
The implementation returns no term.

## What Mean-Field Truncation Does Not Promise

Mean-field truncation preserves

```math
\langle \psi | O | \psi \rangle
```

at the moment of truncation. It does not make the later time evolution exact.
After more Trotter steps and more truncations, the propagated observable is
still approximate.

It also does not necessarily preserve the operator norm or variance. The demo
in `examples/mean_field_demo.jl` notes that repeated mean-field steps can
increase the operator's L2 norm and the variance on the reference state. So the
method is best understood as a structured approximation, not a free accuracy
guarantee.

Finally, the current implementation is specialized to `Ket{N}` computational
basis references. A true mean-field expansion around a general eigenvector or
superposition would require using more general local expectation values, such
as nonzero `<X_i>` or `<Y_i>`, and probably a different implementation path.

## One-Sentence Summary

`MeanFieldTruncation(k, psi)` replaces high-weight Pauli strings by their
order-`k` fluctuation expansion around the product reference `psi`, keeping only
lower-weight pieces while preserving the reference expectation value exactly at
each truncation step.
