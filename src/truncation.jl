using Random

# ============================================================
# Abstract Types
# ============================================================

"""
    TruncationStrategy

Abstract supertype for term-truncation strategies applied by `truncate!` and
the `truncation`/`local_truncation` keywords of `evolve!`. Define a new
strategy by subtyping and implementing `_apply!(O, s)`.
"""
abstract type TruncationStrategy end

"""
    CorrectionAccumulator

Abstract supertype for truncation-error trackers passed to `truncate!` and
`evolve!`: observables are measured before and after each truncation and the
differences accumulate. See `EnergyCorrection`, `EnergyVarianceCorrection`,
`NoCorrection`. Define a new accumulator by subtyping and implementing
`_measure(O, corr)` and `_accumulate!(corr, before, after)`.
"""
abstract type CorrectionAccumulator end


# ============================================================
# Truncation Strategy Types
# ============================================================

"""
    NoTruncation()

Identity truncation — does nothing.
"""
struct NoTruncation <: TruncationStrategy end

"""
    CoeffTruncation(thresh::Float64)

Remove Pauli terms with |coefficient| <= `thresh`.
"""
struct CoeffTruncation <: TruncationStrategy
    thresh::Float64
end
CoeffTruncation() = CoeffTruncation(1e-6)

"""
    WeightTruncation(max_weight::Int)

Remove Pauli terms with Pauli weight > `max_weight`.
"""
struct WeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    XWeightTruncation(max_weight::Int)

Remove Pauli terms with X-weight (number of X/Y factors) > `max_weight`.
"""
struct XWeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    MajoranaWeightTruncation(max_weight::Int)

Remove Pauli terms with Majorana weight > `max_weight`.
"""
struct MajoranaWeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    WeightDampedTruncation(alpha::Float64, thresh::Float64)

Remove Pauli terms with |coefficient|·exp(-alpha·weight) <= `thresh`,
i.e. a coefficient threshold that grows exponentially with Pauli weight.
`alpha = 0` reduces to `CoeffTruncation(thresh)`; large `alpha` approaches
a hard weight cutoff.
"""
struct WeightDampedTruncation <: TruncationStrategy
    alpha::Float64
    thresh::Float64
end
WeightDampedTruncation(alpha::Real) = WeightDampedTruncation(alpha, 1e-6)

"""
    XWeightDampedTruncation(alpha::Float64, thresh::Float64)

Remove Pauli terms with |coefficient|·exp(-alpha·x_weight) <= `thresh`,
i.e. a coefficient threshold that grows exponentially with X-weight (the
number of X/Y factors). `alpha = 0` reduces to `CoeffTruncation(thresh)`;
large `alpha` approaches a hard X-weight cutoff.
"""
struct XWeightDampedTruncation <: TruncationStrategy
    alpha::Float64
    thresh::Float64
end
XWeightDampedTruncation(alpha::Real) = XWeightDampedTruncation(alpha, 1e-6)

"""
    StochasticCoeffTruncation(epsilon::Float64; rng=Random.default_rng())

Unbiased stochastic compression (Russian Roulette). Wraps `stochastic_clip!`.

For each term with |c| < epsilon:
- Keep with probability |c|/epsilon (promote to epsilon·sign(c))
- Delete with probability 1 - |c|/epsilon
"""
struct StochasticCoeffTruncation <: TruncationStrategy
    epsilon::Float64
    rng::AbstractRNG
end
StochasticCoeffTruncation(epsilon::Float64) = StochasticCoeffTruncation(epsilon, Random.default_rng())

"""
    StochasticSamplingTruncation(n_keep::Int; rng=Random.default_rng())

Stochastically sample `n_keep` terms via importance sampling with probabilities
proportional to |c_i|^2. Kept terms are rescaled to preserve norm.
"""
struct StochasticSamplingTruncation <: TruncationStrategy
    n_keep::Int
    rng::AbstractRNG
end
StochasticSamplingTruncation(n_keep::Int) = StochasticSamplingTruncation(n_keep, Random.default_rng())

"""
    AdaptiveTruncation(max_terms::Int, min_thresh::Float64)

If the number of terms exceeds `max_terms`, increase the clipping threshold
to reduce the operator size. Otherwise clip at `min_thresh`.
"""
struct AdaptiveTruncation <: TruncationStrategy
    max_terms::Int
    min_thresh::Float64
end
AdaptiveTruncation(; max_terms::Int=10000, min_thresh::Float64=1e-12) = AdaptiveTruncation(max_terms, min_thresh)

"""
    MeanFieldTruncation(
        max_weight::Int,
        reference::Union{
            Ket{N},
            KetSum{N},
            ProductDensityReference{N},
            ProductBlochReference{N},
        },
    )

Replace each Pauli term with weight > `max_weight` by its order-`max_weight`
mean-field factorization around `reference`.

Unlike `WeightTruncation`, which discards high-weight terms, this strategy
expands each high-weight string in single-site fluctuations
`δP_i = P_i − ⟨P_i⟩ I` and keeps the lower-weight pieces. For computational-basis
`Ket` references, the optimized factorization preserves
`⟨reference|O|reference⟩` at every truncation order. For `KetSum` references,
the strategy uses local single-site expectations and is a mean-field
approximation for entangled states.
"""
struct MeanFieldTruncation{N,R} <: TruncationStrategy
    max_weight::Int
    reference::R
end

MeanFieldTruncation(max_weight::Int, reference::Ket{N}) where N =
    MeanFieldTruncation{N,typeof(reference)}(max_weight, reference)
MeanFieldTruncation(max_weight::Int, reference::KetSum{N}) where N =
    MeanFieldTruncation{N,typeof(reference)}(max_weight, reference)
MeanFieldTruncation(max_weight::Int, reference::ProductDensityReference{N}) where N =
    MeanFieldTruncation{N,typeof(reference)}(max_weight, reference)
MeanFieldTruncation(max_weight::Int, reference::ProductBlochReference{N}) where N =
    MeanFieldTruncation{N,typeof(reference)}(max_weight, reference)
MeanFieldTruncation{N}(max_weight::Int, reference::Ket{N}) where N =
    MeanFieldTruncation{N,typeof(reference)}(max_weight, reference)
MeanFieldTruncation{N}(max_weight::Int, reference::KetSum{N}) where N =
    MeanFieldTruncation{N,typeof(reference)}(max_weight, reference)
MeanFieldTruncation{N}(max_weight::Int, reference::ProductDensityReference{N}) where N =
    MeanFieldTruncation{N,typeof(reference)}(max_weight, reference)
MeanFieldTruncation{N}(max_weight::Int, reference::ProductBlochReference{N}) where N =
    MeanFieldTruncation{N,typeof(reference)}(max_weight, reference)

"""
    SingleSiteMeanFieldDecoupling(
        max_weight::Int,
        reference::Union{
            Ket{N},
            ProductDensityReference{N},
            ProductBlochReference{N},
        },
        normalize::Bool=false,
    )

For each Pauli term with weight greater than `max_weight`, replace one
nonidentity factor at a time by its local expectation value and sum the
results. This is the sparse single-site mean-field decoupling (SMFD) rule

```math
\\mathcal M(P_S) =
\\sum_{j \\in S} \\langle P_j \\rangle_{\\mathrm{reference}}
I_j \\prod_{i \\in S \\setminus \\{j\\}} P_i.
```

SMFD lowers each selected term by one site per application. It therefore
enforces the requested weight bound when the input weight is at most
`max_weight + 1`, as in bounded evolution under one- and two-site generators.
Raw SMFD does not generally preserve the reference expectation value. Set
`normalize=true` to divide each replacement by the number of contributing
sites. For a product reference, this normalized rule preserves the reference
expectation of every input term and removes SMFD's combinatorial overcounting.
"""
struct SingleSiteMeanFieldDecoupling{N,R} <: TruncationStrategy
    max_weight::Int
    reference::R
    normalize::Bool
end

SingleSiteMeanFieldDecoupling(
    max_weight::Int,
    reference::Ket{N};
    normalize::Bool=false,
) where N =
    SingleSiteMeanFieldDecoupling{N,typeof(reference)}(
        max_weight,
        reference,
        normalize,
    )
SingleSiteMeanFieldDecoupling(
    max_weight::Int,
    reference::ProductDensityReference{N},
    ;
    normalize::Bool=false,
) where N =
    SingleSiteMeanFieldDecoupling{N,typeof(reference)}(
        max_weight,
        reference,
        normalize,
    )
SingleSiteMeanFieldDecoupling(
    max_weight::Int,
    reference::ProductBlochReference{N},
    ;
    normalize::Bool=false,
) where N =
    SingleSiteMeanFieldDecoupling{N,typeof(reference)}(
        max_weight,
        reference,
        normalize,
    )
SingleSiteMeanFieldDecoupling(
    max_weight::Int,
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    normalize::Bool,
) where N =
    SingleSiteMeanFieldDecoupling(max_weight, reference; normalize=normalize)
SingleSiteMeanFieldDecoupling{N}(
    max_weight::Int,
    reference::Ket{N},
    ;
    normalize::Bool=false,
) where N =
    SingleSiteMeanFieldDecoupling{N,typeof(reference)}(
        max_weight,
        reference,
        normalize,
    )
SingleSiteMeanFieldDecoupling{N}(
    max_weight::Int,
    reference::ProductDensityReference{N},
    ;
    normalize::Bool=false,
) where N =
    SingleSiteMeanFieldDecoupling{N,typeof(reference)}(
        max_weight,
        reference,
        normalize,
    )
SingleSiteMeanFieldDecoupling{N}(
    max_weight::Int,
    reference::ProductBlochReference{N},
    ;
    normalize::Bool=false,
) where N =
    SingleSiteMeanFieldDecoupling{N,typeof(reference)}(
        max_weight,
        reference,
        normalize,
    )
SingleSiteMeanFieldDecoupling{N}(
    max_weight::Int,
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    normalize::Bool,
) where N =
    SingleSiteMeanFieldDecoupling{N}(
        max_weight,
        reference;
        normalize=normalize,
    )

"""
    RecursiveSingleSiteMeanFieldDecoupling(
        max_weight::Int,
        reference::Union{
            Ket{N},
            ProductDensityReference{N},
            ProductBlochReference{N},
        };
        normalize::Bool=false,
    )

Strict-weight variant of [`SingleSiteMeanFieldDecoupling`](@ref). It applies
the same one-site rule repeatedly until every surviving term has weight at
most `max_weight`. Thus an input at `max_weight + 1` behaves exactly like the
original strategy, while an input at `max_weight + r` receives `r` layers and
ends at the cutoff.

Set `normalize=true` to normalize every layer by its number of contributing
sites. For product references this preserves the reference expectation at
each layer and avoids multiplicity from different site-removal orders.
"""
struct RecursiveSingleSiteMeanFieldDecoupling{N,R} <: TruncationStrategy
    max_weight::Int
    reference::R
    normalize::Bool
end

function RecursiveSingleSiteMeanFieldDecoupling(
    max_weight::Int,
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    ;
    normalize::Bool=false,
) where N
    max_weight >= 0 || throw(ArgumentError("max_weight must be nonnegative"))
    return RecursiveSingleSiteMeanFieldDecoupling{N,typeof(reference)}(
        max_weight,
        reference,
        normalize,
    )
end

RecursiveSingleSiteMeanFieldDecoupling(
    max_weight::Int,
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    normalize::Bool,
) where N =
    RecursiveSingleSiteMeanFieldDecoupling(
        max_weight,
        reference;
        normalize=normalize,
    )

function RecursiveSingleSiteMeanFieldDecoupling{N}(
    max_weight::Int,
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    ;
    normalize::Bool=false,
) where N
    max_weight >= 0 || throw(ArgumentError("max_weight must be nonnegative"))
    return RecursiveSingleSiteMeanFieldDecoupling{N,typeof(reference)}(
        max_weight,
        reference,
        normalize,
    )
end

RecursiveSingleSiteMeanFieldDecoupling{N}(
    max_weight::Int,
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    normalize::Bool,
) where N =
    RecursiveSingleSiteMeanFieldDecoupling{N}(
        max_weight,
        reference;
        normalize=normalize,
    )

"""
    CompositeTruncation(strategies...)

Apply multiple truncation strategies in sequence.

Strategies are stored as a typed `Tuple` rather than `Vector{TruncationStrategy}`,
so the per-element dispatches inside `_apply!` resolve at compile time and the
inner `coeff_clip!` / `weight_clip!` calls inline. Constructing via the variadic
form (`CompositeTruncation(CoeffTruncation(1e-4), WeightTruncation(5))`) is
the supported call style; an `AbstractVector` constructor is also provided
for convenience but converts to a tuple internally.
"""
struct CompositeTruncation{S<:Tuple} <: TruncationStrategy
    strategies::S
end
CompositeTruncation(s::TruncationStrategy...) = CompositeTruncation(s)
CompositeTruncation(v::AbstractVector{<:TruncationStrategy}) = CompositeTruncation(Tuple(v))


# ============================================================
# _apply! — raw truncation dispatch (internal)
# ============================================================

function _apply!(O::PauliSum{N}, ::NoTruncation) where N
    return O
end

function _apply!(O::PauliSum{N}, s::CoeffTruncation) where N
    return coeff_clip!(O, s.thresh)
end

function _apply!(O::PauliSum{N}, s::WeightTruncation) where N
    return weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::XWeightTruncation) where N
    return x_weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::MajoranaWeightTruncation) where N
    return majorana_weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::WeightDampedTruncation) where N
    return weight_damped_clip!(O, s.alpha, s.thresh)
end

function _apply!(O::PauliSum{N}, s::XWeightDampedTruncation) where N
    return x_weight_damped_clip!(O, s.alpha, s.thresh)
end

function _apply!(O::PauliSum{N}, s::StochasticCoeffTruncation) where N
    return stochastic_clip!(O, s.epsilon; rng=s.rng)
end

function _apply!(O::PauliSum{N}, s::StochasticSamplingTruncation) where N
    length(O) <= s.n_keep && return O

    keys_vec = collect(keys(O))
    weights = [abs2(O[k]) for k in keys_vec]
    norm_sq = sum(weights)
    sampling_keys = [rand(s.rng)^(1.0/w) for w in weights]

    kept_idx = partialsortperm(sampling_keys, 1:s.n_keep, rev=true)
    kept_set = Set(keys_vec[i] for i in kept_idx)

    kept_norm_sq = sum(abs2(O[k]) for k in kept_set)
    filter!(p -> p.first in kept_set, O)

    if kept_norm_sq > 0
        scale = sqrt(norm_sq / kept_norm_sq)
        for k in keys(O)
            O[k] *= scale
        end
    end

    return O
end

function _apply!(O::PauliSum{N}, s::AdaptiveTruncation) where N
    if length(O) > s.max_terms
        coeffs = sort(abs.(collect(values(O))))
        if length(coeffs) > s.max_terms
            thresh = coeffs[end - s.max_terms]
            coeff_clip!(O, thresh)
        end
    else
        coeff_clip!(O, s.min_thresh)
    end
    return O
end

# Recursive tail-pop iteration over the heterogeneous tuple of strategies so
# each `_apply!(O, strategy)` resolves at compile time and inlines.
@inline _apply_tup!(O, ::Tuple{}) = O
@inline _apply_tup!(O, s::Tuple)  = (_apply!(O, first(s)); _apply_tup!(O, Base.tail(s)))

function _apply!(O::PauliSum{N}, s::CompositeTruncation) where N
    _apply_tup!(O, s.strategies)
    return O
end

function _apply!(O::PauliSum{N,T}, s::MeanFieldTruncation{N}) where {N,T}
    return mean_field_factorize!(O, s.reference, s.max_weight)
end

function _apply!(
    O::PauliSum{N,T},
    s::SingleSiteMeanFieldDecoupling{N},
) where {N,T}
    return single_site_mean_field_decouple!(
        O,
        s.reference,
        s.max_weight;
        normalize=s.normalize,
    )
end

function _apply!(
    O::PauliSum{N,T},
    s::RecursiveSingleSiteMeanFieldDecoupling{N},
) where {N,T}
    return recursive_single_site_mean_field_decouple!(
        O,
        s.reference,
        s.max_weight;
        normalize=s.normalize,
    )
end


# ============================================================
# Correction Accumulator Types
# ============================================================

"""
    NoCorrection()

Track nothing during truncation. Zero overhead.
"""
struct NoCorrection <: CorrectionAccumulator end

"""
    EnergyCorrection(ψ::Ket{N})

Track accumulated change in ⟨ψ|O|ψ⟩ due to truncation.
"""
mutable struct EnergyCorrection{N} <: CorrectionAccumulator
    ψ::Ket{N}
    accumulated_energy::Float64
end
EnergyCorrection(ψ::Ket{N}) where N = EnergyCorrection{N}(ψ, 0.0)

"""
    EnergyVarianceCorrection(ψ::Ket{N})

Track accumulated changes in both ⟨ψ|O|ψ⟩ and Var(O,ψ) due to truncation.
"""
mutable struct EnergyVarianceCorrection{N} <: CorrectionAccumulator
    ψ::Ket{N}
    accumulated_energy::Float64
    accumulated_variance::Float64
end
EnergyVarianceCorrection(ψ::Ket{N}) where N = EnergyVarianceCorrection{N}(ψ, 0.0, 0.0)


# ============================================================
# measure — snapshot quantities before/after truncation
# ============================================================

_measure(::AnyPauliSum, ::NoCorrection) = nothing

function _measure(O::AnyPauliSum{N}, corr::EnergyCorrection{N}) where N
    return (energy = real(expectation_value(O, corr.ψ)),)
end

function _measure(O::AnyPauliSum{N}, corr::EnergyVarianceCorrection{N}) where N
    return (energy = real(expectation_value(O, corr.ψ)),
            variance = real(variance(O, corr.ψ)))
end


# ============================================================
# _accumulate! — update accumulator with before/after diffs
# ============================================================

_accumulate!(::NoCorrection, before, after) = nothing

function _accumulate!(corr::EnergyCorrection, before, after)
    corr.accumulated_energy += after.energy - before.energy
end

function _accumulate!(corr::EnergyVarianceCorrection, before, after)
    corr.accumulated_energy += after.energy - before.energy
    corr.accumulated_variance += after.variance - before.variance
end


# ============================================================
# truncate! — unified entry point
# ============================================================

"""
    truncate!(O::PauliSum, strategy::TruncationStrategy,
              corr::CorrectionAccumulator=NoCorrection())

Apply `strategy` to truncate `O` in-place. If a `CorrectionAccumulator` is
provided, measure quantities before and after truncation and accumulate the
differences.

Users can define new strategies by subtyping `TruncationStrategy` and
implementing `_apply!(O, s)`. New correction types are defined by subtyping
`CorrectionAccumulator` and implementing `_measure(O, corr)` and
`_accumulate!(corr, before, after)`.
"""
function truncate!(O::AnyPauliSum, strategy::TruncationStrategy,
                   corr::CorrectionAccumulator=NoCorrection())
    before = _measure(O, corr)
    _apply!(O, strategy)
    after = _measure(O, corr)
    _accumulate!(corr, before, after)
    return O
end
