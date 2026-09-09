"""
    single_site_mean_field_decouple(pb::PauliBasis{N}, c,
                                    reference::Ket{N};
                                    normalize=false) -> PauliSum

Apply single-site mean-field decoupling (SMFD) to the Pauli term `c * pb`,

```math
\\mathcal M(P_S) =
\\sum_{j \\in S} \\langle P_j \\rangle_{\\mathrm{reference}}
I_j \\prod_{i \\in S \\setminus \\{j\\}} P_i.
```

The sum runs only over the nonidentity support of `pb`. For a computational
basis `Ket`, only pure-`Z` factors have nonzero local means. Each nonzero output
term therefore has weight exactly `weight(pb) - 1`.

SMFD is a one-step decoupling, not the order-`k` centered-fluctuation expansion
implemented by [`mean_field_factorize`](@ref).

With `normalize=true`, divide the result by the number of contributing
pure-`Z` sites. This averages the one-site replacements instead of summing
them.
"""
function single_site_mean_field_decouple(pb::PauliBasis{N}, c::T,
                                         reference::Ket{N};
                                         normalize::Bool=false) where {N,T}
    z_only = pb.z & ~pb.x
    n_z = count_ones(z_only)
    R = normalize && n_z > 0 ? typeof(c / n_z) : T
    result = PauliSum(N, R)
    scale = normalize && n_z > 0 ? inv(R(n_z)) : one(R)

    for q in get_on_bits(z_only)
        mask = _bitmask(q)
        local_mean = iszero(reference.v & mask) ? 1 : -1
        pb_new = PauliBasis{N}(pb.z & ~mask, pb.x)
        _add_term!(result, pb_new, R(c) * local_mean * scale)
    end

    return result
end

single_site_mean_field_decouple(
    pb::PauliBasis{N},
    c,
    reference::Ket{N},
    normalize::Bool,
) where N =
    single_site_mean_field_decouple(
        pb,
        c,
        reference;
        normalize=normalize,
    )


"""
    single_site_mean_field_decouple(
        pb::PauliBasis{N}, c, reference::ProductDensityReference{N};
        normalize=false
    ) -> PauliSum

Apply single-site mean-field decoupling (SMFD) around a uniform diagonal
product-density reference. Pure-`Z` sites contribute the local mean
`reference.magnetization`; `X` and `Y` sites have zero local mean and therefore
produce no output term. With `normalize=true`, average the result over the
number of contributing pure-`Z` sites.
"""
function single_site_mean_field_decouple(
    pb::PauliBasis{N},
    c::T,
    reference::ProductDensityReference{N,S},
    ;
    normalize::Bool=false,
) where {N,T,S}
    z_only = pb.z & ~pb.x
    n_z = count_ones(z_only)
    R0 = promote_type(T, S)
    R = normalize && n_z > 0 ? typeof(R0(c) / n_z) : R0
    result = PauliSum(N, R)
    local_mean = R(reference.magnetization)
    iszero(local_mean) && return result
    scale = normalize && n_z > 0 ? inv(R(n_z)) : one(R)

    for q in get_on_bits(z_only)
        mask = _bitmask(q)
        pb_new = PauliBasis{N}(pb.z & ~mask, pb.x)
        _add_term!(result, pb_new, R(c) * local_mean * scale)
    end

    return result
end

single_site_mean_field_decouple(
    pb::PauliBasis{N},
    c,
    reference::ProductDensityReference{N},
    normalize::Bool,
) where N =
    single_site_mean_field_decouple(
        pb,
        c,
        reference;
        normalize=normalize,
    )


"""
    single_site_mean_field_decouple(
        pb::PauliBasis{N}, c, reference::ProductBlochReference{N};
        normalize=false
    ) -> PauliSum

Apply single-site mean-field decoupling (SMFD) around a uniform product-Bloch
reference. Every nonidentity factor can contribute: an `X`, `Y`, or `Z` factor
is replaced by `reference.x`, `reference.y`, or `reference.z`, respectively.
Factors whose local expectation is zero produce no output term.

With `normalize=true`, average over the number of factors with nonzero local
expectation. For a product-Bloch reference, this normalized rule preserves the
reference expectation of each input Pauli term (including the zero-expectation
cases).
"""
function single_site_mean_field_decouple(
    pb::PauliBasis{N},
    c::T,
    reference::ProductBlochReference{N,S},
    ;
    normalize::Bool=false,
) where {N,T,S}
    support = pb.z | pb.x
    n_contributing = 0
    for q in get_on_bits(support)
        mask = _bitmask(q)
        has_z = !iszero(pb.z & mask)
        has_x = !iszero(pb.x & mask)
        local_mean = has_z ?
                     (has_x ? reference.y : reference.z) :
                     reference.x
        n_contributing += !iszero(local_mean)
    end

    R0 = promote_type(T, S)
    R = normalize && n_contributing > 0 ?
        typeof(R0(c) / n_contributing) : R0
    result = PauliSum(N, R)
    iszero(n_contributing) && return result
    scale = normalize ? inv(R(n_contributing)) : one(R)

    for q in get_on_bits(support)
        mask = _bitmask(q)
        has_z = !iszero(pb.z & mask)
        has_x = !iszero(pb.x & mask)
        local_mean = R(
            has_z ?
            (has_x ? reference.y : reference.z) :
            reference.x,
        )
        iszero(local_mean) && continue
        pb_new = PauliBasis{N}(pb.z & ~mask, pb.x & ~mask)
        _add_term!(result, pb_new, R(c) * local_mean * scale)
    end

    return result
end

single_site_mean_field_decouple(
    pb::PauliBasis{N},
    c,
    reference::ProductBlochReference{N},
    normalize::Bool,
) where N =
    single_site_mean_field_decouple(
        pb,
        c,
        reference;
        normalize=normalize,
    )


"""
    single_site_mean_field_decouple!(
        O::PauliSum{N}, reference, max_weight::Int; normalize=false
    )

Replace every term of `O` whose weight is greater than `max_weight` by one
application of [`single_site_mean_field_decouple`](@ref). Terms at or below the
threshold are unchanged.

One SMFD application lowers a term by exactly one site. Consequently, this
operation bounds the result by `max_weight` only when the input was already
bounded by `max_weight + 1`. More-overweight inputs remain above the requested
threshold after this deliberately non-recursive step.

Set `normalize=true` to average each replacement over the sites whose local
Pauli expectation is nonzero in the supplied reference.
"""
function single_site_mean_field_decouple!(
    O::PauliSum{N,T},
    reference::Ket{N},
    max_weight::Int,
    ;
    normalize::Bool=false,
) where {N,T}
    high = [(pb, c) for (pb, c) in O if weight(pb) > max_weight]

    # Remove the complete snapshot before adding replacements. SMFD output can
    # itself remain overweight, so adding and removing one key at a time could
    # accidentally decouple a newly generated collision more than once.
    for (pb, _) in high
        delete!(O, pb)
    end
    for (pb, c) in high
        sum!(
            O,
            single_site_mean_field_decouple(
                pb,
                c,
                reference;
                normalize=normalize,
            ),
        )
    end

    return O
end

function single_site_mean_field_decouple!(
    O::PauliSum{N,T},
    reference::ProductDensityReference{N},
    max_weight::Int,
    ;
    normalize::Bool=false,
) where {N,T}
    high = [(pb, c) for (pb, c) in O if weight(pb) > max_weight]

    for (pb, _) in high
        delete!(O, pb)
    end
    for (pb, c) in high
        sum!(
            O,
            single_site_mean_field_decouple(
                pb,
                c,
                reference;
                normalize=normalize,
            ),
        )
    end

    return O
end

function single_site_mean_field_decouple!(
    O::PauliSum{N,T},
    reference::ProductBlochReference{N},
    max_weight::Int,
    ;
    normalize::Bool=false,
) where {N,T}
    high = [(pb, c) for (pb, c) in O if weight(pb) > max_weight]

    for (pb, _) in high
        delete!(O, pb)
    end
    for (pb, c) in high
        sum!(
            O,
            single_site_mean_field_decouple(
                pb,
                c,
                reference;
                normalize=normalize,
            ),
        )
    end

    return O
end

single_site_mean_field_decouple!(
    O::PauliSum{N},
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    max_weight::Int,
    normalize::Bool,
) where N =
    single_site_mean_field_decouple!(
        O,
        reference,
        max_weight;
        normalize=normalize,
    )


"""
    recursive_single_site_mean_field_decouple(
        pb::PauliBasis{N}, c, reference, max_weight::Int;
        normalize=false
    ) -> PauliSum

Repeatedly apply [`single_site_mean_field_decouple`](@ref) to `c * pb` until
every surviving term has weight at most `max_weight`.

For an input of weight `max_weight + 1`, this is exactly the original one-pass
SMFD rule. Additional passes activate only when the input is more than one
site above the cutoff: a weight-`max_weight + r` term receives `r` one-site
decoupling layers.

With `normalize=true`, every layer averages over its currently contributing
sites. Consequently, normalized recursive SMFD preserves the supplied product
reference expectation layer by layer while removing the ordered-removal
combinatorial overcounting.
"""
function recursive_single_site_mean_field_decouple(
    pb::PauliBasis{N},
    c::T,
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    max_weight::Int,
    ;
    normalize::Bool=false,
) where {N,T}
    max_weight >= 0 || throw(ArgumentError("max_weight must be nonnegative"))
    if weight(pb) <= max_weight
        result = PauliSum(N, T)
        result[pb] = c
        return result
    end

    result = single_site_mean_field_decouple(
        pb,
        c,
        reference;
        normalize=normalize,
    )
    return recursive_single_site_mean_field_decouple!(
        result,
        reference,
        max_weight;
        normalize=normalize,
    )
end

recursive_single_site_mean_field_decouple(
    pb::PauliBasis{N},
    c,
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    max_weight::Int,
    normalize::Bool,
) where N =
    recursive_single_site_mean_field_decouple(
        pb,
        c,
        reference,
        max_weight;
        normalize=normalize,
    )


"""
    recursive_single_site_mean_field_decouple!(
        O::PauliSum{N}, reference, max_weight::Int; normalize=false
    )

In-place recursive SMFD. The existing
[`single_site_mean_field_decouple!`](@ref) remains a deliberately one-pass
operation; this separate function repeats that operation only while terms
remain above `max_weight`.
"""
function recursive_single_site_mean_field_decouple!(
    O::PauliSum{N},
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    max_weight::Int,
    ;
    normalize::Bool=false,
) where N
    max_weight >= 0 || throw(ArgumentError("max_weight must be nonnegative"))
    while any(weight(pb) > max_weight for pb in keys(O))
        single_site_mean_field_decouple!(
            O,
            reference,
            max_weight;
            normalize=normalize,
        )
    end
    return O
end

recursive_single_site_mean_field_decouple!(
    O::PauliSum{N},
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    max_weight::Int,
    normalize::Bool,
) where N =
    recursive_single_site_mean_field_decouple!(
        O,
        reference,
        max_weight;
        normalize=normalize,
    )
