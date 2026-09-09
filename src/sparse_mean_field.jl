# ============================================================
# Dict-free mean-field transforms for SparsePauliVector.
#
# Replacement terms are written directly as packed (z, x, coefficient)
# triples into the reusable SPV workspace, then sorted and deduplicated by
# the same merge kernel used by vectorized evolution. No PauliBasis objects
# or intermediate PauliSum dictionaries are created in the term loop.
# ============================================================

@inline _lowbit(x::W) where {W<:Unsigned} = x & (~x + one(W))

@inline function _push_workspace!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    z::W,
    x::W,
    c,
) where {N,W,T}
    m += 1
    if m > length(v.ws)
        resize!(v.ws, max(m, max(16, 2 * length(v.ws))))
    end
    @inbounds v.ws[m] = (z, x, convert(T, c))
    return m
end

function _merge_before_mean_field!(v::SparsePauliVector)
    v.an == 0 && return v
    m = _gather_append!(v)
    _sort_ws!(v.ws, 1, m)
    _merge_spv!(v, m, NOFILTER)
    return v
end

function _replace_from_workspace!(v::SparsePauliVector, m::Int)
    _sort_ws!(v.ws, 1, m)
    v.n = 0
    _merge_spv!(v, m, NOFILTER)
    return v
end


# ------------------------------------------------------------
# Computational-basis reference
# ------------------------------------------------------------

function _emit_ket_combinations!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    available::W,
    choose::Int,
    chosen::W,
    z_only::W,
    y_z_mask::W,
    x::W,
    ket_bits::W,
    c::T,
    factor::Int,
) where {N,W,T}
    if choose == 0
        parity = count_ones((z_only & ~chosen) & ket_bits) & 1
        sign = 1 - 2 * parity
        return _push_workspace!(
            v,
            m,
            y_z_mask | chosen,
            x,
            c * sign * factor,
        )
    end

    remaining = available
    while count_ones(remaining) >= choose
        bit = _lowbit(remaining)
        remaining &= ~bit
        m = _emit_ket_combinations!(
            v,
            m,
            remaining,
            choose - 1,
            chosen | bit,
            z_only,
            y_z_mask,
            x,
            ket_bits,
            c,
            factor,
        )
    end
    return m
end

function _emit_mean_field_term!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    z::W,
    x::W,
    c::T,
    reference::Ket{N},
    k::Int,
) where {N,W,T}
    n_xy = count_ones(x)
    n_xy > k && return m

    z_only = z & ~x
    n_z = count_ones(z_only)
    y_z_mask = z & x
    budget = k - n_xy
    ket_bits = ((reference.v % UInt128) % W)

    for t in 0:min(budget, n_z)
        factor = _partial_alt_binom(n_z - t, budget - t)
        iszero(factor) && continue
        m = _emit_ket_combinations!(
            v,
            m,
            z_only,
            t,
            zero(W),
            z_only,
            y_z_mask,
            x,
            ket_bits,
            c,
            factor,
        )
    end
    return m
end


# ------------------------------------------------------------
# Uniform diagonal product-density reference
# ------------------------------------------------------------

function _emit_product_combinations!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    available::W,
    choose::Int,
    chosen::W,
    y_z_mask::W,
    x::W,
    contribution,
) where {N,W,T}
    if choose == 0
        return _push_workspace!(v, m, y_z_mask | chosen, x, contribution)
    end

    remaining = available
    while count_ones(remaining) >= choose
        bit = _lowbit(remaining)
        remaining &= ~bit
        m = _emit_product_combinations!(
            v,
            m,
            remaining,
            choose - 1,
            chosen | bit,
            y_z_mask,
            x,
            contribution,
        )
    end
    return m
end

function _emit_mean_field_term!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    z::W,
    x::W,
    c::T,
    reference::ProductDensityReference{N},
    k::Int,
) where {N,W,T}
    n_xy = count_ones(x)
    n_xy > k && return m

    z_only = z & ~x
    n_z = count_ones(z_only)
    y_z_mask = z & x
    budget = k - n_xy
    magnetization = reference.magnetization

    for t in 0:min(budget, n_z)
        factor = _partial_alt_binom(n_z - t, budget - t)
        iszero(factor) && continue
        mean_rest = magnetization ^ (n_z - t)
        iszero(mean_rest) && continue
        contribution = c * mean_rest * factor
        m = _emit_product_combinations!(
            v,
            m,
            z_only,
            t,
            zero(W),
            y_z_mask,
            x,
            contribution,
        )
    end
    return m
end


# ------------------------------------------------------------
# Uniform product-Bloch reference
# ------------------------------------------------------------

@inline function _packed_bloch_mean(
    z::W,
    x::W,
    bit::W,
    reference::ProductBlochReference,
) where {W<:Unsigned}
    has_z = !iszero(z & bit)
    has_x = !iszero(x & bit)
    return has_z ?
           (has_x ? reference.y : reference.z) :
           reference.x
end

function _emit_bloch_combinations!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    available::W,
    choose::Int,
    chosen::W,
    support::W,
    z::W,
    x::W,
    c::T,
    factor::Int,
    reference::ProductBlochReference{N},
) where {N,W,T}
    if choose == 0
        mean_rest = one(T)
        remaining = support & ~chosen
        while !iszero(remaining)
            bit = _lowbit(remaining)
            mean_rest *= _packed_bloch_mean(z, x, bit, reference)
            remaining &= ~bit
        end
        contribution = c * mean_rest * factor
        iszero(contribution) && return m
        return _push_workspace!(
            v,
            m,
            z & chosen,
            x & chosen,
            contribution,
        )
    end

    remaining = available
    while count_ones(remaining) >= choose
        bit = _lowbit(remaining)
        remaining &= ~bit
        m = _emit_bloch_combinations!(
            v,
            m,
            remaining,
            choose - 1,
            chosen | bit,
            support,
            z,
            x,
            c,
            factor,
            reference,
        )
    end
    return m
end

function _emit_mean_field_term!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    z::W,
    x::W,
    c::T,
    reference::ProductBlochReference{N},
    k::Int,
) where {N,W,T}
    support = z | x
    n = count_ones(support)
    for t in 0:min(k, n)
        factor = _partial_alt_binom(n - t, k - t)
        iszero(factor) && continue
        m = _emit_bloch_combinations!(
            v,
            m,
            support,
            t,
            zero(W),
            support,
            z,
            x,
            c,
            factor,
            reference,
        )
    end
    return m
end


# ------------------------------------------------------------
# Generic KetSum reference
# ------------------------------------------------------------

function _local_pauli_mean(
    reference::KetSum{N,S},
    zmask::Int128,
    xmask::Int128,
    norm_sq,
    ::Type{R},
) where {N,S,R}
    symplectic = (4 - count_ones(zmask & xmask) % 4) % 4
    result = zero(R)
    for (ket, amplitude) in reference
        bra_bits = ket.v ⊻ xmask
        bra_amplitude = get(reference, Ket{N}(bra_bits), zero(S))
        sign = count_ones(zmask & bra_bits) & 1
        phase = PHASE_TBL[(symplectic + 2 * sign) % 4 + 1]
        result += conj(R(bra_amplitude)) * R(amplitude) * phase
    end
    return result / norm_sq
end

function _local_mean_tables(reference::KetSum{N,S}) where {N,S}
    R = promote_type(S, ComplexF64)
    norm_sq = zero(real(R))
    for amplitude in values(reference)
        norm_sq += abs2(amplitude)
    end
    iszero(norm_sq) &&
        throw(ArgumentError("mean-field reference KetSum must have nonzero norm"))

    mean_z = Vector{R}(undef, N)
    mean_x = Vector{R}(undef, N)
    mean_y = Vector{R}(undef, N)
    for q in 1:N
        mask = Int128(1) << (q - 1)
        mean_z[q] = _local_pauli_mean(
            reference,
            mask,
            Int128(0),
            norm_sq,
            R,
        )
        mean_x[q] = _local_pauli_mean(
            reference,
            Int128(0),
            mask,
            norm_sq,
            R,
        )
        mean_y[q] = _local_pauli_mean(
            reference,
            mask,
            mask,
            norm_sq,
            R,
        )
    end
    return mean_z, mean_x, mean_y
end

@inline function _mean_at_site(
    z::W,
    x::W,
    bit::W,
    q::Int,
    mean_z,
    mean_x,
    mean_y,
) where {W<:Unsigned}
    has_z = !iszero(z & bit)
    has_x = !iszero(x & bit)
    return has_z ? (has_x ? mean_y[q] : mean_z[q]) : mean_x[q]
end

function _emit_ketsum_combinations!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    available::W,
    choose::Int,
    chosen::W,
    support::W,
    z::W,
    x::W,
    c::T,
    factor::Int,
    mean_z,
    mean_x,
    mean_y,
) where {N,W,T}
    if choose == 0
        mean_rest = one(eltype(mean_z))
        remaining = support & ~chosen
        while !iszero(remaining)
            bit = _lowbit(remaining)
            q = trailing_zeros(bit) + 1
            mean_rest *= _mean_at_site(
                z,
                x,
                bit,
                q,
                mean_z,
                mean_x,
                mean_y,
            )
            remaining &= ~bit
        end
        contribution = c * mean_rest * factor
        iszero(contribution) && return m
        return _push_workspace!(
            v,
            m,
            z & chosen,
            x & chosen,
            contribution,
        )
    end

    remaining = available
    while count_ones(remaining) >= choose
        bit = _lowbit(remaining)
        remaining &= ~bit
        m = _emit_ketsum_combinations!(
            v,
            m,
            remaining,
            choose - 1,
            chosen | bit,
            support,
            z,
            x,
            c,
            factor,
            mean_z,
            mean_x,
            mean_y,
        )
    end
    return m
end

function _emit_mean_field_term!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    z::W,
    x::W,
    c::T,
    reference::KetSum{N},
    k::Int,
    mean_z,
    mean_x,
    mean_y,
) where {N,W,T}
    support = z | x
    n = count_ones(support)
    for t in 0:min(k, n)
        factor = _partial_alt_binom(n - t, k - t)
        iszero(factor) && continue
        m = _emit_ketsum_combinations!(
            v,
            m,
            support,
            t,
            zero(W),
            support,
            z,
            x,
            c,
            factor,
            mean_z,
            mean_x,
            mean_y,
        )
    end
    return m
end


# ------------------------------------------------------------
# Public in-place mean-field API
# ------------------------------------------------------------

function mean_field_factorize!(
    v::SparsePauliVector{N,W,T},
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    k::Int,
) where {N,W,T}
    _merge_before_mean_field!(v)
    k >= 0 && _spv_is_weight_bounded(v, k) && return v
    m = 0
    @inbounds for i in 1:v.n
        z = v.z[i]
        x = v.x[i]
        c = v.c[i]
        if count_ones(z | x) <= k
            m = _push_workspace!(v, m, z, x, c)
        else
            m = _emit_mean_field_term!(v, m, z, x, c, reference, k)
        end
    end
    return _replace_from_workspace!(v, m)
end

function mean_field_factorize!(
    v::SparsePauliVector{N,W,T},
    reference::KetSum{N},
    k::Int,
) where {N,W,T}
    _merge_before_mean_field!(v)
    k >= 0 && _spv_is_weight_bounded(v, k) && return v
    mean_z, mean_x, mean_y = _local_mean_tables(reference)
    m = 0
    @inbounds for i in 1:v.n
        z = v.z[i]
        x = v.x[i]
        c = v.c[i]
        if count_ones(z | x) <= k
            m = _push_workspace!(v, m, z, x, c)
        else
            m = _emit_mean_field_term!(
                v,
                m,
                z,
                x,
                c,
                reference,
                k,
                mean_z,
                mean_x,
                mean_y,
            )
        end
    end
    return _replace_from_workspace!(v, m)
end


# ------------------------------------------------------------
# Single-site mean-field decoupling
# ------------------------------------------------------------

function _emit_smfd_term!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    z::W,
    x::W,
    c::T,
    reference::Ket{N},
    normalize::Bool,
) where {N,W,T}
    z_only = z & ~x
    n_z = count_ones(z_only)
    iszero(n_z) && return m
    scale = normalize ? inv(T(n_z)) : one(T)
    ket_bits = ((reference.v % UInt128) % W)
    remaining = z_only
    while !iszero(remaining)
        bit = _lowbit(remaining)
        local_mean = iszero(ket_bits & bit) ? 1 : -1
        m = _push_workspace!(
            v,
            m,
            z & ~bit,
            x,
            c * local_mean * scale,
        )
        remaining &= ~bit
    end
    return m
end

function _emit_smfd_term!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    z::W,
    x::W,
    c::T,
    reference::ProductDensityReference{N},
    normalize::Bool,
) where {N,W,T}
    z_only = z & ~x
    n_z = count_ones(z_only)
    iszero(n_z) && return m
    local_mean = reference.magnetization
    iszero(local_mean) && return m
    scale = normalize ? inv(n_z) : 1
    remaining = z_only
    while !iszero(remaining)
        bit = _lowbit(remaining)
        m = _push_workspace!(
            v,
            m,
            z & ~bit,
            x,
            c * local_mean * scale,
        )
        remaining &= ~bit
    end
    return m
end

function _emit_smfd_term!(
    v::SparsePauliVector{N,W,T},
    m::Int,
    z::W,
    x::W,
    c::T,
    reference::ProductBlochReference{N},
    normalize::Bool,
) where {N,W,T}
    support = z | x
    n_contributing = 0
    remaining = support
    while !iszero(remaining)
        bit = _lowbit(remaining)
        n_contributing += !iszero(
            _packed_bloch_mean(z, x, bit, reference),
        )
        remaining &= ~bit
    end
    iszero(n_contributing) && return m

    scale = normalize ? inv(T(n_contributing)) : one(T)
    remaining = support
    while !iszero(remaining)
        bit = _lowbit(remaining)
        local_mean = _packed_bloch_mean(z, x, bit, reference)
        if !iszero(local_mean)
            m = _push_workspace!(
                v,
                m,
                z & ~bit,
                x & ~bit,
                c * local_mean * scale,
            )
        end
        remaining &= ~bit
    end
    return m
end

function single_site_mean_field_decouple!(
    v::SparsePauliVector{N,W,T},
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    max_weight::Int;
    normalize::Bool=false,
) where {N,W,T}
    _merge_before_mean_field!(v)
    max_weight >= 0 &&
        _spv_is_weight_bounded(v, max_weight) &&
        return v
    m = 0
    @inbounds for i in 1:v.n
        z = v.z[i]
        x = v.x[i]
        c = v.c[i]
        if count_ones(z | x) <= max_weight
            m = _push_workspace!(v, m, z, x, c)
        else
            m = _emit_smfd_term!(
                v,
                m,
                z,
                x,
                c,
                reference,
                normalize,
            )
        end
    end
    return _replace_from_workspace!(v, m)
end

single_site_mean_field_decouple!(
    v::SparsePauliVector{N},
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    max_weight::Int,
    normalize::Bool,
) where {N} =
    single_site_mean_field_decouple!(
        v,
        reference,
        max_weight;
        normalize=normalize,
    )

"""
    recursive_single_site_mean_field_decouple!(
        v::SparsePauliVector{N}, reference, max_weight::Int; normalize=false
    )

Packed-SPV implementation of recursive SMFD. Reuses the one-pass workspace
kernel until the requested weight bound is reached; the original one-pass
method remains unchanged.
"""
function recursive_single_site_mean_field_decouple!(
    v::SparsePauliVector{N},
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
    _merge_before_mean_field!(v)
    while !_spv_is_weight_bounded(v, max_weight)
        single_site_mean_field_decouple!(
            v,
            reference,
            max_weight;
            normalize=normalize,
        )
    end
    return v
end

recursive_single_site_mean_field_decouple!(
    v::SparsePauliVector{N},
    reference::Union{
        Ket{N},
        ProductDensityReference{N},
        ProductBlochReference{N},
    },
    max_weight::Int,
    normalize::Bool,
) where N =
    recursive_single_site_mean_field_decouple!(
        v,
        reference,
        max_weight;
        normalize=normalize,
    )


# ------------------------------------------------------------
# Truncation-strategy integration
# ------------------------------------------------------------

function _apply!(
    v::SparsePauliVector{N,W,T},
    strategy::MeanFieldTruncation{N},
) where {N,W,T}
    return mean_field_factorize!(
        v,
        strategy.reference,
        strategy.max_weight,
    )
end

function _apply!(
    v::SparsePauliVector{N,W,T},
    strategy::SingleSiteMeanFieldDecoupling{N},
) where {N,W,T}
    return single_site_mean_field_decouple!(
        v,
        strategy.reference,
        strategy.max_weight;
        normalize=strategy.normalize,
    )
end

function _apply!(
    v::SparsePauliVector{N,W,T},
    strategy::RecursiveSingleSiteMeanFieldDecoupling{N},
) where {N,W,T}
    return recursive_single_site_mean_field_decouple!(
        v,
        strategy.reference,
        strategy.max_weight;
        normalize=strategy.normalize,
    )
end

function _spv_is_weight_bounded(v::SparsePauliVector, max_weight::Int)
    @inbounds for i in 1:v.n
        count_ones(v.z[i] | v.x[i]) <= max_weight || return false
    end
    return true
end
