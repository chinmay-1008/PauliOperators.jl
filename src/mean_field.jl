"""
    _partial_alt_binom(n::Int, k_max::Int) -> Int

Compute `Σ_{m=0}^{min(k_max,n)} C(n,m) (-1)^m`.

Edge cases:
- `k_max < 0` returns `0`.
- `k_max >= n` returns `1` if `n == 0` else `0` (the full alternating row sum vanishes).
"""
function _partial_alt_binom(n::Int, k_max::Int)
    k_max < 0 && return 0
    if k_max >= n
        return n == 0 ? 1 : 0
    end
    result = 1
    binom  = 1  # C(n, 0)
    for m in 1:k_max
        binom  = binom * (n - m + 1) ÷ m   # C(n, m) — exact integer update
        result += iseven(m) ? binom : -binom
    end
    return result
end


"""
    _foreach_combination(f, n::Int, k::Int)

Invoke `f(buf)` for every size-`k` subset of `{1,…,n}`, where `buf::Vector{Int}`
is a shared buffer of length `k` holding the indices in increasing order.
Empty iff `k < 0` or `k > n`.
"""
function _foreach_combination(f::F, n::Int, k::Int) where {F}
    (k < 0 || k > n) && return
    if k == 0
        f(Int[])
        return
    end
    buf = Vector{Int}(undef, k)
    _rec_combo!(f, buf, 1, 1, n, k)
    return
end

function _rec_combo!(f::F, buf::Vector{Int}, depth::Int, start::Int,
                     n::Int, k::Int) where {F}
    if depth > k
        f(buf)
        return
    end
    for i in start:(n - k + depth)
        buf[depth] = i
        _rec_combo!(f, buf, depth + 1, i + 1, n, k)
    end
    return
end


@inline _bitmask(q::Int) = Int128(1) << (q - 1)

function _add_term!(O::PauliSum{N}, pb::PauliBasis{N}, c) where {N}
    if haskey(O, pb)
        O[pb] += c
    else
        O[pb] = c
    end
    return O
end


"""
    mean_field_factorize(pb::PauliBasis{N}, c, ψ::Ket{N}, k::Int) -> PauliSum{N,T}

Order-`k` mean-field factorization of the single Pauli term `c · pb` around the
computational-basis reference `ψ`.

Replaces `pb` with a sum of Pauli strings of weight ≤ `k` that is exact when
`k ≥ weight(pb)` and preserves `⟨ψ|·|ψ⟩` for every `k`. Uses the multinomial
fluctuation decomposition with `δP_j = P_j − ⟨P_j⟩ I`; on a computational-basis
reference only pure-Z qubits contribute non-trivially, so enumeration is over
subsets of the pure-Z qubit positions in `pb`.
"""
function mean_field_factorize(pb::PauliBasis{N}, c::T, ψ::Ket{N}, k::Int) where {N,T}
    result = PauliSum(N, T)

    n_xy = count_ones(pb.x)
    if n_xy > k
        return result
    end

    z_only = pb.z & ~pb.x
    z_pos  = get_on_bits(z_only)
    n_z    = length(z_pos)

    ε = Vector{Int}(undef, n_z)
    full_ε = 1
    for (i, q) in enumerate(z_pos)
        εi = ((ψ.v >> (q - 1)) & 1 == 1) ? -1 : 1
        ε[i]   = εi
        full_ε *= εi
    end

    y_z_mask = pb.z & pb.x   # Z-bits carried by Y qubits (unchanged by factorization)
    budget   = k - n_xy      # max |T|

    for t in 0:min(budget, n_z)
        f = _partial_alt_binom(n_z - t, budget - t)
        f == 0 && continue
        _foreach_combination(n_z, t) do T_idx
            T_mask = Int128(0)
            ε_rest = full_ε
            for i in T_idx
                T_mask |= Int128(1) << (z_pos[i] - 1)
                ε_rest *= ε[i]
            end
            pb_new  = PauliBasis{N}(y_z_mask | T_mask, pb.x)
            contrib = c * ε_rest * f
            _add_term!(result, pb_new, contrib)
        end
    end

    return result
end


"""
    mean_field_factorize(pb::PauliBasis{N}, c, ψ::KetSum{N}, k::Int) -> PauliSum

Order-`k` single-site mean-field factorization around a general `KetSum`
reference. Each local Pauli factor is decomposed as
`P_i = ⟨P_i⟩_ψ I + δP_i`; fluctuation products of order `> k` are discarded,
then the kept terms are collected back into ordinary Pauli strings.

Unlike the computational-basis `Ket` specialization, this generic `KetSum`
version uses all local one-site expectations, including nonzero `X` or `Y`
expectations when present. For entangled references this is a single-site
mean-field approximation and does not generally preserve the full reference
expectation value at low order.
"""
function mean_field_factorize(pb::PauliBasis{N}, c::T, ψ::KetSum{N,S}, k::Int) where {N,T,S}
    R = promote_type(T, S, ComplexF64)
    result = PauliSum(N, R)

    n = weight(pb)
    if k < 0
        return result
    elseif k >= n
        result[pb] = R(c)
        return result
    end

    norm_sq = inner_product(ψ, ψ)
    iszero(norm_sq) && throw(ArgumentError("mean-field reference KetSum must have nonzero norm"))

    pos = get_on_bits(pb.z | pb.x)
    means = Vector{R}(undef, n)
    for (i, q) in enumerate(pos)
        mask = _bitmask(q)
        local_pb = PauliBasis{N}(pb.z & mask, pb.x & mask)
        means[i] = R(matrix_element(ψ, local_pb, ψ) / norm_sq)
    end

    for t in 0:min(k, n)
        f = _partial_alt_binom(n - t, k - t)
        f == 0 && continue
        _foreach_combination(n, t) do T_idx
            T_mask = Int128(0)
            mean_rest = one(R)
            next_keep = 1
            for i in 1:n
                if next_keep <= length(T_idx) && T_idx[next_keep] == i
                    T_mask |= _bitmask(pos[i])
                    next_keep += 1
                else
                    mean_rest *= means[i]
                end
            end

            pb_new = PauliBasis{N}(pb.z & T_mask, pb.x & T_mask)
            contrib = R(c) * mean_rest * f
            if !iszero(contrib)
                _add_term!(result, pb_new, contrib)
            end
        end
    end

    return result
end


"""
    mean_field_factorize!(O::PauliSum{N,T}, ψ::Union{Ket{N},KetSum{N}}, k::Int)

In-place replacement of every term in `O` with `weight(pb) > k` by its order-`k`
mean-field factorization around `ψ`. See [`mean_field_factorize`](@ref).
"""
function mean_field_factorize!(O::PauliSum{N,T}, ψ::Ket{N}, k::Int) where {N,T}
    high = [pb for (pb, _) in O if weight(pb) > k]
    for pb in high
        c = pop!(O, pb)
        sum!(O, mean_field_factorize(pb, c, ψ, k))
    end
    return O
end

function mean_field_factorize!(O::PauliSum{N,T}, ψ::KetSum{N}, k::Int) where {N,T}
    high = [pb for (pb, _) in O if weight(pb) > k]
    for pb in high
        c = pop!(O, pb)
        sum!(O, mean_field_factorize(pb, c, ψ, k))
    end
    return O
end
