
"""
    expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, ket::Ket{N})

⟨k|P|k⟩ for a computational-basis ket: nonzero only for diagonal Paulis
(`x == 0`), where it equals `(-1)^popcount(z & k) ⋅ coeff(p)`.
"""
function expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, ket::Ket{N}) where N
    sgn = count_ones(p.z & ket.v)
    return PHASE_TBL[2*sgn%4+1] * (p.x == 0) * coeff(p)
end



"""
    expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, d::Union{Dyad{N}, DyadBasis{N}})

tr(P ⋅ |ket⟩⟨bra|) = ⟨bra|P|ket⟩: nonzero only when the Pauli's `x` string
connects the two basis states (`ket ⊻ bra == x`). Note the dyad's own
coefficient is *not* included — `DyadSum` methods carry coefficients in the
dictionary values.
"""
function expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, d::Union{Dyad{N}, DyadBasis{N}}) where N
    sgn = count_ones(p.z & d.bra.v)  # sgn <j| = <j| z
    val = d.ket.v ⊻ d.bra.v == p.x # <j|x|i>
    return val * PHASE_TBL[(symplectic_phase(p) + 2*sgn)%4 + 1]*coeff(p)
end

"""
    expectation_value(p::PauliSum{N,T}, d::Union{Ket{N}, Dyad{N}, DyadBasis{N}})

Expectation value of a sum of Paulis against a basis state or dyad —
the coefficient-weighted sum of the per-term expectation values.
"""
function expectation_value(p::PauliSum{N,T}, d::Union{Ket{N}, Dyad{N}, DyadBasis{N}}) where {N,T}
    eval = zero(T)
    for (pi,ci) in p
        eval += expectation_value(pi, d) * ci
    end
    return eval 
end

function expectation_value(p::Union{PauliBasis{N}, Pauli{N}}, d::DyadSum{N,T}) where {N,T}
    eval = zero(T)
    for (di,ci) in d
        eval += expectation_value(p, di) * ci
    end
    return eval 
end

function expectation_value(p::PauliSum{N,T}, d::DyadSum{N,T}) where {N,T}
    eval = zero(T)
    for (pi,ci) in p
        for (dj,cj) in d
            eval += expectation_value(pi, dj) * ci * cj
        end
    end
    return eval 
end

"""
    expectation_value(O::AnyPauliSum, v::KetSum)

⟨v|O|v⟩ for a linear combination of basis states, including all cross terms
⟨k₂|P|k₁⟩. Works for both `PauliSum` and `SparsePauliVector`.
"""
function expectation_value(O::AnyPauliSum, v::KetSum)
    ev = 0
    for (p,c) in O
        for (k1,c1) in v
            ev += expectation_value(p,k1)*c*c1'*c1
            for (k2,c2) in v
                k2 != k1 || continue
                ev += matrix_element(k2', p, k1)*c*c2'*c1
            end
        end
    end
    return ev
end


"""
    matrix_element(b::Bra{N}, p::PauliBasis{N}, k::Ket{N})

⟨b|P|k⟩: nonzero only when `b.v == k.v ⊻ p.x`, i.e. each Pauli connects a
ket to exactly one bra. Also accepts `AnyPauliSum` and `KetSum` arguments.
"""
function matrix_element(b::Bra{N}, p::PauliBasis{N}, k::Ket{N}) where N
    sgn = count_ones(p.z & b.v)  # sgn <j| = <j| z
    val = k.v ⊻ b.v == p.x # <j|x|i>
    return val * PHASE_TBL[(symplectic_phase(p) + 2*sgn)%4 + 1]
end

function matrix_element(b::Bra{N}, p::AnyPauliSum{N,T}, k::Ket{N}) where {N,T}
    eval = zero(T)
    for (pi,ci) in p
        eval += matrix_element(b, pi, k) * ci
    end
    return eval 
end

function matrix_element(b::KetSum{N}, p::PauliBasis{N}, k::KetSum{N}) where {N}
    if length(k) < length(b)
        pk = p*k
        return inner_product(b, pk)
    else 
        pb = p*b
        return inner_product(pb, k)
    end
end

function matrix_element(b::KetSum{N}, p::AnyPauliSum{N}, k::KetSum{N}) where {N}
    σ = p*k
    return inner_product(b,σ)
end

"""
    expectation_value(p, reference::ProductDensityReference)

Evaluate `tr(reference * p)` without constructing a dense density matrix.
Only pure-`Z` Pauli strings have nonzero expectation values.
"""
function expectation_value(
    p::PauliBasis{N},
    reference::ProductDensityReference{N,T},
) where {N,T}
    iszero(p.x) || return zero(T)
    return reference.magnetization ^ count_ones(p.z)
end

function expectation_value(
    p::Pauli{N},
    reference::ProductDensityReference{N,T},
) where {N,T}
    return coeff(p) * expectation_value(PauliBasis(p), reference)
end

function expectation_value(
    O::AnyPauliSum{N,T},
    reference::ProductDensityReference{N,R},
) where {N,T,R}
    S = promote_type(T, R)
    result = zero(S)
    for (p, c) in O
        result += c * expectation_value(p, reference)
    end
    return result
end

"""
    expectation_value(p, reference::ProductBlochReference)

Evaluate a Pauli operator against a uniform product state from its local Bloch
components. Since the reference is a product state, a Pauli-string expectation
is the product of its one-site expectations.
"""
function expectation_value(
    p::PauliBasis{N},
    reference::ProductBlochReference{N,T},
) where {N,T}
    n_y = count_ones(p.z & p.x)
    n_x = count_ones(p.x & ~p.z)
    n_z = count_ones(p.z & ~p.x)
    return reference.x^n_x * reference.y^n_y * reference.z^n_z
end

function expectation_value(
    p::Pauli{N},
    reference::ProductBlochReference{N,T},
) where {N,T}
    return coeff(p) * expectation_value(PauliBasis(p), reference)
end

function expectation_value(
    O::AnyPauliSum{N,T},
    reference::ProductBlochReference{N,R},
) where {N,T,R}
    S = promote_type(T, R)
    result = zero(S)
    for (p, c) in O
        result += c * expectation_value(p, reference)
    end
    return result
end
