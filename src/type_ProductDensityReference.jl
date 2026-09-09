"""
    ProductDensityReference(N::Int, p::Real)

Uniform diagonal product-density reference on `N` qubits,

`rho(p) = [p |0><0| + (1-p) |1><1|]` tensored over all `N` qubits.

The local Pauli expectations are `⟨X⟩ = ⟨Y⟩ = 0` and
`⟨Z⟩ = 2p - 1`.
"""
struct ProductDensityReference{N,T<:AbstractFloat}
    p::T
    magnetization::T
end

function ProductDensityReference(N::Int, p::Real)
    1 <= N <= 128 || throw(ArgumentError("number of qubits must be in 1:128"))
    p_float = float(p)
    isfinite(p_float) || throw(ArgumentError("p must be finite"))
    zero(p_float) <= p_float <= one(p_float) ||
        throw(ArgumentError("p must satisfy 0 <= p <= 1"))
    return ProductDensityReference{N,typeof(p_float)}(
        p_float,
        2 * p_float - one(p_float),
    )
end

function Base.show(io::IO, ref::ProductDensityReference{N}) where {N}
    print(io, "ProductDensityReference(", N, ", ", ref.p, ")")
end
