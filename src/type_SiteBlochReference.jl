"""
    SiteBlochReference(x, y, z)
    SiteBlochReference(moments::AbstractMatrix)
    SiteBlochReference(reference::ProductBlochReference)

Single-qubit product reference whose Bloch vector may differ from site to site.
Site `q` has the local density matrix `(I + x[q] X + y[q] Y + z[q] Z) / 2`, so
`⟨X_q⟩ = x[q]`, `⟨Y_q⟩ = y[q]`, and `⟨Z_q⟩ = z[q]`.

The matrix constructor takes an `N × 3` array whose rows are the per-site
`(⟨X⟩, ⟨Y⟩, ⟨Z⟩)`. Converting a uniform [`ProductBlochReference`](@ref) gives a
reference that every mean-field routine treats identically to the uniform one.
"""
struct SiteBlochReference{N,T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
end

function SiteBlochReference(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    z::AbstractVector{<:Real},
)
    N = length(x)
    1 <= N <= 128 || throw(ArgumentError("number of qubits must be in 1:128"))
    length(y) == N && length(z) == N ||
        throw(DimensionMismatch("x, y and z must have the same length"))

    T = float(promote_type(eltype(x), eltype(y), eltype(z)))
    xs, ys, zs = Vector{T}(x), Vector{T}(y), Vector{T}(z)

    for q in 1:N
        all(isfinite, (xs[q], ys[q], zs[q])) ||
            throw(ArgumentError("Bloch-vector components must be finite (site $q)"))
        xs[q]^2 + ys[q]^2 + zs[q]^2 <= one(T) + 8 * eps(T) ||
            throw(ArgumentError("Bloch-vector norm must not exceed one (site $q)"))
    end

    return SiteBlochReference{N,T}(xs, ys, zs)
end

function SiteBlochReference(moments::AbstractMatrix{<:Real})
    size(moments, 2) == 3 ||
        throw(DimensionMismatch("moments must be N × 3, rows (⟨X⟩, ⟨Y⟩, ⟨Z⟩)"))
    return SiteBlochReference(moments[:, 1], moments[:, 2], moments[:, 3])
end

SiteBlochReference(reference::ProductBlochReference{N}) where {N} =
    SiteBlochReference(
        fill(reference.x, N),
        fill(reference.y, N),
        fill(reference.z, N),
    )

# Local mean of the factor on site `q` of a Pauli string with symplectic bits
# `has_z`, `has_x` (Y when both are set).
@inline _site_bloch_mean(reference::SiteBlochReference, q::Int, has_z::Bool, has_x::Bool) =
    has_z ? (has_x ? reference.y[q] : reference.z[q]) : reference.x[q]

function Base.show(io::IO, reference::SiteBlochReference{N}) where {N}
    print(io, "SiteBlochReference(", N, " sites)")
end
