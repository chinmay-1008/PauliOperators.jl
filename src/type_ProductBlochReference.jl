"""
    ProductBlochReference(N, x, y, z)
    ProductBlochReference(N; theta, phi=0)

Uniform single-qubit product reference on `N` sites. The local density matrix
has Bloch vector `(x, y, z)`, so the local Pauli expectations are
`⟨X⟩ = x`, `⟨Y⟩ = y`, and `⟨Z⟩ = z`.

The angular constructor creates the pure product state

`cos(theta/2)|0⟩ + exp(im*phi)sin(theta/2)|1⟩`

on every site.
"""
struct ProductBlochReference{N,T<:AbstractFloat}
    x::T
    y::T
    z::T
end

function ProductBlochReference(N::Integer, x::Real, y::Real, z::Real)
    1 <= N <= 128 || throw(ArgumentError("number of qubits must be in 1:128"))
    x_float, y_float, z_float = promote(float(x), float(y), float(z))
    all(isfinite, (x_float, y_float, z_float)) ||
        throw(ArgumentError("Bloch-vector components must be finite"))

    norm_sq = x_float^2 + y_float^2 + z_float^2
    norm_sq <= one(norm_sq) ||
        throw(ArgumentError("Bloch-vector norm must not exceed one"))

    return ProductBlochReference{Int(N),typeof(x_float)}(
        x_float,
        y_float,
        z_float,
    )
end

function ProductBlochReference(
    N::Integer;
    theta::Real,
    phi::Real=zero(theta),
)
    isfinite(theta) || throw(ArgumentError("theta must be finite"))
    isfinite(phi) || throw(ArgumentError("phi must be finite"))
    1 <= N <= 128 || throw(ArgumentError("number of qubits must be in 1:128"))
    transverse = sin(theta)
    x, y, z = promote(
        transverse * cos(phi),
        transverse * sin(phi),
        cos(theta),
    )
    # The trigonometric form is a unit Bloch vector by construction. Build it
    # directly so a one-ulp sin/cos identity error is not rejected by the
    # strict component constructor above.
    return ProductBlochReference{Int(N),typeof(x)}(x, y, z)
end

function Base.show(io::IO, reference::ProductBlochReference{N}) where {N}
    print(
        io,
        "ProductBlochReference(",
        N,
        ", ",
        reference.x,
        ", ",
        reference.y,
        ", ",
        reference.z,
        ")",
    )
end
