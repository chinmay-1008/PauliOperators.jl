using Pkg
Pkg.activate("../")
Pkg.instantiate()

using PauliOperators
using LinearAlgebra

function heisenberg_xxx(N::Int; J::Real=1.0)
    H = PauliSum(N, ComplexF64)
    for i in 1:(N-1)
        H[PauliBasis(Pauli(N; X=[i, i+1]))] = 1J + 0im
        H[PauliBasis(Pauli(N; Y=[i, i+1]))] = 1J + 0im
        H[PauliBasis(Pauli(N; Z=[i, i+1]))] = 1J + 0im
    end
    return H
end

N = 6
J = 1.0
H = heisenberg_xxx(N; J=J)
Hm = Hermitian(Matrix(H))
vals, vecs = eigen(Hm)

E0 = vals[1]
ψ0 = vecs[:, 1]
probs0 = abs.(ψ0) .^ 2
order0 = partialsortperm(probs0, 1:min(8, length(probs0)), rev=true)

println("Ground-state energy E0 = $(round(E0; digits=8))")
println("Ground-state computational-basis amplitudes (top 8):")
for idx in order0
    println("  |$(idx-1)⟩ : amplitude=$(round(ψ0[idx]; digits=6)), weight=$(round(probs0[idx]; digits=6))")
end
