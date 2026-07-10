using Pkg
Pkg.activate("..")
Pkg.instantiate()

using PauliOperators
using LinearAlgebra
using Plots

function heisenberg_1d(N::Int; J::Real=1.0)
    H = PauliSum(N, ComplexF64)
    for i in 1:(N-1)
        H[PauliBasis(Pauli(N; X=[i, i+1]))] = 0.1J + 0im
        H[PauliBasis(Pauli(N; Y=[i, i+1]))] = 0.1J + 0im
        H[PauliBasis(Pauli(N; Z=[i, i+1]))] = 2J + 0im
    end
    return H
end

H = heisenberg_1d(4)
Hm = Hermitian(Matrix(H))
vals, vecs = eigen(Hm)

heat = abs.(vecs) .^ 2

p = heatmap(
    1:size(heat, 2),
    1:size(heat, 1),
    heat;
    xlabel="Eigenvector index",
    ylabel="Basis-state index",
    title="Hamiltonian eigenvector heatmap (|⟨basis|eig⟩|^2)",
    color=:thermal,
    clims=(0, 1),
    size=(900, 600),
    margin=5Plots.mm,
)

outfile = joinpath(@__DIR__, "eigenvector_heatmap.png")
savefig(p, outfile)
println("Saved heatmap to: $outfile")
println("Eigenvalues:")
display(vals)
