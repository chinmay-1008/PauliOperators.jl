using PauliOperators
using Test

@testset "Paulis.jl" begin
    include("test_operator_methods.jl")
    include("test_Pauli.jl")
    include("test_Ket.jl")
    include("test_multiplication.jl")
    include("test_addition.jl")
    include("test_allocations.jl")
    include("test_stochastic.jl")
    include("test_phase1.jl")
    include("test_truncation.jl")
    include("test_sparse_pauli_vector.jl")
    include("test_spv_equivalence.jl")
    include("test_spv_evolution.jl")
    include("test_spv_allocations.jl")
    include("test_evolution.jl")
    include("test_analysis.jl")
    include("test_mean_field.jl")
    include("test_product_density_reference.jl")
    include("test_product_bloch_reference.jl")
    include("test_digital_quantum_magnetism.jl")
    include("test_single_site_mean_field.jl")
    include("test_vectorized_mean_field.jl")
    include("test_channels.jl")
    include("test_transformations.jl")
end
