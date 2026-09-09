using PauliOperators
using LinearAlgebra
using Random
using Test

function product_bloch_ketsum(N::Int, theta::Real, phi::Real)
    state = KetSum(N, T=ComplexF64)
    c = cos(theta / 2)
    excited = cis(phi) * sin(theta / 2)
    for bits in 0:(Int(1) << N) - 1
        n_excited = count_ones(bits)
        state[Ket(N, bits)] = c^(N - n_excited) * excited^n_excited
    end
    return state
end

@testset "Product-Bloch mean-field reference" begin
    @testset "construction and local expectations" begin
        @test_throws ArgumentError ProductBlochReference(0, 0.0, 0.0, 1.0)
        @test_throws ArgumentError ProductBlochReference(129, 0.0, 0.0, 1.0)
        @test_throws ArgumentError ProductBlochReference(3, Inf, 0.0, 0.0)
        @test_throws ArgumentError ProductBlochReference(3, 0.8, 0.8, 0.0)
        @test_throws ArgumentError ProductBlochReference(
            3,
            1.0 + 1e-10,
            0.0,
            0.0,
        )

        theta = 0.37
        xz_reference = ProductBlochReference(3; theta=theta)
        @test xz_reference.x ≈ sin(theta)
        @test xz_reference.y ≈ 0.0 atol=1e-15
        @test xz_reference.z ≈ cos(theta)
        @test expectation_value(PauliBasis("XII"), xz_reference) ≈ sin(theta)
        @test expectation_value(PauliBasis("YII"), xz_reference) ≈ 0.0 atol=1e-15
        @test expectation_value(PauliBasis("ZII"), xz_reference) ≈ cos(theta)

        yz_reference = ProductBlochReference(3; theta=theta, phi=pi / 2)
        @test yz_reference.x ≈ 0.0 atol=1e-15
        @test yz_reference.y ≈ sin(theta)
        @test yz_reference.z ≈ cos(theta)

        mixed = ProductBlochReference(3, 0.2, -0.3, 0.4)
        @test expectation_value(PauliBasis("XYZ"), mixed) ≈
              mixed.x * mixed.y * mixed.z
        @test sprint(show, mixed) ==
              "ProductBlochReference(3, 0.2, -0.3, 0.4)"
    end

    @testset "explicit tilted-state parity" begin
        N = 4
        theta = 0.43
        for phi in (0.0, pi / 4, pi / 2)
            reference = ProductBlochReference(N; theta=theta, phi=phi)
            state = product_bloch_ketsum(N, theta, phi)
            for pauli_string in (
                "IIII",
                "XIII",
                "YIII",
                "ZIII",
                "XYZI",
                "XXYY",
                "ZZZZ",
            )
                pauli = PauliBasis(pauli_string)
                @test expectation_value(pauli, reference) ≈
                      expectation_value(PauliSum(pauli), state) atol=1e-12
            end
        end

        rng = MersenneTwister(0x71A7ED)
        alphabet = ('I', 'X', 'Y', 'Z')
        for _ in 1:30
            theta_random = pi * rand(rng)
            phi_random = 2pi * rand(rng)
            reference = ProductBlochReference(
                N;
                theta=theta_random,
                phi=phi_random,
            )
            state = product_bloch_ketsum(N, theta_random, phi_random)
            pauli = PauliBasis(String(rand(rng, alphabet, N)))
            @test expectation_value(pauli, reference) ≈
                  expectation_value(PauliSum(pauli), state) atol=2e-12
        end
    end

    @testset "mixed Bloch vector parity" begin
        N = 3
        reference = ProductBlochReference(N, 0.21, -0.34, 0.47)
        local_density = ComplexF64[
            1 + reference.z reference.x - im * reference.y
            reference.x + im * reference.y 1 - reference.z
        ] / 2
        density = foldl(kron, fill(local_density, N))
        rng = MersenneTwister(0xD3517)
        alphabet = ('I', 'X', 'Y', 'Z')
        for _ in 1:30
            pauli = PauliBasis(String(rand(rng, alphabet, N)))
            @test expectation_value(pauli, reference) ≈
                  tr(Matrix(pauli) * density) atol=2e-12
        end
    end

    @testset "mean-field expectation preservation" begin
        N = 6
        reference = ProductBlochReference(N; theta=0.51, phi=0.28)
        rng = MersenneTwister(0xB10C)
        alphabet = ('I', 'X', 'Y', 'Z')

        for _ in 1:40
            pauli_string = String(rand(rng, alphabet, N))
            pauli = PauliBasis(pauli_string)
            coefficient = randn(rng)
            for max_weight in 0:weight(pauli)
                projected = mean_field_factorize(
                    pauli,
                    coefficient,
                    reference,
                    max_weight,
                )
                @test expectation_value(projected, reference) ≈
                      coefficient * expectation_value(pauli, reference) atol=2e-11
                @test all(weight(term) <= max_weight for term in keys(projected))
            end
        end
    end

    @testset "normalized single-site expectation preservation" begin
        N = 6
        reference = ProductBlochReference(N; theta=0.51, phi=0.28)
        rng = MersenneTwister(0x51A61E)
        alphabet = ('I', 'X', 'Y', 'Z')

        for _ in 1:40
            pauli = PauliBasis(String(rand(rng, alphabet, N)))
            iszero(weight(pauli)) && continue
            coefficient = randn(rng)
            projected = single_site_mean_field_decouple(
                pauli,
                coefficient,
                reference;
                normalize=true,
            )
            @test expectation_value(projected, reference) ≈
                  coefficient * expectation_value(pauli, reference) atol=2e-12
            @test all(
                weight(term) == weight(pauli) - 1 for term in keys(projected)
            )
        end
    end

    @testset "dictionary and SPV parity" begin
        N = 6
        reference = ProductBlochReference(N; theta=pi / 18)
        operator = PauliSum(N, Float64)
        operator[PauliBasis("ZZZZZZ")] = 0.7
        operator[PauliBasis("XYZXYZ")] = -0.2
        operator[PauliBasis("XXZZII")] = 0.5
        operator[PauliBasis("YIIIII")] = 0.3

        dictionary_result = deepcopy(operator)
        vector_result = SparsePauliVector(operator; T=Float64)
        strategy = MeanFieldTruncation(3, reference)
        truncate!(dictionary_result, strategy)
        truncate!(vector_result, strategy)

        @test isapprox(
            dictionary_result,
            PauliSum(vector_result);
            atol=1e-12,
        )
        @test expectation_value(dictionary_result, reference) ≈
              expectation_value(operator, reference) atol=1e-12
        @test expectation_value(vector_result, reference) ≈
              expectation_value(operator, reference) atol=1e-12

        smfd_dictionary = deepcopy(operator)
        smfd_vector = SparsePauliVector(operator; T=Float64)
        smfd_strategy = SingleSiteMeanFieldDecoupling(
            3,
            reference;
            normalize=true,
        )
        truncate!(smfd_dictionary, smfd_strategy)
        truncate!(smfd_vector, smfd_strategy)
        @test isapprox(
            smfd_dictionary,
            PauliSum(smfd_vector);
            atol=1e-12,
        )
        @test expectation_value(smfd_dictionary, reference) ≈
              expectation_value(operator, reference) atol=1e-12
        @test expectation_value(smfd_vector, reference) ≈
              expectation_value(operator, reference) atol=1e-12
    end
end
