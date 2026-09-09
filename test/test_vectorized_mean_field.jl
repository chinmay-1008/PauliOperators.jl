using PauliOperators
using Test
using Random

function _random_operator(::Val{N}, nterms::Int; rng=Xoshiro(1)) where {N}
    operator = PauliSum(N, ComplexF64)
    mask = (Int128(1) << N) - 1
    while length(operator) < nterms
        pauli = PauliBasis{N}(
            rand(rng, Int128) & mask,
            rand(rng, Int128) & mask,
        )
        operator[pauli] =
            randn(rng) + randn(rng) * im
    end
    return operator
end

function _test_spv_matches_dict(operator, strategy; atol=1e-12)
    expected = deepcopy(operator)
    truncate!(expected, strategy)

    actual = SparsePauliVector(operator)
    truncate!(actual, strategy)

    @test isapprox(PauliSum(actual), expected; atol=atol)
    @test PauliOperators.check_spv(actual)
    @test actual.an == 0
    return actual
end

@testset "Vectorized mean-field" begin
    @testset "computational-basis reference parity" begin
        N = 8
        rng = Xoshiro(0x51a7)
        operator = _random_operator(Val(N), 80; rng=rng)
        reference = Ket(N, 0b10110101)

        for k in 0:5
            actual = _test_spv_matches_dict(
                operator,
                MeanFieldTruncation(k, reference),
            )
            @test all(weight(pauli) <= k for pauli in keys(actual))
        end
    end

    @testset "product-density reference parity" begin
        N = 7
        operator = _random_operator(Val(N), 60; rng=Xoshiro(0x9dd))
        for probability in (0.0, 0.37, 0.5, 1.0), k in 0:4
            reference = ProductDensityReference(N, probability)
            actual = _test_spv_matches_dict(
                operator,
                MeanFieldTruncation(k, reference),
            )
            @test all(weight(pauli) <= k for pauli in keys(actual))
        end
    end

    @testset "KetSum local-mean parity" begin
        N = 5
        reference = KetSum(N, T=ComplexF64)
        reference[Ket(N, 0b00000)] = 0.3 + 0.1im
        reference[Ket(N, 0b00001)] = -0.2 + 0.4im
        reference[Ket(N, 0b10110)] = 0.5 - 0.3im
        reference[Ket(N, 0b11111)] = -0.1 - 0.2im
        operator = _random_operator(Val(N), 35; rng=Xoshiro(0xb311))

        for k in 0:4
            actual = _test_spv_matches_dict(
                operator,
                MeanFieldTruncation(k, reference);
                atol=2e-12,
            )
            @test all(weight(pauli) <= k for pauli in keys(actual))
        end
    end

    @testset "SMFD parity and normalization" begin
        N = 7
        operator = _random_operator(Val(N), 70; rng=Xoshiro(0x5efd))
        references = (
            Ket(N, 0b1010110),
            ProductDensityReference(N, 0.73),
            ProductBlochReference(N; theta=0.61, phi=0.37),
        )

        for reference in references, normalize in (false, true), k in (2, 4)
            _test_spv_matches_dict(
                operator,
                SingleSiteMeanFieldDecoupling(
                    k,
                    reference;
                    normalize=normalize,
                ),
            )
            _test_spv_matches_dict(
                operator,
                RecursiveSingleSiteMeanFieldDecoupling(
                    k,
                    reference;
                    normalize=normalize,
                ),
            )
        end
    end

    @testset "recursive SMFD closes multi-site weight gaps" begin
        N = 7
        reference = ProductBlochReference(N; theta=0.47, phi=0.31)
        operator = PauliSum{N,Float64}(
            PauliBasis("XYZXYZX") => 0.6,
            PauliBasis("ZZZZZII") => -0.2,
            PauliBasis("XXIIIII") => 0.3,
        )
        strategy = RecursiveSingleSiteMeanFieldDecoupling(
            2,
            reference;
            normalize=true,
        )

        dictionary_result = deepcopy(operator)
        vector_result = SparsePauliVector(operator; T=Float64)
        truncate!(dictionary_result, strategy)
        truncate!(vector_result, strategy)

        @test all(weight(term) <= 2 for term in keys(dictionary_result))
        @test all(
            count_ones(vector_result.z[i] | vector_result.x[i]) <= 2 for
            i in 1:vector_result.n
        )
        @test isapprox(
            dictionary_result,
            PauliSum(vector_result);
            atol=2e-12,
        )
        @test expectation_value(dictionary_result, reference) ≈
              expectation_value(operator, reference) atol=2e-12
        @test expectation_value(vector_result, reference) ≈
              expectation_value(operator, reference) atol=2e-12

        generators = PauliBasis{N}[
            PauliBasis("ZZIIIII"),
            PauliBasis("IZZIIII"),
            PauliBasis("IIZZIII"),
            PauliBasis("IIIZZII"),
            PauliBasis("IIIIZZI"),
        ]
        angles = [0.13, -0.21, 0.17, 0.09, -0.15]
        expected_evolved = deepcopy(operator)
        for (generator, angle) in zip(generators, angles)
            evolve!(expected_evolved, generator, angle)
        end
        truncate!(expected_evolved, strategy)

        actual_evolved = SparsePauliVector(
            operator;
            T=Float64,
            capacity_factor=1,
            append_factor=0.01,
            min_capacity=1,
        )
        counters = PauliOperators.WindowCounters(1)
        evolve!(
            actual_evolved,
            generators,
            angles;
            window=5,
            truncation=strategy,
            counters=counters,
        )
        @test all(
            count_ones(
                actual_evolved.z[i] | actual_evolved.x[i],
            ) <= 2 for i in 1:actual_evolved.n
        )
        @test isapprox(
            PauliSum(actual_evolved),
            expected_evolved;
            atol=2e-12,
        )
    end

    @testset "sequence evolution parity at window one" begin
        N = 6
        operator = PauliSum(N, ComplexF64)
        operator[PauliBasis("XXIIII")] = 0.8 + 0im
        operator[PauliBasis("IZIIII")] = -0.3 + 0im
        generators = PauliBasis{N}[
            PauliBasis("IZZIII"),
            PauliBasis("YIIIII"),
            PauliBasis("IIXXII"),
            PauliBasis("ZIIIIZ"),
        ]
        angles = [0.17, -0.31, 0.23, 0.09]

        ketsum_reference = KetSum(N, T=ComplexF64)
        ketsum_reference[Ket(N, 0b000000)] = inv(sqrt(2)) + 0im
        ketsum_reference[Ket(N, 0b000001)] = inv(sqrt(2)) + 0im

        strategies = (
            MeanFieldTruncation(3, Ket(N, 0b010101)),
            MeanFieldTruncation(
                3,
                ProductDensityReference(N, 0.81),
            ),
            MeanFieldTruncation(3, ketsum_reference),
            SingleSiteMeanFieldDecoupling(
                3,
                Ket(N, 0);
                normalize=true,
            ),
            CompositeTruncation(
                MeanFieldTruncation(3, Ket(N, 0b010101)),
                CoeffTruncation(1e-6),
            ),
        )

        for strategy in strategies
            expected = evolve(
                operator,
                generators,
                angles;
                truncation=strategy,
            )
            actual = SparsePauliVector(operator)
            evolve!(
                actual,
                generators,
                angles;
                window=1,
                truncation=strategy,
            )
            @test isapprox(PauliSum(actual), expected; atol=2e-12)
            @test PauliOperators.check_spv(actual)
        end
    end

    @testset "vectorized evolution fallback parity" begin
        N = 6
        reference = Ket(N, 0b101010)
        strategy = MeanFieldTruncation(2, reference)

        # An initially overweight term disables the bounded append fusion.
        overweight = PauliSum{N,ComplexF64}(
            PauliBasis("ZZZIII") => 0.7 + 0im,
        )
        generators = PauliBasis{N}[PauliBasis("XIIIII")]
        angles = [0.19]
        expected = evolve(
            overweight,
            generators,
            angles;
            truncation=strategy,
        )
        actual = SparsePauliVector(overweight)
        evolve!(actual, generators, angles; truncation=strategy)
        @test isapprox(PauliSum(actual), expected; atol=1e-12)

        # A weight-three generator also takes the exact boundary path.
        bounded = PauliSum{N,ComplexF64}(
            PauliBasis("XXIIII") => 0.4 + 0im,
        )
        generators = PauliBasis{N}[PauliBasis("ZZZIII")]
        expected = evolve(
            bounded,
            generators,
            angles;
            truncation=strategy,
        )
        actual = SparsePauliVector(bounded)
        evolve!(actual, generators, angles; truncation=strategy)
        @test isapprox(PauliSum(actual), expected; atol=1e-12)

        # Corrections disable fusion and retain dictionary measurement
        # semantics.
        generators = PauliBasis{N}[
            PauliBasis("IZZIII"),
            PauliBasis("IIXXII"),
        ]
        angles = [0.11, -0.23]
        dict_correction = EnergyCorrection(reference)
        expected = evolve(
            bounded,
            generators,
            angles;
            truncation=strategy,
            correction=dict_correction,
        )
        spv_correction = EnergyCorrection(reference)
        actual = SparsePauliVector(bounded)
        evolve!(
            actual,
            generators,
            angles;
            truncation=strategy,
            correction=spv_correction,
        )
        @test isapprox(PauliSum(actual), expected; atol=1e-12)
        @test spv_correction.accumulated_energy ≈
              dict_correction.accumulated_energy atol=1e-12

        # window > 1 applies mean field at the same explicit cadence as a
        # hand-written dictionary loop.
        expected = deepcopy(bounded)
        for (i, (generator, angle)) in enumerate(zip(generators, angles))
            evolve!(expected, generator, angle)
            (i % 2 == 0 || i == length(generators)) &&
                truncate!(expected, strategy)
        end
        actual = SparsePauliVector(bounded; capacity_factor=8, append_factor=4)
        evolve!(
            actual,
            generators,
            angles;
            window=2,
            truncation=strategy,
        )
        @test isapprox(PauliSum(actual), expected; atol=1e-12)

        # Even when tight capacity forces intermediate deduplication, the
        # non-compilable mean-field strategy runs only at window boundaries.
        tight = SparsePauliVector(
            bounded;
            capacity_factor=1,
            append_factor=0.01,
            min_capacity=1,
        )
        counters = PauliOperators.WindowCounters(cld(length(generators), 2))
        evolve!(
            tight,
            generators,
            angles;
            window=2,
            truncation=strategy,
            counters=counters,
        )
        @test sum(counters.early_merges) > 0
        @test isapprox(PauliSum(tight), expected; atol=1e-12)
    end

    @testset "correction and composite strategy parity" begin
        N = 5
        reference = Ket(N, 0b10101)
        operator = _random_operator(Val(N), 25; rng=Xoshiro(0xc011))
        strategy = CompositeTruncation(
            MeanFieldTruncation(2, reference),
            CoeffTruncation(1e-4),
        )

        dict_correction = EnergyCorrection(reference)
        expected = deepcopy(operator)
        truncate!(expected, strategy, dict_correction)

        spv_correction = EnergyCorrection(reference)
        actual = SparsePauliVector(operator)
        truncate!(actual, strategy, spv_correction)

        @test isapprox(PauliSum(actual), expected; atol=1e-12)
        @test spv_correction.accumulated_energy ≈
              dict_correction.accumulated_energy atol=1e-12
    end
end
