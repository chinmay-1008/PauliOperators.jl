using PauliOperators
using Test

function legacy_evolve_with_truncation(O, generators, angles, truncation;
                                       correction=NoCorrection())
    out = deepcopy(O)
    for (generator, angle) in zip(generators, angles)
        evolve!(out, generator, angle)
        truncate!(out, truncation, correction)
    end
    return out
end

@testset "Product-density mean-field reference" begin
    @testset "construction and expectations" begin
        @test_throws ArgumentError ProductDensityReference(0, 0.5)
        @test_throws ArgumentError ProductDensityReference(129, 0.5)
        @test_throws ArgumentError ProductDensityReference(4, -0.01)
        @test_throws ArgumentError ProductDensityReference(4, 1.01)
        @test_throws ArgumentError ProductDensityReference(4, Inf)

        reference = ProductDensityReference(4, 0.75)
        @test reference.p == 0.75
        @test reference.magnetization == 0.5
        @test sprint(show, reference) == "ProductDensityReference(4, 0.75)"

        @test expectation_value(PauliBasis("IIII"), reference) == 1.0
        @test expectation_value(PauliBasis("ZIII"), reference) == 0.5
        @test expectation_value(PauliBasis("ZZII"), reference) == 0.25
        @test expectation_value(PauliBasis("XIII"), reference) == 0.0
        @test expectation_value(PauliBasis("IYZI"), reference) == 0.0

        O = PauliSum(4, ComplexF64)
        O[PauliBasis("IIII")] = 0.25 + 0im
        O[PauliBasis("ZIII")] = 2.0 + 0im
        O[PauliBasis("ZZII")] = -1.0 + 0im
        O[PauliBasis("XIII")] = 7.0 + 0im
        @test expectation_value(O, reference) ≈ 1.0 + 0im
    end

    @testset "known factorizations" begin
        reference = ProductDensityReference(4, 0.75) # m = 1/2

        zz = mean_field_factorize(PauliBasis("ZZII"), 1.0 + 0im,
                                  reference, 1)
        expected_zz = PauliSum(4, ComplexF64)
        expected_zz[PauliBasis("ZIII")] = 0.5 + 0im
        expected_zz[PauliBasis("IZII")] = 0.5 + 0im
        expected_zz[PauliBasis("IIII")] = -0.25 + 0im
        @test isapprox(zz, expected_zz; atol=1e-12)

        xzz = mean_field_factorize(PauliBasis("XZZI"), 1.0 + 0im,
                                   reference, 2)
        expected_xzz = PauliSum(4, ComplexF64)
        expected_xzz[PauliBasis("XZII")] = 0.5 + 0im
        expected_xzz[PauliBasis("XIZI")] = 0.5 + 0im
        expected_xzz[PauliBasis("XIII")] = -0.25 + 0im
        @test isapprox(xzz, expected_xzz; atol=1e-12)

        xxyz = mean_field_factorize(PauliBasis("XXYZ"), 1.0 + 0im,
                                    reference, 3)
        expected_xxyz = PauliSum(4, ComplexF64)
        expected_xxyz[PauliBasis("XXYI")] = 0.5 + 0im
        @test isapprox(xxyz, expected_xxyz; atol=1e-12)

        unchanged = mean_field_factorize(PauliBasis("XXYZ"), 2.0 + 0im,
                                         reference, 4)
        @test unchanged == PauliSum{4,ComplexF64}(
            PauliBasis("XXYZ") => 2.0 + 0im,
        )
    end

    @testset "pure and infinite-temperature limits" begin
        N = 5
        all_zero = Ket(N, Int128(0))
        all_one = Ket(N, Int128(2^N - 1))
        reference_zero = ProductDensityReference(N, 1.0)
        reference_one = ProductDensityReference(N, 0.0)

        for pauli_string in ("ZZZZZ", "XZZII", "YIZZZ", "XXYZZ")
            pb = PauliBasis(pauli_string)
            for k in 0:weight(pb)
                @test isapprox(
                    mean_field_factorize(pb, 0.7 + 0.2im, reference_zero, k),
                    mean_field_factorize(pb, 0.7 + 0.2im, all_zero, k);
                    atol=1e-12,
                )
                @test isapprox(
                    mean_field_factorize(pb, 0.7 + 0.2im, reference_one, k),
                    mean_field_factorize(pb, 0.7 + 0.2im, all_one, k);
                    atol=1e-12,
                )
            end
        end

        O = PauliSum(N, ComplexF64)
        O[PauliBasis("IIIII")] = 0.2 + 0im
        O[PauliBasis("ZIIII")] = 0.3 + 0im
        O[PauliBasis("ZZIII")] = 0.4 + 0im
        O[PauliBasis("XZZZI")] = 0.5 + 0im
        O[PauliBasis("ZZZZZ")] = 0.6 + 0im

        mf = deepcopy(O)
        wt = deepcopy(O)
        truncate!(mf, MeanFieldTruncation(2,
                                          ProductDensityReference(N, 0.5)))
        truncate!(wt, WeightTruncation(2))
        @test mf == wt
    end

    @testset "strategy bounds output weight" begin
        N = 6
        reference = ProductDensityReference(N, 0.63)
        O = PauliSum(N, ComplexF64)
        O[PauliBasis("ZZZZZZ")] = 1.0 + 0im
        O[PauliBasis("XZZZZI")] = 0.5 + 0im
        O[PauliBasis("XXYZII")] = 0.25 + 0im
        O[PauliBasis("XYIIII")] = 0.75 + 0im
        truncate!(O, MeanFieldTruncation(3, reference))
        @test all(weight(pb) <= 3 for pb in keys(O))
        @test O[PauliBasis("XYIIII")] == 0.75 + 0im
    end

    @testset "fused evolution matches legacy path" begin
        N = 4
        reference = ProductDensityReference(N, 0.8)
        angle = 0.37

        # Growth: the sine branch of XXII under IZZI has weight three.
        growth_O = PauliSum(Pauli("XXII"))
        growth_G = PauliBasis("IZZI")
        mf = MeanFieldTruncation(2, reference)
        fast = evolve(growth_O, [growth_G], [angle]; truncation=mf)
        legacy = legacy_evolve_with_truncation(growth_O, [growth_G],
                                               [angle], mf)
        @test isapprox(fast, legacy; atol=1e-12)
        @test all(weight(pb) <= 2 for pb in keys(fast))

        # Shrink: an anticommuting two-site product can reduce weight by one.
        shrink_O = PauliSum(Pauli("XXII"))
        shrink_G = PauliBasis("XYII")
        fast = evolve(shrink_O, [shrink_G], [angle]; truncation=mf)
        legacy = legacy_evolve_with_truncation(shrink_O, [shrink_G],
                                               [angle], mf)
        @test isapprox(fast, legacy; atol=1e-12)

        # Collision between an existing cosine key and its paired sine key.
        collision_O = PauliSum(4, ComplexF64)
        collision_O[PauliBasis("XIII")] = 0.8 + 0im
        collision_O[PauliBasis("YIZI")] = -0.3 + 0im
        collision_G = PauliBasis("ZIZI")
        fast = evolve(collision_O, [collision_G], [angle]; truncation=mf)
        legacy = legacy_evolve_with_truncation(collision_O, [collision_G],
                                               [angle], mf)
        @test isapprox(fast, legacy; atol=1e-12)

        weight_strategy = WeightTruncation(2)
        fast = evolve(growth_O, [growth_G], [angle];
                      truncation=weight_strategy)
        legacy = legacy_evolve_with_truncation(growth_O, [growth_G],
                                               [angle], weight_strategy)
        @test isapprox(fast, legacy; atol=1e-12)

        composite = CompositeTruncation(mf, CoeffTruncation(0.1))
        fast = evolve(growth_O, [growth_G], [angle]; truncation=composite)
        legacy = legacy_evolve_with_truncation(growth_O, [growth_G],
                                               [angle], composite)
        @test isapprox(fast, legacy; atol=1e-12)
    end

    @testset "fused evolution fallback conditions" begin
        N = 4
        reference = ProductDensityReference(N, 0.7)
        mf = MeanFieldTruncation(2, reference)
        angle = 0.21

        # Generator weight above two.
        O = PauliSum(Pauli("XIII"))
        generators = [PauliBasis("ZZZI")]
        fast = evolve(O, generators, [angle]; truncation=mf)
        legacy = legacy_evolve_with_truncation(O, generators, [angle], mf)
        @test isapprox(fast, legacy; atol=1e-12)

        # Input is not initially bounded.
        overweight = PauliSum(Pauli("ZZZI"))
        generators = [PauliBasis("XIII")]
        fast = evolve(overweight, generators, [angle]; truncation=mf)
        legacy = legacy_evolve_with_truncation(overweight, generators,
                                               [angle], mf)
        @test isapprox(fast, legacy; atol=1e-12)

        # A coefficient strategy before the limiter cannot be reordered.
        ordered = CompositeTruncation(CoeffTruncation(0.1), mf)
        fast = evolve(O, generators, [angle]; truncation=ordered)
        legacy = legacy_evolve_with_truncation(O, generators, [angle], ordered)
        @test isapprox(fast, legacy; atol=1e-12)

        # Correction accumulators retain the legacy measurement semantics.
        ket = Ket(N, 0)
        correction_fast = EnergyCorrection(ket)
        correction_legacy = EnergyCorrection(ket)
        fast = evolve(O, generators, [angle]; truncation=mf,
                      correction=correction_fast)
        legacy = legacy_evolve_with_truncation(
            O,
            generators,
            [angle],
            mf;
            correction=correction_legacy,
        )
        @test isapprox(fast, legacy; atol=1e-12)
        @test correction_fast.accumulated_energy ≈
              correction_legacy.accumulated_energy atol=1e-12
    end
end
