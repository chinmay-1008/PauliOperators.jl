using PauliOperators
using LinearAlgebra
using Test

function legacy_smfd_evolve(O, generators, angles, truncation)
    out = deepcopy(O)
    for (generator, angle) in zip(generators, angles)
        evolve!(out, generator, angle)
        truncate!(out, truncation)
    end
    return out
end

@testset "Single-site mean-field decoupling (SMFD)" begin
    @testset "supported reference types and strategy construction" begin
        ket = Ket(2, 0)
        density = ProductDensityReference(2, 0.75)
        bloch = ProductBlochReference(2; theta=0.4, phi=0.2)
        ket_sum = KetSum(ket; T=ComplexF64)

        @test SingleSiteMeanFieldDecoupling(1, ket).reference === ket
        @test SingleSiteMeanFieldDecoupling(1, density).reference === density
        @test SingleSiteMeanFieldDecoupling(1, bloch).reference === bloch
        @test !SingleSiteMeanFieldDecoupling(1, ket).normalize
        @test SingleSiteMeanFieldDecoupling(
            1,
            ket;
            normalize=true,
        ).normalize
        @test SingleSiteMeanFieldDecoupling(1, density, true).normalize
        @test RecursiveSingleSiteMeanFieldDecoupling(
            1,
            ket;
            normalize=true,
        ).normalize
        @test RecursiveSingleSiteMeanFieldDecoupling(1, density).reference ===
              density
        @test RecursiveSingleSiteMeanFieldDecoupling(1, bloch).reference ===
              bloch
        @test_throws ArgumentError RecursiveSingleSiteMeanFieldDecoupling(
            -1,
            ket,
        )
        @test_throws MethodError SingleSiteMeanFieldDecoupling(1, ket_sum)
        @test_throws MethodError RecursiveSingleSiteMeanFieldDecoupling(
            1,
            ket_sum,
        )
        @test_throws MethodError single_site_mean_field_decouple(
            PauliBasis("ZZ"),
            1.0 + 0im,
            ket_sum,
        )
    end

    @testset "XZZY formula for both reference types" begin
        pb = PauliBasis("XZZY")
        ket = Ket(4, 0)
        density = ProductDensityReference(4, 0.75) # <Z> = 1/2

        expected_ket = PauliSum(4, ComplexF64)
        expected_ket[PauliBasis("XIZY")] = 1.0 + 0im
        expected_ket[PauliBasis("XZIY")] = 1.0 + 0im

        expected_density = PauliSum(4, ComplexF64)
        expected_density[PauliBasis("XIZY")] = 0.5 + 0im
        expected_density[PauliBasis("XZIY")] = 0.5 + 0im

        normalized_ket = 0.5 * expected_ket
        normalized_density = 0.5 * expected_density

        @test isapprox(
            single_site_mean_field_decouple(pb, 1.0 + 0im, ket),
            expected_ket;
            atol=1e-12,
        )
        @test isapprox(
            single_site_mean_field_decouple(pb, 1.0 + 0im, density),
            expected_density;
            atol=1e-12,
        )
        @test isapprox(
            single_site_mean_field_decouple(
                pb,
                1.0 + 0im,
                ket;
                normalize=true,
            ),
            normalized_ket;
            atol=1e-12,
        )
        @test isapprox(
            single_site_mean_field_decouple(
                pb,
                1.0 + 0im,
                density,
                true,
            ),
            normalized_density;
            atol=1e-12,
        )
    end

    @testset "only nonzero local means contribute" begin
        ket = Ket(4, 0)
        density = ProductDensityReference(4, 0.8)

        for reference in (ket, density)
            @test isempty(
                single_site_mean_field_decouple(
                    PauliBasis("XYYX"),
                    2.0 + 0im,
                    reference,
                ),
            )
        end

        infinite_temperature = ProductDensityReference(4, 0.5)
        @test isempty(
            single_site_mean_field_decouple(
                PauliBasis("ZZZZ"),
                1.0 + 0im,
                infinite_temperature,
            ),
        )
    end

    @testset "pure product-density limits match computational kets" begin
        N = 5
        all_zero = Ket(N, 0)
        all_one = Ket(N, 2^N - 1)
        density_zero = ProductDensityReference(N, 1.0)
        density_one = ProductDensityReference(N, 0.0)

        for pauli_string in ("ZZZZZ", "XZZII", "YIZZZ", "XXYZZ")
            pb = PauliBasis(pauli_string)
            @test isapprox(
                single_site_mean_field_decouple(
                    pb,
                    0.7 + 0.2im,
                    density_zero,
                ),
                single_site_mean_field_decouple(
                    pb,
                    0.7 + 0.2im,
                    all_zero,
                );
                atol=1e-12,
            )
            @test isapprox(
                single_site_mean_field_decouple(
                    pb,
                    0.7 + 0.2im,
                    density_one,
                ),
                single_site_mean_field_decouple(
                    pb,
                    0.7 + 0.2im,
                    all_one,
                );
                atol=1e-12,
            )
        end
    end

    @testset "normalization removes reference-expectation overcounting" begin
        pb = PauliBasis("ZZ")
        ket = Ket(2, 0)
        density = ProductDensityReference(2, 0.75)

        original = PauliSum{2,ComplexF64}(pb => 1.0 + 0im)
        ket_smfd = single_site_mean_field_decouple(pb, 1.0 + 0im, ket)
        density_smfd =
            single_site_mean_field_decouple(pb, 1.0 + 0im, density)
        ket_normalized = single_site_mean_field_decouple(
            pb,
            1.0 + 0im,
            ket;
            normalize=true,
        )
        density_normalized = single_site_mean_field_decouple(
            pb,
            1.0 + 0im,
            density;
            normalize=true,
        )

        @test expectation_value(ket_smfd, ket) ≈
              2 * expectation_value(original, ket)
        @test expectation_value(density_smfd, density) ≈
              2 * expectation_value(original, density)
        @test expectation_value(ket_normalized, ket) ≈
              expectation_value(original, ket)
        @test expectation_value(density_normalized, density) ≈
              expectation_value(original, density)
        @test norm(ket_normalized) < norm(original)
        @test norm(density_normalized) < norm(original)
    end

    @testset "in-place strategy applies exactly one decoupling step" begin
        reference = Ket(4, 0)
        original = PauliSum(4, ComplexF64)
        original[PauliBasis("ZZZZ")] = 1.0 + 0im
        original[PauliBasis("IZZZ")] = 2.0 + 0im

        expected = PauliSum(4, ComplexF64)
        for (pb, c) in original
            sum!(
                expected,
                single_site_mean_field_decouple(pb, c, reference),
            )
        end

        actual = deepcopy(original)
        truncate!(actual, SingleSiteMeanFieldDecoupling(2, reference))

        @test isapprox(actual, expected; atol=1e-12)
        @test haskey(actual, PauliBasis("IZZZ"))
        @test maximum(weight(pb) for pb in keys(actual)) == 3
    end

    @testset "recursive strategy reaches the strict cutoff" begin
        N = 6
        max_weight = 2
        pauli = PauliBasis("ZZZZZZ")
        ket = Ket(N, 0)
        original = PauliSum{N,Float64}(pauli => 1.0)

        one_pass = deepcopy(original)
        truncate!(
            one_pass,
            SingleSiteMeanFieldDecoupling(
                max_weight,
                ket;
                normalize=true,
            ),
        )
        @test maximum(weight, keys(one_pass)) == 5

        recursive = deepcopy(original)
        truncate!(
            recursive,
            RecursiveSingleSiteMeanFieldDecoupling(
                max_weight,
                ket;
                normalize=true,
            ),
        )
        @test length(recursive) == binomial(N, max_weight)
        @test all(weight(term) == max_weight for term in keys(recursive))
        @test all(
            coefficient ≈ inv(binomial(N, max_weight)) for
            coefficient in values(recursive)
        )
        @test expectation_value(recursive, ket) ≈
              expectation_value(original, ket) atol=1e-12

        # At k+1 the recursive strategy is exactly the old one-pass rule;
        # only larger weight gaps activate additional passes.
        adjacent = PauliBasis("ZZZIII")
        old_adjacent = single_site_mean_field_decouple(
            adjacent,
            0.7,
            ket;
            normalize=true,
        )
        new_adjacent = recursive_single_site_mean_field_decouple(
            adjacent,
            0.7,
            ket,
            max_weight;
            normalize=true,
        )
        @test isapprox(old_adjacent, new_adjacent; atol=1e-12)

        density = ProductDensityReference(N, 0.8)
        density_result = recursive_single_site_mean_field_decouple(
            pauli,
            1.0,
            density,
            max_weight;
            normalize=true,
        )
        magnetization = density.magnetization
        expected_coefficient =
            magnetization^(N - max_weight) / binomial(N, max_weight)
        @test all(
            coefficient ≈ expected_coefficient for
            coefficient in values(density_result)
        )
        @test expectation_value(density_result, density) ≈
              expectation_value(original, density) atol=1e-12

        bloch = ProductBlochReference(N; theta=0.51, phi=0.28)
        mixed_pauli = PauliBasis("XYZXYZ")
        mixed_original = PauliSum{N,Float64}(mixed_pauli => 0.4)
        bloch_result = recursive_single_site_mean_field_decouple(
            mixed_pauli,
            0.4,
            bloch,
            max_weight;
            normalize=true,
        )
        @test all(weight(term) <= max_weight for term in keys(bloch_result))
        @test expectation_value(bloch_result, bloch) ≈
              expectation_value(mixed_original, bloch) atol=2e-12

        no_local_means = recursive_single_site_mean_field_decouple(
            PauliBasis("XXXXXX"),
            1.0,
            density,
            max_weight;
            normalize=true,
        )
        @test isempty(no_local_means)
    end

    @testset "terms within the threshold remain unchanged" begin
        reference = ProductDensityReference(4, 0.7)
        O = PauliSum(4, ComplexF64)
        O[PauliBasis("XXII")] = 0.25 + 0im
        O[PauliBasis("ZZZI")] = 0.5 + 0im

        truncate!(O, SingleSiteMeanFieldDecoupling(2, reference))

        @test O[PauliBasis("XXII")] == 0.25 + 0im
        @test all(weight(pb) <= 2 for pb in keys(O))
    end

    @testset "normalized in-place strategy averages each replacement" begin
        reference = Ket(4, 0)
        original = PauliSum{4,ComplexF64}(
            PauliBasis("ZZZZ") => 1.0 + 0im,
        )
        expected = single_site_mean_field_decouple(
            PauliBasis("ZZZZ"),
            1.0 + 0im,
            reference;
            normalize=true,
        )

        actual = deepcopy(original)
        truncate!(
            actual,
            SingleSiteMeanFieldDecoupling(
                3,
                reference;
                normalize=true,
            ),
        )

        @test isapprox(actual, expected; atol=1e-12)
        @test expectation_value(actual, reference) ≈
              expectation_value(original, reference)
        @test norm(actual) ≈ 0.5 atol=1e-12
    end

    @testset "fused evolution matches apply-after-evolution path" begin
        N = 4
        O = PauliSum(Pauli("XXII"))
        generators = [PauliBasis("IZZI")]
        angles = [0.37]

        for reference in (Ket(N, 0), ProductDensityReference(N, 0.8))
            for normalize in (false, true)
                strategy = SingleSiteMeanFieldDecoupling(
                    2,
                    reference;
                    normalize=normalize,
                )
                fast = evolve(O, generators, angles; truncation=strategy)
                legacy = legacy_smfd_evolve(O, generators, angles, strategy)

                @test isapprox(fast, legacy; atol=1e-12)
                @test all(weight(pb) <= 2 for pb in keys(fast))
            end
        end
    end
end
