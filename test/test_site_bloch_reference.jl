using PauliOperators
using Random
using Test

# Random operator on N qubits with every string of weight 2..N_max_weight
# drawn uniformly, real coefficients.
function _random_pauli_sum(rng, N, n_terms, max_string_weight)
    O = PauliSum(N, Float64)
    letters = ('X', 'Y', 'Z')
    while length(O) < n_terms
        sites = randperm(rng, N)[1:rand(rng, 2:max_string_weight)]
        chars = fill('I', N)
        for q in sites
            chars[q] = rand(rng, letters)
        end
        O[PauliBasis(String(chars))] = randn(rng)
    end
    return O
end

function _random_site_reference(rng, N)
    moments = zeros(N, 3)
    for q in 1:N
        direction = randn(rng, 3)
        moments[q, :] .= rand(rng) .* direction ./ sqrt(sum(abs2, direction))
    end
    return SiteBlochReference(moments)
end

_as_pauli_sum(v::SparsePauliVector) = PauliSum(v)

function _max_coefficient_gap(a::PauliSum, b::PauliSum)
    gap = 0.0
    for key in union(keys(a), keys(b))
        gap = max(gap, abs(get(a, key, 0.0) - get(b, key, 0.0)))
    end
    return gap
end

@testset "SiteBlochReference" begin
    rng = MersenneTwister(20261002)
    N = 7

    @testset "construction and validation" begin
        reference = SiteBlochReference([0.1, 0.2], [0.0, 0.3], [0.5, -0.4])
        @test reference isa SiteBlochReference{2,Float64}
        @test reference.y[2] == 0.3
        @test_throws ArgumentError SiteBlochReference([1.0], [0.5], [0.0])
        @test_throws DimensionMismatch SiteBlochReference([0.1, 0.2], [0.0], [0.5, 0.1])
        @test_throws DimensionMismatch SiteBlochReference(zeros(3, 2))

        uniform = ProductBlochReference(N, 0.3, -0.2, 0.6)
        converted = SiteBlochReference(uniform)
        @test converted isa SiteBlochReference{N}
        @test all(converted.x .== 0.3) && all(converted.y .== -0.2) && all(converted.z .== 0.6)
    end

    @testset "expectation value is the product of per-site means" begin
        reference = _random_site_reference(rng, N)
        pb = PauliBasis("XIYZIIX")
        expected = reference.x[1] * reference.y[3] * reference.z[4] * reference.x[7]
        @test expectation_value(pb, reference) ≈ expected
        @test expectation_value(PauliBasis("IIIIIII"), reference) == 1.0
    end

    @testset "uniform per-site reference reproduces ProductBlochReference" begin
        uniform = ProductBlochReference(N, 0.35, 0.15, -0.55)
        site = SiteBlochReference(uniform)
        O = _random_pauli_sum(rng, N, 120, 5)

        @test expectation_value(O, site) ≈ expectation_value(O, uniform) atol = 1e-13

        for normalize in (false, true)
            a = single_site_mean_field_decouple!(deepcopy(O), uniform, 3; normalize = normalize)
            b = single_site_mean_field_decouple!(deepcopy(O), site, 3; normalize = normalize)
            @test _max_coefficient_gap(a, b) < 1e-14

            va = single_site_mean_field_decouple!(SparsePauliVector(O), uniform, 3; normalize = normalize)
            vb = single_site_mean_field_decouple!(SparsePauliVector(O), site, 3; normalize = normalize)
            @test _max_coefficient_gap(_as_pauli_sum(va), _as_pauli_sum(vb)) < 1e-14
        end
    end

    @testset "non-uniform: PauliSum and packed paths agree" begin
        site = _random_site_reference(rng, N)
        O = _random_pauli_sum(rng, N, 150, 6)
        for normalize in (false, true)
            a = single_site_mean_field_decouple!(deepcopy(O), site, 3; normalize = normalize)
            v = single_site_mean_field_decouple!(SparsePauliVector(O), site, 3; normalize = normalize)
            @test _max_coefficient_gap(a, _as_pauli_sum(v)) < 1e-13
        end
    end

    @testset "non-uniform: factors use their own site's mean" begin
        site = SiteBlochReference(
            [0.1, 0.2, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.7],
        )
        result = single_site_mean_field_decouple(PauliBasis("XXXZ"), 1.0, site)
        @test result[PauliBasis("IXXZ")] ≈ 0.1
        @test result[PauliBasis("XIXZ")] ≈ 0.2
        @test result[PauliBasis("XXIZ")] ≈ 0.3
        @test result[PauliBasis("XXXI")] ≈ 0.7
        @test length(result) == 4
    end

    @testset "normalized rule preserves the per-site reference expectation" begin
        site = _random_site_reference(rng, N)
        O = _random_pauli_sum(rng, N, 150, 6)
        decoupled = single_site_mean_field_decouple!(deepcopy(O), site, 3; normalize = true)
        # One SMFD pass keeps every term's reference expectation; strings of
        # weight 6 stay above the cutoff but still contribute the same mean.
        @test expectation_value(decoupled, site) ≈ expectation_value(O, site) atol = 1e-12

        recursive = recursive_single_site_mean_field_decouple!(SparsePauliVector(O), site, 2; normalize = true)
        @test all(weight(pb) <= 2 for pb in keys(_as_pauli_sum(recursive)))
        @test expectation_value(recursive, site) ≈ expectation_value(O, site) atol = 1e-12
    end

    @testset "truncation strategy inside evolve!" begin
        site = _random_site_reference(rng, N)
        uniform = ProductBlochReference(N, 0.2, 0.1, 0.5)
        # Two-site generators on a weight-3 start: one SMFD pass per gate is
        # then enough to hold the weight-3 bound.
        generators = [PauliBasis("XXIIIII"), PauliBasis("IZZIIII"), PauliBasis("IIYXIII"), PauliBasis("IIIZZII")]
        angles = [0.3, -0.7, 0.4, 0.9]
        start = SparsePauliVector(PauliSum(Pauli("ZZIZIII")); T = Float64)

        strategy = SingleSiteMeanFieldDecoupling(3, site; normalize = true)
        @test strategy isa SingleSiteMeanFieldDecoupling{N}
        operator = deepcopy(start)
        evolve!(operator, generators, angles; window = 1, truncation = strategy)
        @test all(weight(pb) <= 3 for pb in keys(_as_pauli_sum(operator)))

        a = deepcopy(start)
        b = deepcopy(start)
        evolve!(a, generators, angles; window = 1,
                truncation = SingleSiteMeanFieldDecoupling(3, uniform; normalize = true))
        evolve!(b, generators, angles; window = 1,
                truncation = SingleSiteMeanFieldDecoupling(3, SiteBlochReference(uniform); normalize = true))
        @test _max_coefficient_gap(_as_pauli_sum(a), _as_pauli_sum(b)) < 1e-13
    end

    @testset "RCFT (mean_field_factorize!) with a per-site reference" begin
        uniform = ProductBlochReference(N, 0.35, 0.15, -0.55)
        O = _random_pauli_sum(rng, N, 120, 6)

        # Uniform per-site reference reproduces the uniform one, on both paths.
        a = mean_field_factorize!(deepcopy(O), uniform, 3)
        b = mean_field_factorize!(deepcopy(O), SiteBlochReference(uniform), 3)
        @test _max_coefficient_gap(a, b) < 1e-13
        va = mean_field_factorize!(SparsePauliVector(O), uniform, 3)
        vb = mean_field_factorize!(SparsePauliVector(O), SiteBlochReference(uniform), 3)
        @test _max_coefficient_gap(_as_pauli_sum(va), _as_pauli_sum(vb)) < 1e-13

        # Non-uniform: PauliSum and packed paths agree, the weight bound holds, and
        # the reference expectation is preserved (every dropped piece has at least
        # k + 1 zero-mean fluctuation factors).
        site = _random_site_reference(rng, N)
        p = mean_field_factorize!(deepcopy(O), site, 3)
        v = mean_field_factorize!(SparsePauliVector(O), site, 3)
        @test _max_coefficient_gap(p, _as_pauli_sum(v)) < 1e-12
        @test all(weight(pb) <= 3 for pb in keys(p))
        @test expectation_value(p, site) ≈ expectation_value(O, site) atol = 1e-12

        # Each factor uses its own site's mean: XZ on sites 1, 2 at k = 0 keeps only
        # the identity with coefficient <X_1><Z_2>.
        two_site = SiteBlochReference([0.4, 0.0], [0.0, 0.0], [0.0, -0.5])
        single = mean_field_factorize(PauliBasis("XZ"), 1.0, two_site, 0)
        @test single[PauliBasis("II")] ≈ 0.4 * -0.5
        @test length(single) == 1

        # Strategy inside evolve!.
        generators = [PauliBasis("XXIIIII"), PauliBasis("IZZIIII"), PauliBasis("IIYXIII"), PauliBasis("IIIZZII")]
        angles = [0.3, -0.7, 0.4, 0.9]
        start = SparsePauliVector(PauliSum(Pauli("ZZIZIII")); T = Float64)
        strategy = MeanFieldTruncation(3, site)
        @test strategy isa MeanFieldTruncation{N}
        operator = deepcopy(start)
        evolve!(operator, generators, angles; window = 1, truncation = strategy)
        @test all(weight(pb) <= 3 for pb in keys(_as_pauli_sum(operator)))

        x = deepcopy(start)
        y = deepcopy(start)
        evolve!(x, generators, angles; window = 1, truncation = MeanFieldTruncation(3, uniform))
        evolve!(y, generators, angles; window = 1, truncation = MeanFieldTruncation(3, SiteBlochReference(uniform)))
        @test _max_coefficient_gap(_as_pauli_sum(x), _as_pauli_sum(y)) < 1e-13
    end
end
