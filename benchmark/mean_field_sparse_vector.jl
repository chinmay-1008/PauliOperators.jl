using PauliOperators
using BenchmarkTools
using Random
using Printf

const DEFAULT_SCENARIOS = [
    (N=16, nterms=1_000, k=2, rotations=30),
    (N=16, nterms=5_000, k=3, rotations=100),
    (N=20, nterms=1_000, k=4, rotations=30),
    (N=20, nterms=5_000, k=3, rotations=100),
]

function random_operator(N::Int, nterms::Int, max_weight::Int, rng)
    operator = PauliSum(N, ComplexF64)
    positions = collect(1:N)
    while length(operator) < nterms
        shuffle!(rng, positions)
        term_weight = rand(rng, 1:max_weight)
        z = Int128(0)
        x = Int128(0)
        for q in @view positions[1:term_weight]
            bit = Int128(1) << (q - 1)
            axis = rand(rng, 1:3)
            axis != 1 && (z |= bit)
            axis != 2 && (x |= bit)
        end
        operator[PauliBasis{N}(z, x)] = randn(rng) + 0im
    end
    return operator
end

function random_generators(N::Int, count::Int, rng)
    generators = Vector{PauliBasis{N}}(undef, count)
    for i in eachindex(generators)
        q1 = rand(rng, 1:N)
        q2 = rand(rng, 1:(N - 1))
        q2 >= q1 && (q2 += 1)
        z = Int128(0)
        x = Int128(0)
        for q in (q1, q2)
            bit = Int128(1) << (q - 1)
            axis = rand(rng, 1:3)
            axis != 1 && (z |= bit)
            axis != 2 && (x |= bit)
        end
        generators[i] = PauliBasis{N}(z, x)
    end
    return generators
end

function evolve_dict!(operator, generators, angles, strategy)
    for (generator, angle) in zip(generators, angles)
        evolve!(operator, generator, angle)
        truncate!(operator, strategy)
    end
    return operator
end

function max_coefficient_difference(a::PauliSum{N}, b::PauliSum{N}) where {N}
    all_keys = union(keys(a), keys(b))
    isempty(all_keys) && return 0.0
    return maximum(
        abs(get(a, key, 0.0 + 0im) - get(b, key, 0.0 + 0im))
        for key in all_keys
    )
end

function measured(benchmarkable, samples::Int)
    benchmarkable.params.samples = samples
    benchmarkable.params.evals = 1
    benchmarkable.params.seconds = 120.0
    return median(run(benchmarkable))
end

function strategy_cases(N::Int, k::Int)
    reference = ProductDensityReference(N, 0.78)
    return (
        (name="mean_field", strategy=MeanFieldTruncation(k, reference)),
        (
            name="smfd_normalized",
            strategy=SingleSiteMeanFieldDecoupling(
                k,
                reference;
                normalize=true,
            ),
        ),
    )
end

function benchmark_case(scenario, strategy_case; samples=5)
    (; N, nterms, k, rotations) = scenario
    (; name, strategy) = strategy_case
    strategy_seed = name == "mean_field" ? UInt64(0x4d46) : UInt64(0x534d4644)
    seed = UInt64(N) ⊻
           (UInt64(nterms) << 8) ⊻
           (UInt64(k) << 32) ⊻
           (UInt64(rotations) << 40) ⊻
           strategy_seed
    rng = Xoshiro(seed)
    operator = random_operator(N, nterms, k, rng)
    overweight = random_operator(N, nterms, min(k + 1, N), rng)
    generators = random_generators(N, rotations, rng)
    angles = fill(0.07, rotations)
    vectorized = SparsePauliVector(
        operator;
        T=Float64,
        capacity_factor=8,
        append_factor=4,
    )
    vectorized_overweight = SparsePauliVector(
        overweight;
        T=Float64,
        capacity_factor=8,
        append_factor=4,
    )

    dict_direct = deepcopy(overweight)
    truncate!(dict_direct, strategy)
    spv_direct = copy(vectorized_overweight)
    truncate!(spv_direct, strategy)
    direct_diff = max_coefficient_difference(dict_direct, PauliSum(spv_direct))
    direct_diff <= 1e-12 || error("direct parity failure: $direct_diff")
    PauliOperators.check_spv(spv_direct)

    dict_result = evolve_dict!(deepcopy(operator), generators, angles, strategy)
    spv_result = copy(vectorized)
    evolve!(
        spv_result,
        generators,
        angles;
        window=1,
        truncation=strategy,
    )
    evolution_diff = max_coefficient_difference(dict_result, PauliSum(spv_result))
    evolution_diff <= 1e-10 || error("evolution parity failure: $evolution_diff")
    PauliOperators.check_spv(spv_result)

    dict_direct_bench = @benchmarkable truncate!(result, $strategy) setup=(
        result=deepcopy($overweight)
    )
    spv_direct_bench = @benchmarkable truncate!(result, $strategy) setup=(
        result=copy($vectorized_overweight)
    )
    dict_evolution_bench = @benchmarkable evolve_dict!(
        result,
        $generators,
        $angles,
        $strategy,
    ) setup=(result=deepcopy($operator))
    spv_evolution_bench = @benchmarkable evolve!(
        result,
        $generators,
        $angles;
        window=1,
        truncation=$strategy,
    ) setup=(result=copy($vectorized))
    conversion_bench = @benchmarkable SparsePauliVector(
        $operator;
        T=Float64,
        capacity_factor=8,
        append_factor=4,
    )
    dict_end_to_end_bench = @benchmarkable evolve(
        $operator,
        $generators,
        $angles;
        truncation=$strategy,
    )
    spv_end_to_end_bench = @benchmarkable begin
        result = SparsePauliVector(
            $operator;
            T=Float64,
            capacity_factor=8,
            append_factor=4,
        )
        evolve!(
            result,
            $generators,
            $angles;
            window=1,
            truncation=$strategy,
        )
    end

    dict_direct_est = measured(dict_direct_bench, samples)
    spv_direct_est = measured(spv_direct_bench, samples)
    dict_evolution_est = measured(dict_evolution_bench, samples)
    spv_evolution_est = measured(spv_evolution_bench, samples)
    conversion_est = measured(conversion_bench, samples)
    dict_end_to_end_est = measured(dict_end_to_end_bench, samples)
    spv_end_to_end_est = measured(spv_end_to_end_bench, samples)

    saved_per_rotation = (
        dict_evolution_est.time - spv_evolution_est.time
    ) / rotations
    break_even = saved_per_rotation > 0 ?
        ceil(Int, conversion_est.time / saved_per_rotation) : typemax(Int)

    return (
        strategy=name,
        N=N,
        initial_terms=nterms,
        k=k,
        rotations=rotations,
        final_terms=length(spv_result),
        direct_max_diff=direct_diff,
        evolution_max_diff=evolution_diff,
        max_diff=max(direct_diff, evolution_diff),
        direct_dict_ms=dict_direct_est.time / 1e6,
        direct_spv_ms=spv_direct_est.time / 1e6,
        direct_speedup=dict_direct_est.time / spv_direct_est.time,
        direct_dict_allocs=dict_direct_est.allocs,
        direct_spv_allocs=spv_direct_est.allocs,
        direct_dict_kib=dict_direct_est.memory / 2.0^10,
        direct_spv_kib=spv_direct_est.memory / 2.0^10,
        evolution_dict_ms=dict_evolution_est.time / 1e6,
        evolution_spv_ms=spv_evolution_est.time / 1e6,
        evolution_speedup=dict_evolution_est.time / spv_evolution_est.time,
        conversion_ms=conversion_est.time / 1e6,
        conversion_allocs=conversion_est.allocs,
        conversion_mib=conversion_est.memory / 2.0^20,
        end_to_end_dict_ms=dict_end_to_end_est.time / 1e6,
        end_to_end_spv_ms=spv_end_to_end_est.time / 1e6,
        end_to_end_speedup=dict_end_to_end_est.time / spv_end_to_end_est.time,
        end_to_end_dict_allocs=dict_end_to_end_est.allocs,
        end_to_end_spv_allocs=spv_end_to_end_est.allocs,
        end_to_end_dict_mib=dict_end_to_end_est.memory / 2.0^20,
        end_to_end_spv_mib=spv_end_to_end_est.memory / 2.0^20,
        break_even_rotations=break_even,
        dict_allocs=dict_evolution_est.allocs,
        spv_allocs=spv_evolution_est.allocs,
        dict_mib=dict_evolution_est.memory / 2.0^20,
        spv_mib=spv_evolution_est.memory / 2.0^20,
    )
end

function print_result(result)
    @printf(
        "%s N=%d terms=%d k=%d rotations=%d final=%d diff=%.3e\n",
        result.strategy,
        result.N,
        result.initial_terms,
        result.k,
        result.rotations,
        result.final_terms,
        result.max_diff,
    )
    @printf(
        "  direct: %.3f ms -> %.3f ms (%.2fx), %.2f -> %.2f KiB, %d -> %d allocs\n",
        result.direct_dict_ms,
        result.direct_spv_ms,
        result.direct_speedup,
        result.direct_dict_kib,
        result.direct_spv_kib,
        result.direct_dict_allocs,
        result.direct_spv_allocs,
    )
    @printf(
        "  evolution: %.3f ms -> %.3f ms (%.2fx), %.2f -> %.2f MiB, %d -> %d allocs\n",
        result.evolution_dict_ms,
        result.evolution_spv_ms,
        result.evolution_speedup,
        result.dict_mib,
        result.spv_mib,
        result.dict_allocs,
        result.spv_allocs,
    )
    @printf(
        "  end-to-end: %.3f ms -> %.3f ms (%.2fx), %.2f -> %.2f MiB, %d -> %d allocs\n",
        result.end_to_end_dict_ms,
        result.end_to_end_spv_ms,
        result.end_to_end_speedup,
        result.end_to_end_dict_mib,
        result.end_to_end_spv_mib,
        result.end_to_end_dict_allocs,
        result.end_to_end_spv_allocs,
    )
    @printf(
        "  conversion: %.3f ms, %.2f MiB, %d allocs; break-even %s rotations\n",
        result.conversion_ms,
        result.conversion_mib,
        result.conversion_allocs,
        result.break_even_rotations == typemax(Int) ? "never" : string(result.break_even_rotations),
    )
end

function run_matrix(; scenarios=DEFAULT_SCENARIOS, samples=5)
    results = NamedTuple[]
    for scenario in scenarios
        for strategy_case in strategy_cases(scenario.N, scenario.k)
            result = benchmark_case(scenario, strategy_case; samples=samples)
            push!(results, result)
            print_result(result)
        end
    end
    return results
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_matrix()
end
