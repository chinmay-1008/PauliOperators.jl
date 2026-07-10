#!/usr/bin/env julia

# Diagnostic study for ordinary mean-field truncation and multireference
# cumulants. This example uses the PauliOperators.jl public API for operator
# construction, expectation values, traces, truncation, and evolution.

ENV["GKSwstype"] = "100"

import Pkg
Pkg.activate(@__DIR__; io=devnull)
Pkg.instantiate(; io=devnull)

using LinearAlgebra
using PauliOperators
using Plots
using Printf
using Random
using Statistics

const OUTPUT_DIR = joinpath(@__DIR__, "outputs", "mean_field_cumulant_study")

struct InfiniteTemperatureReference{N} end

function basis_from_factors(N::Int, factors)
    X = Int[]
    Y = Int[]
    Z = Int[]
    seen = Set{Int}()
    for (site, op) in factors
        1 <= site <= N || throw(ArgumentError("site $site is outside 1:$N"))
        site in seen && throw(ArgumentError("duplicate operator on site $site"))
        push!(seen, site)
        op == 'X' ? push!(X, site) :
        op == 'Y' ? push!(Y, site) :
        op == 'Z' ? push!(Z, site) :
        throw(ArgumentError("unsupported Pauli factor $op"))
    end
    return PauliBasis(Pauli(N; X=X, Y=Y, Z=Z))
end

function factors_from_basis(pb::PauliBasis)
    str = string(pb)
    return [(site, str[site]) for site in eachindex(str) if str[site] != 'I']
end

function moment(factors, ref::Union{Ket{N},KetSum{N}}) where {N}
    isempty(factors) && return 1.0 + 0im
    return expectation_value(PauliSum(basis_from_factors(N, factors); T=ComplexF64), ref)
end

function moment(factors, ::InfiniteTemperatureReference{N}) where {N}
    isempty(factors) && return 1.0 + 0im
    O = PauliSum(basis_from_factors(N, factors); T=ComplexF64)
    return tr(O) / 2^N
end

function set_partitions_indices(n::Int)
    n < 0 && throw(ArgumentError("partition size must be nonnegative"))
    n == 0 && return [Vector{Vector{Int}}()]
    smaller = set_partitions_indices(n - 1)
    partitions = Vector{Vector{Vector{Int}}}()
    for partition in smaller
        with_singleton = [copy(block) for block in partition]
        push!(with_singleton, [n])
        push!(partitions, with_singleton)
        for block_idx in eachindex(partition)
            grown = [copy(block) for block in partition]
            push!(grown[block_idx], n)
            push!(partitions, grown)
        end
    end
    return partitions
end

function cumulant(factors, ref)
    n = length(factors)
    n == 0 && return 0.0 + 0im
    total = 0.0 + 0im
    for partition in set_partitions_indices(n)
        block_count = length(partition)
        coefficient = (-1)^(block_count - 1) * factorial(block_count - 1)
        block_product = 1.0 + 0im
        for block in partition
            block_product *= moment(factors[block], ref)
        end
        total += coefficient * block_product
    end
    return total
end

function vector_to_ketsum(v::AbstractVector, N::Int; cutoff=1e-10)
    ref = KetSum(N, T=ComplexF64)
    for idx in eachindex(v)
        abs(v[idx]) > cutoff || continue
        ref[Ket(N, Int128(idx - 1))] = ComplexF64(v[idx])
    end
    norm_sq = inner_product(ref, ref)
    norm_sq == 0 && throw(ArgumentError("vector produced an empty KetSum"))
    inv_norm = inv(sqrt(norm_sq))
    for ket in keys(ref)
        ref[ket] *= inv_norm
    end
    return ref
end

function heisenberg_xxz(N::Int; J::Real=1.0, xy::Real=0.1, zz::Real=2.0)
    H = PauliSum(N, ComplexF64)
    for site in 1:(N - 1)
        H[PauliBasis(Pauli(N; X=[site, site + 1]))] = xy * J + 0im
        H[PauliBasis(Pauli(N; Y=[site, site + 1]))] = xy * J + 0im
        H[PauliBasis(Pauli(N; Z=[site, site + 1]))] = zz * J + 0im
    end
    return H
end

function heisenberg_xxx(N::Int; J::Real=1.0)
    H = PauliSum(N, ComplexF64)
    for site in 1:(N - 1)
        H[PauliBasis(Pauli(N; X=[site, site + 1]))] = J + 0im
        H[PauliBasis(Pauli(N; Y=[site, site + 1]))] = J + 0im
        H[PauliBasis(Pauli(N; Z=[site, site + 1]))] = J + 0im
    end
    return H
end

function exact_ev_curve(H::PauliSum, O::PauliSum, ref, times)
    Hm = Hermitian(Matrix(H))
    Om = Matrix(O)
    eig = eigen(Hm)
    psi0 = ComplexF64.(Vector(ref))
    psi0 ./= norm(psi0)
    coeffs = eig.vectors' * psi0
    ev = zeros(Float64, length(times))
    for (idx, t) in enumerate(times)
        psit = eig.vectors * (cis.(-t .* eig.values) .* coeffs)
        ev[idx] = real(psit' * Om * psit)
    end
    return ev
end

function trotter_curve(H::PauliSum{N,T}, O::PauliSum{N,T}, ref, times, dt,
                       truncation::TruncationStrategy) where {N,T}
    generators, angles = trotterize(H, dt; n_trotter=1, order=2)
    Ot = deepcopy(O)
    ev = zeros(Float64, length(times))
    n_terms = zeros(Int, length(times))
    ev[1] = real(expectation_value(Ot, ref))
    n_terms[1] = length(Ot)
    for step in 2:length(times)
        Ot = evolve(Ot, generators, angles; truncation=truncation)
        ev[step] = real(expectation_value(Ot, ref))
        n_terms[step] = length(Ot)
    end
    return (ev=ev, n_terms=n_terms, final_operator=Ot)
end

function max_weight_seen(O::PauliSum)
    isempty(O) && return 0
    return maximum(weight(pb) for pb in keys(O))
end

function pearson(x, y; atol=1e-12)
    length(x) == length(y) || throw(DimensionMismatch("vectors must have same length"))
    length(x) < 2 && return NaN
    (maximum(abs.(x)) < atol || maximum(abs.(y)) < atol) && return NaN
    xc = x .- mean(x)
    yc = y .- mean(y)
    denom = norm(xc) * norm(yc)
    denom == 0 && return NaN
    return dot(xc, yc) / denom
end

function pauli_cumulant_stats(pb::PauliBasis, ref; max_triples=12)
    factors = factors_from_basis(pb)
    n = length(factors)
    max_pair = 0.0
    max_triple = 0.0
    pair_count = 0
    triple_count = 0
    if n >= 2
        for i in 1:(n - 1), j in (i + 1):n
            max_pair = max(max_pair, abs(cumulant(factors[[i, j]], ref)))
            pair_count += 1
        end
    end
    if n >= 3
        for i in 1:(n - 2), j in (i + 1):(n - 1), k in (j + 1):n
            triple_count >= max_triples && break
            max_triple = max(max_triple, abs(cumulant(factors[[i, j, k]], ref)))
            triple_count += 1
        end
    end
    full_kappa = n <= 4 ? abs(cumulant(factors, ref)) : NaN
    return (weight=n, max_pair=max_pair, max_triple=max_triple,
            full_kappa=full_kappa, pair_count=pair_count, triple_count=triple_count)
end

function normalize_curve(x; atol=1e-12)
    m = maximum(abs.(x))
    m < atol && return zero.(x)
    return x ./ m
end

function write_metrics_csv(path, rows)
    open(path, "w") do io
        println(io, "scenario,metric,value")
        for (scenario, metric, value) in rows
            println(io, "\"$scenario\",\"$metric\",$value")
        end
    end
end

function write_diagnostics_csv(path, rows)
    open(path, "w") do io
        println(io, "time,expectation,abs_error,pair_signal,triple_signal,high_terms_seen")
        for row in rows
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g,%d\n",
                    row.time, row.expectation, row.abs_error, row.pair_signal,
                    row.triple_signal, row.high_terms_seen)
        end
    end
end

function assert_close(label, actual, expected; atol=1e-9)
    abs(actual - expected) <= atol && return
    error("$label failed: got $actual, expected $expected")
end

function run_reference_tests()
    product_ref = Ket(4, 0)
    assert_close("product kappa Z1 Z2", cumulant([(1, 'Z'), (2, 'Z')], product_ref), 0.0 + 0im)
    assert_close("product kappa X1 X2", cumulant([(1, 'X'), (2, 'X')], product_ref), 0.0 + 0im)

    bell = KetSum(2, T=ComplexF64)
    bell[Ket(2, 0b00)] = inv(sqrt(2)) + 0im
    bell[Ket(2, 0b11)] = inv(sqrt(2)) + 0im
    assert_close("Bell <Z1>", moment([(1, 'Z')], bell), 0.0 + 0im)
    assert_close("Bell <Z2>", moment([(2, 'Z')], bell), 0.0 + 0im)
    assert_close("Bell <Z1 Z2>", moment([(1, 'Z'), (2, 'Z')], bell), 1.0 + 0im)
    assert_close("Bell kappa Z1 Z2", cumulant([(1, 'Z'), (2, 'Z')], bell), 1.0 + 0im)

    zz = PauliBasis("ZZ")
    mf_low = mean_field_factorize(zz, 1.0 + 0im, bell, 1)
    assert_close("single-site mean-field Bell ZZ expectation",
                 expectation_value(mf_low, bell), 0.0 + 0im)
    isempty(mf_low) || error("Bell ZZ should collapse to no terms under one-body KetSum mean field")

    infinite = InfiniteTemperatureReference{4}()
    for factors in [[(1, 'Z')], [(1, 'X'), (3, 'Y')], [(1, 'Z'), (2, 'Z'), (4, 'X')]]
        assert_close("infinite-temperature moment", moment(factors, infinite), 0.0 + 0im)
        assert_close("infinite-temperature cumulant", cumulant(factors, infinite), 0.0 + 0im)
    end

    return (bell=bell, bell_mf_low=mf_low)
end

function run_product_reference_baseline(output_dir)
    N = 10
    dt = 0.04
    T_max = 1.0
    times = collect(0.0:dt:T_max)
    ref = Ket(repeat([0, 1], 5))
    H = heisenberg_xxz(N; J=1.0, xy=0.1, zz=2.0)
    O = PauliSum(Pauli(N; Z=[1, 6]))
    k = 3
    coeff_threshold = 1e-5

    exact = exact_ev_curve(H, O, ref, times)
    weight_result = trotter_curve(H, O, ref, times, dt,
                                  CompositeTruncation(WeightTruncation(k),
                                                      CoeffTruncation(coeff_threshold)))
    mf_result = trotter_curve(H, O, ref, times, dt,
                              CompositeTruncation(MeanFieldTruncation(k, ref),
                                                  CoeffTruncation(coeff_threshold)))

    max_weight_seen(mf_result.final_operator) <= k ||
        error("mean-field product baseline produced terms above weight $k")

    p = plot(times, exact; label="exact", color=:black, lw=2.4,
             xlabel="t", ylabel="<Z1 Z6(t)>", title="XXZ product-reference baseline",
             framestyle=:box, legend=:bottomleft, size=(850, 520), dpi=160)
    plot!(p, times, weight_result.ev; label="WeightTruncation(k=$k)", color=:red, lw=2, ls=:dash)
    plot!(p, times, mf_result.ev; label="MeanFieldTruncation(k=$k)", color=:blue, lw=2, ls=:dot)
    savefig(p, joinpath(output_dir, "product_reference_curves.png"))

    return (times=times, exact=exact, weight=weight_result, mf=mf_result, k=k,
            dt=dt, T_max=T_max, N=N, coeff_threshold=coeff_threshold)
end

function run_multireference_diagnostic(output_dir)
    N = 8
    dt = 0.05
    T_max = 0.5
    times = collect(0.0:dt:T_max)
    k = 3
    coeff_threshold = 1e-5
    max_terms_per_gate = 10

    H = heisenberg_xxx(N; J=1.0)
    eig = eigen(Hermitian(Matrix(H)))
    ref = vector_to_ketsum(eig.vectors[:, 1], N; cutoff=1e-9)
    O = PauliSum(Pauli(N; Z=[1, 4]))
    exact_value = real(expectation_value(O, ref))

    generators, angles = trotterize(H, dt; n_trotter=1, order=2)
    truncation = CompositeTruncation(MeanFieldTruncation(k, ref), CoeffTruncation(coeff_threshold))
    Ot = deepcopy(O)
    rows = NamedTuple[]
    stats_cache = Dict{String,NamedTuple}()

    push!(rows, (time=times[1], expectation=real(expectation_value(Ot, ref)),
                 abs_error=abs(real(expectation_value(Ot, ref)) - exact_value),
                 pair_signal=0.0, triple_signal=0.0, high_terms_seen=0))

    for step in 2:length(times)
        pair_signal = 0.0
        triple_signal = 0.0
        high_terms_seen = 0
        for (generator, angle) in zip(generators, angles)
            evolve!(Ot, generator, angle)
            high_terms = [(pb, c) for (pb, c) in Ot if weight(pb) > k]
            sort!(high_terms; by=x -> abs(x[2]), rev=true)
            for (pb, c) in Iterators.take(high_terms, max_terms_per_gate)
                key = string(pb)
                stats = get!(stats_cache, key) do
                    pauli_cumulant_stats(pb, ref; max_triples=12)
                end
                pair_signal += abs(c) * stats.max_pair
                triple_signal += abs(c) * stats.max_triple
                high_terms_seen += 1
            end
            truncate!(Ot, truncation)
        end
        ev = real(expectation_value(Ot, ref))
        push!(rows, (time=times[step], expectation=ev, abs_error=abs(ev - exact_value),
                     pair_signal=pair_signal, triple_signal=triple_signal,
                     high_terms_seen=high_terms_seen))
    end

    write_diagnostics_csv(joinpath(output_dir, "multireference_cumulant_diagnostics.csv"), rows)

    errors = [row.abs_error for row in rows]
    pair_signals = [row.pair_signal for row in rows]
    triple_signals = [row.triple_signal for row in rows]
    corr_pair = pearson(errors[2:end], pair_signals[2:end])
    corr_triple = pearson(errors[2:end], triple_signals[2:end])

    p = plot(times, errors; label="abs expectation error", color=:black, lw=2.4,
             xlabel="t", ylabel="normalized diagnostic", title="Heisenberg eigenvector cumulant diagnostic",
             framestyle=:box, legend=:topleft, size=(850, 520), dpi=160)
    plot!(p, times, normalize_curve(pair_signals); label="weighted pair-cumulant signal", color=:blue, lw=2, ls=:dash)
    plot!(p, times, normalize_curve(triple_signals); label="weighted triple-cumulant signal", color=:green, lw=2, ls=:dot)
    savefig(p, joinpath(output_dir, "multireference_error_cumulant.png"))

    return (N=N, dt=dt, T_max=T_max, k=k, coeff_threshold=coeff_threshold,
            exact_value=exact_value, rows=rows, corr_pair=corr_pair,
            corr_triple=corr_triple, stats_cache=stats_cache)
end

function write_summary(path, product, multi)
    open(path, "w") do io
        println(io, "# Mean-Field And Cumulant Reference-State Study Results")
        println(io)
        println(io, "Generated by `mean_field_cumulant_study.jl`.")
        println(io)
        println(io, "## Package Usage")
        println(io, "- This example is its own Julia project and references `PauliOperators` with a local `[sources]` path.")
        println(io, "- Expectations use `expectation_value`; infinite-temperature checks use `tr(O) / 2^N`; propagation uses `trotterize`, `evolve!`, `truncate!`, and `MeanFieldTruncation`.")
        println(io)
        println(io, "## Product Reference Baseline")
        @printf(io, "- XXZ chain: N=%d, dt=%.3g, T_max=%.3g, k=%d, coefficient threshold=%.1e.\n",
                product.N, product.dt, product.T_max, product.k, product.coeff_threshold)
        @printf(io, "- WeightTruncation max error: %.6g.\n",
                maximum(abs.(product.weight.ev .- product.exact)))
        @printf(io, "- MeanFieldTruncation max error: %.6g.\n",
                maximum(abs.(product.mf.ev .- product.exact)))
        @printf(io, "- Final mean-field term count: %d.\n", product.mf.n_terms[end])
        println(io, "- Product-state connected cross-site cumulant checks passed.")
        println(io)
        println(io, "## Bell/GHZ-Style Multireference Check")
        println(io, "- Bell reference has <Z1>=<Z2>=0 and <Z1 Z2>=kappa(Z1,Z2)=1.")
        println(io, "- Current single-site KetSum mean field maps ZZ to no low-order terms at k=1.")
        println(io)
        println(io, "## Heisenberg Multireference Diagnostic")
        @printf(io, "- XXX chain eigenvector: N=%d, dt=%.3g, T_max=%.3g, k=%d.\n",
                multi.N, multi.dt, multi.T_max, multi.k)
        @printf(io, "- Exact stationary reference expectation: %.8g.\n", multi.exact_value)
        @printf(io, "- Final mean-field expectation error: %.6g.\n", multi.rows[end].abs_error)
        @printf(io, "- Error/pair-cumulant-signal correlation: %.6g.\n", multi.corr_pair)
        @printf(io, "- Error/triple-cumulant-signal correlation: %.6g.\n", multi.corr_triple)
        println(io)
        println(io, "## Infinite-Temperature Control")
        println(io, "- Non-identity Pauli moments and connected cumulants vanish, so no cumulant correction is available.")
        println(io)
        println(io, "## Output Files")
        println(io, "- `product_reference_curves.png`")
        println(io, "- `multireference_error_cumulant.png`")
        println(io, "- `metrics.csv`")
        println(io, "- `multireference_cumulant_diagnostics.csv`")
    end
end

function main()
    Random.seed!(0xC0FFEE)
    mkpath(OUTPUT_DIR)
    reference_checks = run_reference_tests()
    product = run_product_reference_baseline(OUTPUT_DIR)
    multi = run_multireference_diagnostic(OUTPUT_DIR)

    metrics = Tuple{String,String,Float64}[]
    push!(metrics, ("product_reference", "weight_max_error", maximum(abs.(product.weight.ev .- product.exact))))
    push!(metrics, ("product_reference", "mean_field_max_error", maximum(abs.(product.mf.ev .- product.exact))))
    push!(metrics, ("product_reference", "mean_field_final_error", abs(product.mf.ev[end] - product.exact[end])))
    push!(metrics, ("product_reference", "mean_field_final_terms", Float64(product.mf.n_terms[end])))
    push!(metrics, ("bell_reference", "kappa_Z1_Z2", real(cumulant([(1, 'Z'), (2, 'Z')], reference_checks.bell))))
    push!(metrics, ("bell_reference", "single_site_mf_ZZ_terms_at_k1", Float64(length(reference_checks.bell_mf_low))))
    push!(metrics, ("multireference_heisenberg", "final_abs_error", multi.rows[end].abs_error))
    push!(metrics, ("multireference_heisenberg", "pair_signal_error_correlation", multi.corr_pair))
    push!(metrics, ("multireference_heisenberg", "triple_signal_error_correlation", multi.corr_triple))

    write_metrics_csv(joinpath(OUTPUT_DIR, "metrics.csv"), metrics)
    write_summary(joinpath(OUTPUT_DIR, "mean_field_cumulant_study_results.md"), product, multi)

    println("Study complete.")
    println("Output directory: $(OUTPUT_DIR)")
    println("Summary: $(joinpath(OUTPUT_DIR, "mean_field_cumulant_study_results.md"))")
end

main()
