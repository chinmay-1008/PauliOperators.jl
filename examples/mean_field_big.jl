#=
Mean-Field Factorization vs Weight Truncation — 2D Ising Dynamics
======================================================================

Compares Heisenberg-picture simulations of ⟨O(t)⟩ for the 2D Ising
model on an Nx × Ny lattice, starting from a product state.

    1. Trotter         — second-order Trotter, coefficient truncation baseline
    2. Trotter + WeightTruncation(k)       — drops all weight > k terms
    3. Trotter + MeanFieldTruncation(k, ψ) — expands weight > k terms in
                                              single-site fluctuations
                                              around |ψ⟩
=#

using PauliOperators
using LinearAlgebra
using Printf

function rectanglebricktopology(nx::Integer, ny::Integer)
    LI = LinearIndices((nx, ny))

    layer_A = [(LI[x, y], LI[x+1, y]) for x in 1:2:(nx-1) for y in 1:ny]
    layer_B = [(LI[x, y], LI[x+1, y]) for x in 2:2:(nx-1) for y in 1:ny]
    layer_C = [(LI[x, y], LI[x, y+1]) for x in 1:nx for y in 1:2:(ny-1)]
    layer_D = [(LI[x, y], LI[x, y+1]) for x in 1:nx for y in 2:2:(ny-1)]

    return vcat(layer_A, layer_B, layer_C, layer_D)
end

function get_ordered_generators(Nx, Ny, J, h)
    N = Nx * Ny
    generators = []
    angles = []
    
    topology = rectanglebricktopology(Nx, Ny)
    
    for (i, j) in topology
        push!(generators, PauliBasis(Pauli(N, Z=[i, j])))
        push!(angles, J)
    end
    
    for i in 1:N
        push!(generators, PauliBasis(Pauli(N, X=[i])))
        push!(angles, h)
    end
    
    return generators, angles
end

function ising_ham(Nx, Ny, J, h)
    generators, angles = get_ordered_generators(Nx, Ny, J, h)
    N = Nx * Ny
    H_sum = PauliSum(N, Float64)
    for j in eachindex(generators)
        H_sum += angles[j] * generators[j]
    end
    return H_sum
end

"""
Heisenberg-picture Trotter evolution of `O`. Prints live progress to the console.
"""
function trotter_ev_curve(H::PauliSum{N,T}, O::PauliSum{N,T}, ψ::Ket{N},
                          times::AbstractVector, dt::Real,
                          truncation::TruncationStrategy,
                          label::String) where {N,T}
    generators, angles = trotterize(H, dt; n_trotter=1, order=1)
    # generators, angles = get_ordered_generators(Nx, Ny, J, h)

    Ot = deepcopy(O)

    num_steps = length(times)
    ev      = zeros(Float64, num_steps)
    n_terms = zeros(Int,     num_steps)
    println(" length generators: $(length(generators))")

    ev[1]      = real(expectation_value(Ot, ψ))
    n_terms[1] = length(Ot)
    
    print("  [$label] Step 1/$num_steps | Terms: $(n_terms[1])")
    for step in 2:num_steps
        Ot = evolve(Ot, generators, angles; truncation=truncation)
        
        ev[step]      = real(expectation_value(Ot, ψ))
        n_terms[step] = length(Ot)
        
        print("\r  [$label] Step $step/$num_steps | Terms: $(n_terms[step])                  ")
    end
    println() 
    return ev, n_terms
end

function main()
    # ── Setup ────────────────────────────────────────────────────────────────
    Nx     = 3
    Ny     = 3
    N      = Nx * Ny
    J      = 1.0
    h      = 0.5
    dt     = 0.1
    T_max  = 5.0
    times  = collect(0.0:dt:T_max)
    k_max  = 5

    H = ising_ham(Nx, Ny, J, h)

    # Initial state
    # ψ = Ket(N, Int128(0b0101010101))
    ψ = Ket(N, 0)
    c_ind = (Nx ÷ 2 + 1) + (Ny ÷ 2) * Nx

    # Observable
    O = PauliSum(Pauli(N; Z=[c_ind]))
    # O += Pauli(N; Z=[3,])
    # mul!(O, 1/norm(O))

    println("=" ^ 82)
    println("  2D Ising | N=$N sites ($Nx × $Ny), J=$J, h=$h, ψ=$(ψ.v), observable Z_$c_ind")
    println("  dt=$dt, T=$T_max, truncation weight k=$k_max")
    println("=" ^ 82)
    println()

    # Run dynamics with live printing
    ev_tr, nt_tr = trotter_ev_curve(H, O, ψ, times, dt, 
                                    CoeffTruncation(1e-6), 
                                    "Trotter (1e-6)")

    ev_wt, nt_wt = trotter_ev_curve(H, O, ψ, times, dt,
                                    CompositeTruncation(WeightTruncation(k_max), CoeffTruncation(1e-5)), 
                                    "WeightTrunc(k=$k_max)")

    ev_mf, nt_mf = trotter_ev_curve(H, O, ψ, times, dt,
                                    CompositeTruncation(MeanFieldTruncation(k_max, ψ), CoeffTruncation(1e-5)), 
                                    "MeanField(k=$k_max)")

    println()
    @printf("  %5s  %+10s  %+10s  %+10s    %5s %5s %5s\n",
            "t", "trotter", "weight($k_max)", "MF($k_max)",
            "#trot", "#wt", "#mf")
    println("  " * "-" ^ 82)
    
    # Print every 5th step to keep the final output block readable
    for i in 1:5:length(times)
        @printf("  %5.2f    %+10.6f  %+10.6f  %+10.6f    %5d %5d %5d\n",
                times[i], ev_tr[i], ev_wt[i], ev_mf[i],
                nt_tr[i], nt_wt[i], nt_mf[i])
    end
    println()

    return (times=times, trotter=ev_tr,
            weight=ev_wt, meanfield=ev_mf,
            nt_tr=nt_tr, nt_wt=nt_wt, nt_mf=nt_mf,
            k=k_max, N=N, Nx=Nx, Ny=Ny)
end

results = main()

# ── Optional plot (requires Plots.jl) ────────────────────────────────────────
try
    using Plots
    using Plots.PlotMeasures 

    p1 = plot(color=:black, lw=2.5, ls=:solid,
              xlabel="t", ylabel="⟨ψ| O(t) |ψ⟩",
              title="2D Ising, $(results.Nx)×$(results.Ny), k=$(results.k)",
              legend=:bottomleft, margin=5mm)
    plot!(p1, results.times, results.trotter,
          label="Trotter (1e-6)", color=:gray, ls=:dash)
    plot!(p1, results.times, results.weight,
          label="WeightTruncation($(results.k))",
          color=:red,  lw=2, marker=:circle, ms=3)
    plot!(p1, results.times, results.meanfield,
          label="MeanFieldTruncation($(results.k), ψ)",
          color=:blue, lw=2, marker=:square, ms=3)

    p2 = plot(results.times, results.nt_tr,
              label="Trotter (1e-4)", yscale=:log10,
              xlabel="t", ylabel="# Pauli terms",
              title="Operator size",
              color=:gray, ls=:dash, margin=5mm)
    plot!(p2, results.times, results.nt_wt, label="WeightTruncation",
          color=:red,  marker=:circle, ms=3)
    plot!(p2, results.times, results.nt_mf, label="MeanFieldTruncation",
          color=:blue, marker=:square, ms=3)

    fig = plot(p1, p2, layout=(2, 1), size=(900, 700), margin=5mm)
    outfile = joinpath(@__DIR__, "mean_field_ising$(results.N).png")
    savefig(fig, outfile)
    println("  Plot saved to: $outfile")
catch e
    if isa(e, ArgumentError) || isa(e, LoadError)
        println("  [Plots.jl not available — skipping plot]")
    else
        rethrow(e)
    end
end