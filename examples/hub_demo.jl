using PauliOperators
using LinearAlgebra
using Printf
using Plots

function JWmapping(N; i::Int, j::Int)
    # Use bit-shifting instead of powers for performance and clarity
    # shift = 2^(idx-1)
    shift_i = Int128(1) << (i - 1)
    shift_j = Int128(1) << (j - 1)

    # Construct C_i^dagger terms
    # X_i string: (Z...Z) X_i, Y_i string: (Z...Z) Y_i
    # Note: Pauli{N}(phase, z_bits, x_bits)
    ax = Pauli{N}(1, shift_i - 1, shift_i)
    ay = Pauli{N}(1, (Int128(1) << i) - 1, shift_i)
    
    # Construct C_j terms
    bx = Pauli{N}(1, shift_j - 1, shift_j)
    by = Pauli{N}(1, (Int128(1) << j) - 1, shift_j)

    # c_i^dagger = 0.5 * (X_i - iY_i)
    # c_j        = 0.5 * (X_j + iY_j)
    # We use PauliBasis only at the end to minimize overhead
    op_i_dag = 0.5 * (PauliBasis(ax) - im * PauliBasis(ay))
    op_j     = 0.5 * (PauliBasis(bx) + im * PauliBasis(by))

    return op_i_dag * op_j
end


"""
 1D Fermi-Hubbard model 
Generate a 1D Fermi-Hubbard Hamiltonian (open boundaries, no PBC)   
using JW mapping into Pauli operators.

Arguments:
- o::Pauli{N} : reference Pauli object
- L::Int       : number of sites
- t::Float64   : hopping amplitude
- U::Float64   : on-site interaction
- k::Int       : number of Trotter steps (can be used later for evolution)

Returns:
- generators::Vector{Pauli{N}}
- parameters::Vector{Float64}
"""

function hubbard_model_1D(L::Int, t::Float64, U::Float64)
    
    N_total = 2 * L   # Total number of fermionic modes (spin up and down)
    H = PauliSum(N_total, Float64)

    # Hopping terms
    for i in 1:L-1
        # spin-up
        a_up = 2*i - 1
        b_up = 2*(i+1) - 1
        hopping_up = JWmapping(N_total, i=a_up, j=b_up) + JWmapping(N_total, i=b_up, j=a_up)

        # spin-down
        a_dn = 2*i
        b_dn = 2*(i+1)
        hopping_dn = JWmapping(N_total, i=a_dn, j=b_dn) + JWmapping(N_total, i=b_dn, j=a_dn)
        
        # Add both
        H += -t * (hopping_up + hopping_dn)
    end

    # On-site interaction terms``
    for i in 1:L
        a_up = 2*i - 1   # spin-up orbital index
        a_dn = 2*i       # spin-down orbital index
        interaction_term = U *JWmapping(N_total, i=a_up, j=a_up) * JWmapping(N_total, i=a_dn, j=a_dn)

        H += interaction_term
    end

    #Filter zero coefficients
    truncate!(H, CoeffTruncation(1e-12))

    return H    
end

function hubbard_model_1D_pair_interleaved(L::Int, t::Float64, U::Float64)
    N_total = 2 * L   # Total number of fermionic modes (spin up and down)
    H = PauliSum(N_total, Float64)

    # Helper function to match the paper's pair-interleaved ordering
    # i is the site index (1 to L)
    # is_up is a boolean: true for spin-up, false for spin-down
    function get_index(i::Int, is_up::Bool)
        if isodd(i)
            # Odd sites: [down, up]
            return is_up ? 2*i : 2*i - 1
        else
            # Even sites: [up, down]
            return is_up ? 2*i - 1 : 2*i
        end
    end

    # Hopping terms
    for i in 1:L-1
        # spin-up
        a_up = get_index(i, true)
        b_up = get_index(i+1, true)
        hopping_up = JWmapping(N_total, i=a_up, j=b_up) + JWmapping(N_total, i=b_up, j=a_up)

        # spin-down
        a_dn = get_index(i, false)
        b_dn = get_index(i+1, false)
        hopping_dn = JWmapping(N_total, i=a_dn, j=b_dn) + JWmapping(N_total, i=b_dn, j=a_dn)
        
        # Add both
        H += -t * (hopping_up + hopping_dn)
    end

    # On-site interaction terms
    for i in 1:L
        a_up = get_index(i, true)
        a_dn = get_index(i, false)
        
        # n_up * n_dn = c^dag_up c_up * c^dag_dn c_dn
        interaction_term = U * JWmapping(N_total, i=a_up, j=a_up) * JWmapping(N_total, i=a_dn, j=a_dn)
        H += interaction_term
    end

    # Filter zero coefficients
    truncate!(H, CoeffTruncation(1e-12))

    return H    
end

function trotter_ev_curve(H::PauliSum{N,T}, O::PauliSum{N,T}, ψ::Ket{N},
                          times::AbstractVector, dt::Real,
                          truncation::TruncationStrategy, correction::CorrectionAccumulator=NoCorrection()) where {N,T}
    generators, angles = trotterize(H, dt; n_trotter=1, order=2)
    Ot = deepcopy(O)
    W_mat   = zeros(Float64, length(times), N + 1)
    ev      = zeros(Float64, length(times))
    var     = zeros(Float64, length(times))
    n_terms = zeros(Int,     length(times))
    ev[1]      = real(expectation_value(Ot, ψ))
    var[1]     = variance(Ot, ψ)
    n_terms[1] = length(Ot)
    W_mat[1, :] = get_weight_distribution(Ot, N)
    l2 = norm(H)
    for step in 2:length(times)
        Ot = evolve(Ot, generators, angles; truncation=truncation, correction=correction)
        # mul!(Ot, l2 / norm(Ot))
        # @show norm(Ot)
        if correction isa EnergyCorrection
            ev[step] = real(expectation_value(Ot, ψ) - correction.accumulated_energy)
        else
            ev[step]  = real(expectation_value(Ot, ψ))
        end
        var[step]     = variance(Ot, ψ)
        n_terms[step] = length(Ot)
        W_mat[step, :] = get_weight_distribution(Ot, N)
    end
    return ev, var, n_terms, W_mat
end

function exact_ev_curve(H::PauliSum{N}, O::PauliSum{N}, ψ::Ket{N},
                        times::AbstractVector) where {N}
    Hm  = Hermitian(Matrix(H))
    Om  = Matrix(O)
    Om2 = Om * Om
    ψv = zeros(ComplexF64, Int(2^N))
    ψv[Int(ψ.v) + 1] = 1.0

    F = eigen(Hm)
    λ, V = F.values, F.vectors
    c0 = V' * ψv

    ev  = zeros(Float64, length(times))
    var = zeros(Float64, length(times))
    for (k, t) in enumerate(times)
        ψt = V * (cis.(-t .* λ) .* c0)
        ev[k]  = real(ψt' * Om  * ψt)
        var[k] = real(ψt' * Om2 * ψt) - ev[k]^2
    end
    return ev, var
end

function get_weight_distribution(ps::PauliSum, N::Int)
    # Array to hold weights from 0 up to N (size N+1 to account for weight 0)
    dist = zeros(Float64, N + 1) 
    
    for (p, c) in ps
        w = weight(p)
        # We use squared coefficients (probability mass)
        dist[w + 1] += abs2(c) 
    end
    
    return dist
end

function run_test()
    L = 60
    N = 2 * L
    t = 1.0
    U = 2.0
    dt = 0.2
    T = 6.0
    # steps = Int(T / dt) + 1
    # println("steps: ", steps)
    times  = collect(0.0:dt:T)
    
    H = hubbard_model_1D_pair_interleaved(L, t, U)
    println("Hubbard Hamiltonian for L=$L sites generated.")
    
    generators, angles = trotterize(H, dt; n_trotter=1, order=1)
    
    # 1. State Initialization
    state = "01"^L
    bitvec = [isodd(i) ? 0 : 1 for i in 1:N] # Cleaner array comprehension
    ψ = Ket(bitvec)
    println("Expectation of H: ", expectation_value(H, ψ))

    # Helper function from your Hamiltonian builder
    function get_index(i::Int, is_up::Bool)
        if isodd(i)
            return is_up ? 2*i : 2*i - 1
        else
            return is_up ? 2*i - 1 : 2*i
        end
    end

    # 2. Target a specific PHYSICAL site
    target_site = 46 
    
    # Get the exact qubit indices for this site's orbitals
    q_up = get_index(target_site, true)
    q_dn = get_index(target_site, false)

    # 3. Build the observables
    n_up_op = 0.5 * (Pauli(N) - Pauli(N, Z = [q_up]))
    n_dn_op = 0.5 * (Pauli(N) - Pauli(N, Z = [q_dn]))
    total_n_op = n_up_op + n_dn_op
    
    println("\n--- Measurements for Physical Site $target_site ---")
    println("Expectation of n_up: ", expectation_value(n_up_op, ψ))
    println("Expectation of n_dn: ", expectation_value(n_dn_op, ψ))
    println("Total occupation:    ", expectation_value(total_n_op, ψ))


    # print("  exact (dense)...\n")
    # @time ev_exact, var_exact = exact_ev_curve(H, n_up_op, ψ, times)
    
    p1 = plot(#times, ev_exact, label="exact",
            color=:black, lw=2.5, ls=:solid,
            xlabel="Time (t)", ylabel="⟨n_$(target_site)(t)⟩",
            title="1D Fermi-Hubbard, L=$(L)",
            legend=:topleft, size=(800, 500), framestyle=:box, margin=5Plots.mm, dpi = 200)

    thresh = 1e-4
    @time ev_tr_up, var_tr_up, nt_tr_up, W_mat_coeft_up = trotter_ev_curve(H, n_up_op, ψ, times, dt, CoeffTruncation(thresh))


    # plot!(p1, times, ev_tr,
            # label="CoeffTruncation($(thresh))-up", color=:blue, lw = 2)

    @time ev_tr_dn, var_tr_dn, nt_tr_dn, W_mat_coeft_dn = trotter_ev_curve(H, n_dn_op, ψ, times, dt, CoeffTruncation(thresh))


    plot!(p1, times, ev_tr,
            label="CoeffTruncation($(thresh))-dn", color=:red, lw = 2)

    # Plot Spin-Up (Maroon line + markers)
    plot!(p1, steps_array, ev_tr_up,
          label="Simulation ↑", 
          color=color_up, 
          lw=2, 
          shape=marker_shape,
          markersize=5,
          markercolor=:white,        # White interior (open marker look)
          markerstrokecolor=color_up, # Colored outline
          markerstrokewidth=2)

    # Plot Spin-Down (Mauve line + markers)
    plot!(p1, steps_array, ev_tr_dn,
          label="Simulation ↓", 
          color=color_dn, 
          lw=2, 
          shape=marker_shape,
          markersize=5,
          markercolor=:white,        # White interior
          markerstrokecolor=color_dn, # Colored outline
          markerstrokewidth=2)
    savefig(p1, "hubbard_demo_exact_n_up$N.png")
end

run_test()
