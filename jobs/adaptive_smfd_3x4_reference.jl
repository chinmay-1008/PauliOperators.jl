# =============================================================================
# Adaptive SMFD Cartesian reference on the 3 x 4 periodic transverse-field
# Ising torus, packaged as a batch job.
#
# Submit from inside this directory, with a bare filename, because the wrapper
# copies $INFILE into $TMPDIR and runs it from there:
#
#     cd jobs
#     sbatch ../slurm.sh adaptive_smfd_3x4_reference.jl
#
# The environment must be instantiated once first, since Manifest.toml is not in
# the repository:
#
#     julia --project=$JULIAENV -e 'using Pkg; Pkg.instantiate()'
#
# SMFD is the only mean-field variant here. Every reference-dependent trajectory
# is propagated with
#
#     CompositeTruncation(
#         SingleSiteMeanFieldDecoupling(max_weight, reference; normalize = true),
#         CoeffTruncation(coefficient_cutoff),
#     )
#
# No RMFD, product-density, or low-weight variant is computed. Hard weight
# truncation is carried as the no-reference baseline, as in the notebooks.
#
# The protocol matches adaptive_smfd_3x4_bloch_vector_reference.ipynb: same
# couplings, digital sequence, observable, thresholds, physical state, and the
# same 516-candidate Bloch-ball grid. The static 516-trajectory bank, the
# time-slice diagnostic, and every plot are dropped; only the adaptive
# (vx, vy, vz) dynamics are computed here.
#
# Nothing is plotted. Everything the plots need is written to
#     results/adaptive_smfd_3x4_reference.jld2
# which is rewritten after each phase so a walltime kill keeps finished work.
# =============================================================================

using Pkg

announce(message) = (println(message); flush(stdout))

# Prefer an explicitly exported environment (the SLURM wrapper sets JULIAENV),
# then the repository this file sits in. If neither holds a Project.toml the
# file has been copied somewhere else to run, so leave the environment that
# julia --project= already established rather than clobbering it.
project_root = get(ENV, "JULIAENV", normpath(joinpath(@__DIR__, "..")))

if isfile(joinpath(project_root, "Project.toml"))
    Pkg.activate(project_root)
else
    @warn "no Project.toml found; keeping the active project" project_root Base.active_project()
end

announce("active project = $(Base.active_project())")

using JLD2
using LinearAlgebra
using PauliOperators
using Printf
using Statistics

# -----------------------------------------------------------------------------
# 1. Protocol
# -----------------------------------------------------------------------------

Lx = 3
Ly = 4
N = Lx * Ly

J = -1.0
h = 2.0
dt = 0.25
theta = pi / 18
phi = 0.0
n_steps = 20

max_weight = 3
coefficient_cutoff = 1e-7
times = collect(0:n_steps)

component_values = collect(-1.0:0.2:1.0)

# Krylov settings for the exact target. Identical in the 3 x 4 and 4 x 5 jobs so
# that the 3 x 4 run, which can still afford a dense cross-check, validates the
# settings the 4 x 5 run depends on.
krylov_dimension = 30
krylov_substeps = 4

site(x, y) = x + (y - 1) * Lx

bonds = Tuple{Int,Int}[]

for y in 1:Ly
    for x in 1:Lx
        current_site = site(x, y)
        right_site = site(mod1(x + 1, Lx), y)
        upper_site = site(x, mod1(y + 1, Ly))

        for neighbor_site in (right_site, upper_site)
            # A periodic wrap of length two would land back on the bond the
            # nearest-neighbor step already recorded, and pushing it twice
            # silently doubles J along that direction. Both extents here exceed
            # two, so this guard never fires; it is kept so the lattice builder
            # is the same one the 2 x 3 notebooks use.
            neighbor_site == current_site && continue
            bond = minmax(current_site, neighbor_site)
            bond in bonds || push!(bonds, bond)
        end
    end
end

@assert length(bonds) == 24
@assert length(unique(bonds)) == length(bonds)

generators = PauliBasis{N}[]
angles = Float64[]

# First half of the transverse-field layer.
for site_index in 1:N
    push!(generators, PauliBasis(Pauli(N; X = [site_index])))
    push!(angles, h * dt)
end

# Full nearest-neighbor interaction layer.
for (first_site, second_site) in bonds
    push!(generators, PauliBasis(Pauli(N; Z = [first_site, second_site])))
    push!(angles, 2 * J * dt)
end

# Second half of the transverse-field layer.
for site_index in 1:N
    push!(generators, PauliBasis(Pauli(N; X = [site_index])))
    push!(angles, h * dt)
end

physical_state = ProductBlochReference(N; theta = theta, phi = phi)
physical_vector = (
    vx = physical_state.x,
    vy = physical_state.y,
    vz = physical_state.z,
)

z2_operator = PauliSum(N, Float64)
z2_operator[PauliBasis(Pauli(N))] = 1 / N

for first_site in 1:(N - 1)
    for second_site in (first_site + 1):N
        pair = PauliBasis(Pauli(N; Z = [first_site, second_site]))
        z2_operator[pair] = 2 / N^2
    end
end

observable = SparsePauliVector(z2_operator; T = Float64)

announce("lattice = $(Lx) x $(Ly), N = $(N), periodic bonds = $(length(bonds))")
announce("rotations per digital step = $(length(generators))")
announce("digital steps = $(n_steps), dt = $(dt)")
@printf(
    "physical vector = (%.6f, %.6f, %.6f)\n",
    physical_vector.vx,
    physical_vector.vy,
    physical_vector.vz,
)
flush(stdout)

results_root = get(ENV, "SMFD_RESULTS_DIR", get(ENV, "SLURM_SUBMIT_DIR", @__DIR__))
results_file = normpath(joinpath(results_root, "results", "adaptive_smfd_3x4_reference.jld2"))
mkpath(dirname(results_file))

announce("results root = $(results_root)")

saved = Dict{String,Any}(
    "parameters" => (
        Lx = Lx,
        Ly = Ly,
        N = N,
        J = J,
        h = h,
        dt = dt,
        theta = theta,
        phi = phi,
        nsteps = n_steps,
        max_weight = max_weight,
        coefficient_cutoff = coefficient_cutoff,
        krylov_dimension = krylov_dimension,
        krylov_substeps = krylov_substeps,
    ),
    "times" => times,
    "bonds" => bonds,
    "physical_vector" => [physical_vector.vx, physical_vector.vy, physical_vector.vz],
)

function checkpoint!()
    jldopen(results_file, "w") do file
        for (key, value) in saved
            file[key] = value
        end
    end
    announce("  checkpoint -> $(results_file)")
end

# -----------------------------------------------------------------------------
# 2. Exact target
#
# The dense eigendecomposition the notebooks use is a 2^N x 2^N object. At N = 20
# that is 8.8 TB, so the exact curve is produced matrix-free instead: the state
# vector is carried through exp(-i H dt) by Lanczos with full reorthogonalization,
# repeated n_steps times. The definition of the target is unchanged, only how it
# is evaluated.
# -----------------------------------------------------------------------------

function tilted_product_state(N, theta, phi)
    zero_amplitude = cos(theta / 2)
    one_amplitude = cis(phi) * sin(theta / 2)
    state = Vector{ComplexF64}(undef, 1 << N)

    for bits in 0:(1 << N) - 1
        number_of_ones = count_ones(bits)
        state[bits + 1] = (
            zero_amplitude^(N - number_of_ones) *
            one_amplitude^number_of_ones
        )
    end

    return state
end

function ising_diagonal(N, bonds, J)
    diagonal = Vector{Float64}(undef, 1 << N)

    for bits in 0:(1 << N) - 1
        total = 0.0

        for (first_site, second_site) in bonds
            first_z = iszero(bits & (1 << (first_site - 1))) ? 1.0 : -1.0
            second_z = iszero(bits & (1 << (second_site - 1))) ? 1.0 : -1.0
            total += J * first_z * second_z
        end

        diagonal[bits + 1] = total
    end

    return diagonal
end

function apply_hamiltonian!(output, input, diagonal, N, h)
    @inbounds @simd for index in eachindex(output)
        output[index] = diagonal[index] * input[index]
    end

    for site_index in 1:N
        mask = 1 << (site_index - 1)
        @inbounds for bits in 0:(length(input) - 1)
            output[bits + 1] += h * input[xor(bits, mask) + 1]
        end
    end

    return output
end

function lanczos_propagate!(state, workspace, basis, diagonal, N, h, tau, krylov_dimension)
    initial_norm = norm(state)
    initial_norm == 0 && return 0.0

    basis[:, 1] .= state ./ initial_norm
    alphas = Float64[]
    betas = Float64[]
    used = 0
    breakdown = false

    for j in 1:krylov_dimension
        current = view(basis, :, j)
        apply_hamiltonian!(workspace, current, diagonal, N, h)
        push!(alphas, real(dot(current, workspace)))
        used = j

        # Two passes of full reorthogonalization. The basis is small and the
        # vectors are cheap here, and it keeps the tridiagonal faithful.
        for _ in 1:2
            for k in 1:j
                previous = view(basis, :, k)
                workspace .-= dot(previous, workspace) .* previous
            end
        end

        residual_norm = norm(workspace)

        # A residual this small is a happy breakdown: the Krylov space has
        # closed on itself and the step is exact, not under-converged.
        if residual_norm < 1e-12
            breakdown = true
            break
        end

        j == krylov_dimension && break

        push!(betas, residual_norm)
        basis[:, j + 1] .= workspace ./ residual_norm
    end

    tridiagonal = zeros(Float64, used, used)

    for index in 1:used
        tridiagonal[index, index] = alphas[index]
    end

    for index in 1:(used - 1)
        tridiagonal[index, index + 1] = betas[index]
        tridiagonal[index + 1, index] = betas[index]
    end

    coefficients = exp(-im * tau * tridiagonal)[:, 1] .* initial_norm

    fill!(state, 0)

    for j in 1:used
        state .+= coefficients[j] .* view(basis, :, j)
    end

    # Weight left on the last Krylov vector: the truncation error of this step.
    return breakdown ? 0.0 : abs(coefficients[used]) / initial_norm
end

function exact_trajectory_krylov(N, bonds, J, h, dt, theta, phi, n_steps)
    dimension = 1 << N
    diagonal = ising_diagonal(N, bonds, J)
    z2_diagonal = [((N - 2 * count_ones(bits)) / N)^2 for bits in 0:(dimension - 1)]

    state = tilted_product_state(N, theta, phi)
    workspace = Vector{ComplexF64}(undef, dimension)
    basis = Matrix{ComplexF64}(undef, dimension, krylov_dimension)

    values = zeros(n_steps + 1)
    values[1] = dot(abs2.(state), z2_diagonal)
    worst_residual = 0.0
    tau = dt / krylov_substeps

    for step in 1:n_steps
        for _ in 1:krylov_substeps
            residual = lanczos_propagate!(
                state,
                workspace,
                basis,
                diagonal,
                N,
                h,
                tau,
                krylov_dimension,
            )
            worst_residual = max(worst_residual, residual)
        end

        values[step + 1] = dot(abs2.(state), z2_diagonal)
    end

    return values, worst_residual
end

announce("")
announce("Exact target (matrix-free Lanczos) ...")
exact_timing = @timed exact_trajectory_krylov(N, bonds, J, h, dt, theta, phi, n_steps)
exact_curve, krylov_residual = exact_timing.value

@printf("  done in %.3f s, worst Krylov residual weight = %.3e\n", exact_timing.time, krylov_residual)
@assert krylov_residual < 1e-10

expected_initial_value = 1 / N + (N - 1) / N * physical_vector.vz^2
@assert abs(exact_curve[1] - expected_initial_value) < 1e-10

# The dense route is still affordable at N = 12, so it is computed here as the
# reference implementation: the Krylov curve above must reproduce it, which is
# what licenses the Krylov-only 4 x 5 job. The dense curve is the one carried
# forward, so this job reproduces the notebook bit for bit.
function dense_hamiltonian(N, bonds, J, h)
    dimension = 1 << N
    hamiltonian = zeros(Float64, dimension, dimension)

    for bits in 0:(dimension - 1)
        row = bits + 1

        for (first_site, second_site) in bonds
            first_z = iszero(bits & (1 << (first_site - 1))) ? 1.0 : -1.0
            second_z = iszero(bits & (1 << (second_site - 1))) ? 1.0 : -1.0
            hamiltonian[row, row] += J * first_z * second_z
        end

        for site_index in 1:N
            flipped_bits = xor(bits, 1 << (site_index - 1))
            hamiltonian[flipped_bits + 1, row] = h
        end
    end

    return Symmetric(hamiltonian)
end

function exact_trajectory_dense(N, bonds, J, h, dt, theta, phi, n_steps)
    spectrum = eigen(dense_hamiltonian(N, bonds, J, h))
    initial_state = tilted_product_state(N, theta, phi)
    initial_eigenbasis = spectrum.vectors' * initial_state
    z2_diagonal = [((N - 2 * count_ones(bits)) / N)^2 for bits in 0:(1 << N) - 1]

    values = zeros(n_steps + 1)

    for step in 0:n_steps
        phases = cis.(-step * dt .* spectrum.values)
        state = spectrum.vectors * (phases .* initial_eigenbasis)
        values[step + 1] = dot(abs2.(state), z2_diagonal)
    end

    return values
end

announce("Exact target (dense eigendecomposition cross-check) ...")
dense_timing = @timed exact_trajectory_dense(N, bonds, J, h, dt, theta, phi, n_steps)
dense_curve = dense_timing.value
krylov_deviation = maximum(abs.(exact_curve .- dense_curve))

@printf("  done in %.3f s\n", dense_timing.time)
@printf("  max |krylov - dense| = %.3e\n", krylov_deviation)
@assert krylov_deviation < 1e-10

exact_curve = dense_curve
saved["krylov_deviation_from_dense"] = krylov_deviation

@printf("  exact initial <Ztot^2> = %.8f\n", exact_curve[1])
@printf("  exact final   <Ztot^2> = %.8f\n", exact_curve[end])
flush(stdout)

saved["exact_curve"] = exact_curve
saved["krylov_residual"] = krylov_residual
checkpoint!()

# -----------------------------------------------------------------------------
# 3. Candidate references and the SMFD propagator
# -----------------------------------------------------------------------------

function cartesian_bloch_candidates(N, component_values, physical_vector)
    candidates = NamedTuple[]

    for vx in component_values
        for vy in component_values
            for vz in component_values
                if vx^2 + vy^2 + vz^2 <= 1 + 1e-12
                    push!(
                        candidates,
                        (
                            vx = vx,
                            vy = vy,
                            vz = vz,
                            reference = ProductBlochReference(N, vx, vy, vz),
                            source = :grid,
                        ),
                    )
                end
            end
        end
    end

    matched_is_present = any(candidates) do candidate
        isapprox(candidate.vx, physical_vector.vx; atol = 1e-12) &&
        isapprox(candidate.vy, physical_vector.vy; atol = 1e-12) &&
        isapprox(candidate.vz, physical_vector.vz; atol = 1e-12)
    end

    if !matched_is_present
        push!(
            candidates,
            (
                vx = physical_vector.vx,
                vy = physical_vector.vy,
                vz = physical_vector.vz,
                reference = ProductBlochReference(N; theta = theta, phi = phi),
                source = :matched,
            ),
        )
    end

    return candidates
end

function vector_distance_squared(candidate, target)
    return (
        (candidate.vx - target.vx)^2 +
        (candidate.vy - target.vy)^2 +
        (candidate.vz - target.vz)^2
    )
end

candidates = cartesian_bloch_candidates(N, component_values, physical_vector)

candidate_vx = [candidate.vx for candidate in candidates]
candidate_vy = [candidate.vy for candidate in candidates]
candidate_vz = [candidate.vz for candidate in candidates]

matched_index = argmin([
    vector_distance_squared(candidate, physical_vector)
    for candidate in candidates
])

announce("reference candidates = $(length(candidates))")

function smfd_strategy(reference)
    return CompositeTruncation(
        SingleSiteMeanFieldDecoupling(max_weight, reference; normalize = true),
        CoeffTruncation(coefficient_cutoff),
    )
end

strategies = [smfd_strategy(candidate.reference) for candidate in candidates]

# The no-reference baseline: over-weight terms are deleted outright rather than
# demoted onto a mean-field reference.
hard_strategy = CompositeTruncation(
    WeightTruncation(max_weight),
    CoeffTruncation(coefficient_cutoff),
)

function advance_one_digital_step(operator, strategy)
    next_operator = deepcopy(operator)
    evolve!(
        next_operator,
        generators,
        angles;
        window = 1,
        truncation = strategy,
    )
    return next_operator
end

function curve_rmse(curve)
    errors = curve[2:end] .- exact_curve[2:end]
    return sqrt(mean(abs2, errors))
end

# Propagate with a prescribed truncation per digital step. A constant schedule
# reproduces a static bank entry; a varying one replays an adaptive path.
function propagate_strategies(step_strategies)
    operator = deepcopy(observable)
    curve = Vector{Float64}(undef, length(times))
    term_counts = Vector{Int}(undef, length(times))

    curve[1] = real(expectation_value(operator, physical_state))
    term_counts[1] = length(operator)

    for step in 1:n_steps
        operator = advance_one_digital_step(operator, step_strategies[step])
        curve[step + 1] = real(expectation_value(operator, physical_state))
        term_counts[step + 1] = length(operator)
    end

    return (curve = curve, term_counts = term_counts)
end

function closest_minimum(values, target; atol = 1e-12)
    minimum_value = minimum(values)
    tied_indices = findall(value -> value <= minimum_value + atol, values)
    distances = [vector_distance_squared(candidates[index], target) for index in tied_indices]
    return tied_indices[argmin(distances)]
end

candidate_vector(index) = (
    vx = candidates[index].vx,
    vy = candidates[index].vy,
    vz = candidates[index].vz,
)

# -----------------------------------------------------------------------------
# 4. Greedy controller
#
# Same selection rule as the notebook, including the tie break toward the
# previously chosen vector, and deterministic in exactly the same way. The one
# difference is bookkeeping: the notebook keeps all 516 trial operators of a step
# alive at once, which at N = 20 is several hundred megabytes of churn per step.
# Here the trial operators are discarded as soon as their expectation value is
# read and the winner is re-evolved, costing one extra propagation out of 516.
# -----------------------------------------------------------------------------

function greedy_adaptive()
    operator = deepcopy(observable)
    curve = Vector{Float64}(undef, length(times))
    term_counts = Vector{Int}(undef, length(times))
    chosen_indices = zeros(Int, length(times))
    candidate_errors = Matrix{Float64}(undef, length(candidates), n_steps)

    curve[1] = real(expectation_value(operator, physical_state))
    term_counts[1] = length(operator)

    for step in 1:n_steps
        step_started = time()
        candidate_values = Vector{Float64}(undef, length(candidates))

        for candidate_index in eachindex(candidates)
            trial_operator = advance_one_digital_step(operator, strategies[candidate_index])
            candidate_values[candidate_index] = real(
                expectation_value(trial_operator, physical_state),
            )
        end

        errors = abs.(candidate_values .- exact_curve[step + 1])
        candidate_errors[:, step] .= errors

        target = step == 1 ? physical_vector : candidate_vector(chosen_indices[step])
        best_index = closest_minimum(errors, target)

        operator = advance_one_digital_step(operator, strategies[best_index])
        curve[step + 1] = candidate_values[best_index]
        term_counts[step + 1] = length(operator)
        chosen_indices[step + 1] = best_index

        @printf(
            "  step %2d/%d  v = (%+.1f, %+.1f, %+.1f)  |err| = %.3e  terms = %6d  %6.1f s\n",
            step,
            n_steps,
            candidates[best_index].vx,
            candidates[best_index].vy,
            candidates[best_index].vz,
            errors[best_index],
            term_counts[step + 1],
            time() - step_started,
        )
        flush(stdout)
    end

    return (
        curve = curve,
        term_counts = term_counts,
        chosen = chosen_indices,
        candidate_errors = candidate_errors,
    )
end

announce("")
announce("Greedy Cartesian controller over $(length(candidates)) references ...")
greedy_timing = @timed greedy_adaptive()
greedy = greedy_timing.value

greedy_vx = [candidates[index].vx for index in greedy.chosen[2:end]]
greedy_vy = [candidates[index].vy for index in greedy.chosen[2:end]]
greedy_vz = [candidates[index].vz for index in greedy.chosen[2:end]]
greedy_norm = sqrt.(greedy_vx .^ 2 .+ greedy_vy .^ 2 .+ greedy_vz .^ 2)

@printf("  done in %.3f s; greedy RMSE = %.6e\n", greedy_timing.time, curve_rmse(greedy.curve))
announce("  greedy vx path = $(greedy_vx)")
announce("  greedy vy path = $(greedy_vy)")
announce("  greedy vz path = $(greedy_vz)")

saved["greedy_curve"] = greedy.curve
saved["greedy_term_counts"] = greedy.term_counts
saved["greedy_vx"] = greedy_vx
saved["greedy_vy"] = greedy_vy
saved["greedy_vz"] = greedy_vz
saved["greedy_norm"] = greedy_norm
saved["greedy_chosen"] = greedy.chosen
saved["greedy_candidate_errors"] = greedy.candidate_errors
saved["greedy_rmse"] = curve_rmse(greedy.curve)
saved["candidate_vx"] = candidate_vx
saved["candidate_vy"] = candidate_vy
saved["candidate_vz"] = candidate_vz
saved["matched_index"] = matched_index
checkpoint!()

# -----------------------------------------------------------------------------
# 5. Baselines
#
# Hard weight truncation, which uses no reference at all, and the matched static
# SMFD reference. The matched schedule is built from the angular constructor so
# the vector is exactly unit norm.
# -----------------------------------------------------------------------------

announce("")
announce("Hard weight truncation baseline ...")
hard_timing = @timed propagate_strategies(fill(hard_strategy, n_steps))
hard = hard_timing.value

@printf(
    "  done in %.3f s; hard weight-%d RMSE = %.6e\n",
    hard_timing.time,
    max_weight,
    curve_rmse(hard.curve),
)
flush(stdout)

saved["hard_curve"] = hard.curve
saved["hard_term_counts"] = hard.term_counts
saved["hard_rmse"] = curve_rmse(hard.curve)
checkpoint!()

announce("")
announce("Matched static SMFD reference ...")
matched_timing = @timed propagate_strategies(fill(smfd_strategy(physical_state), n_steps))
matched = matched_timing.value

@printf("  done in %.3f s; matched RMSE = %.6e\n", matched_timing.time, curve_rmse(matched.curve))
flush(stdout)

saved["matched_curve"] = matched.curve
saved["matched_term_counts"] = matched.term_counts
saved["matched_rmse"] = curve_rmse(matched.curve)
checkpoint!()

# -----------------------------------------------------------------------------
# 6. Summary
# -----------------------------------------------------------------------------

announce("")
announce("3 x 4 summary")

for (label, curve) in [
    ("greedy Cartesian", greedy.curve),
    ("matched static", matched.curve),
    ("hard weight truncation", hard.curve),
]
    @printf("  %-30s RMSE = %.6e\n", label, curve_rmse(curve))
end

announce("")
announce("results written to $(results_file)")
