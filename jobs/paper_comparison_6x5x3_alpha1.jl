# =============================================================================
# Paper comparison against arXiv:2609.12840 (Pérez, Wójtowicz, Plenio) on the
# 3D open transverse-field Ising lattice, packaged as a batch job.
#
# Submit from whatever directory holds this file, next to the submit script:
#
#     sbatch submit.sh paper_comparison_6x5x3_alpha1.jl
#
# The file does not need to live inside the package clone. The environment comes
# from $JULIAENV, which the submit script exports and which must already be
# instantiated; Manifest.toml is not in the repository. Pass a bare filename:
# the wrapper copies $INFILE into $TMPDIR and runs it from there, so a
# path-qualified argument would not resolve.
#
# Results are written under the directory you submit from, not $TMPDIR, so they
# survive a walltime kill.
#
# WHAT THIS COMPUTES
#
# Their state-adapted projection pi_m^{sigma_0} (their Eq. 4 / SI Eq. 34) is our
# MeanFieldTruncation with a product Bloch reference; with the reference set to
# their sigma_0 = |up><up|^{tensor N} it is alpha = 1 in our parametrization.
# Their Hilbert-Schmidt baseline is plain weight truncation, alpha = 0. Both
# arms are propagated here at fixed m, differing ONLY in the projection:
#
#     covariance : MeanFieldTruncation(m, ProductBlochReference(N, 0, 0, 1))
#     HS         : WeightTruncation(m)
#
# No coefficient truncation, matching them -- they truncate nothing during
# propagation, and their eps_nnz = 1e-12 is a counting threshold only.
#
# Their propagator is the continuous flow dO/dt = L_m O solved in an adaptive
# Krylov chart; ours is a second-order Trotter circuit (X/2 -> ZZ -> X/2). The
# dt scan is what bridges them: as dt falls, our deviation from their published
# trajectory must fall at the order of the splitting. At N = 90, m = 1, 2 it
# does, cleanly, by a factor of 4 per halving.
#
# This job exists for the m = 3 arm at N = 90, which is hours rather than
# seconds. Everything else here runs in about a minute and is included so one
# file reproduces the whole benchmark.
#
# COST (measured on one core of a Ryzen 9 5900X, 6 x 5 x 3, finest dt)
#
#     m = 1   both arms                         < 1 s
#     m = 2   HS < 1 s, covariance              ~11 s
#     m = 3   HS                                ~30 s
#     m = 3   covariance, 80 steps to tJ = 1.2  1425 s  (~24 min)
#
# The covariance m = 3 growth, measured every ten steps:
#
#     tJ      0.15   0.30   0.45   0.60   0.75   0.90   1.05   1.20
#     terms   3440  11240  24531  44165  69020  98431 130063 164387
#     s/10    44.9  224.2  205.4  230.8  195.7  178.8  164.7  180.0
#
# Within a single run the per-step cost is FLAT -- mildly falling, even -- while
# the operator grows by a factor of 48. That does NOT mean the dt values cost in
# proportion to their step counts. Measured across the grid, covariance m = 3:
#
#     dt        0.120   0.060   0.030   0.015
#     steps        10      20      40      80
#     wall [s]   24.8   149.4   400.5   901.3
#
# a factor of 36 across an 8x range of step counts. The reason is that the
# retained set is fixed by the light cone and the weight cap rather than by the
# number of rotations -- all four runs end at ~164k strings, within 2.1% -- so a
# fine-dt run simply spends many more steps carrying a large operator.
#
# Whole 24-trajectory sweep at the defaults: 1544 s (0.43 h) measured, essentially
# all of it covariance m = 3. Peak RSS 0.68 GiB -- that is the Julia runtime; the
# operator itself is 40 bytes per term, so 164387 strings is 6.6 MB. Memory is
# not a constraint here at any m. Suggested request:
#
#     #SBATCH --time=04:00:00
#     #SBATCH --mem=8G
#     #SBATCH --cpus-per-task=1
#
# Partial work is never lost regardless: a long trajectory is checkpointed
# mid-run (every PAPER_PROGRESS_EVERY steps, but no more often than
# PAPER_CHECKPOINT_SECONDS apart), so a kill at step 60 of 80 still leaves 60
# steps on disk under the run's *_partial_curve key. Fast runs write nothing
# mid-run and stay silent.
#
# ENVIRONMENT OVERRIDES (all optional)
#
#     PAPER_LATTICE          "6x5x3" (default) or "3x3x2"
#     PAPER_MAX_WEIGHTS      "1,2,3" (default)
#     PAPER_DT_GRID          comma-separated; default depends on the lattice
#     PAPER_GEOMETRIES       "hs,covariance" (default)
#     PAPER_PROGRESS_EVERY   "10" (default) -- step stride at which a long run
#                            may report progress and checkpoint
#     PAPER_CHECKPOINT_SECONDS
#                            "120" (default) -- minimum seconds between those
#                            mid-run checkpoints, so fast runs stay silent
#     PAPER_DATA_DIR         directory holding the published CSVs; if it cannot
#                            be found the job still runs and simply skips the
#                            comparison, leaving it to the notebook
#     PAPER_RESULTS_DIR      where results/ is written; defaults to
#                            $SLURM_SUBMIT_DIR, then this file's directory
#
# Nothing is plotted. Everything the plots need is written to
#     results/paper_comparison_<lattice>_alpha1.jld2
# which is rewritten after each trajectory so a walltime kill keeps finished
# work.
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

lattice_name = get(ENV, "PAPER_LATTICE", "6x5x3")

# Each lattice carries the evaluation grid and comparison window of the
# corresponding published run. At 6 x 5 x 3 the release stops COVARIANCE m = 3
# at tJ = 1.2 (81 points) while every other run, HS m = 3 included, reaches 3.0
# (201 points). So the window depends on the geometry as well as the body order:
# keying it on m alone truncates HS m = 3 to 40% of a window that is fully
# comparable.
if lattice_name == "6x5x3"
    Lx, Ly, Lz = 6, 5, 3
    paper_dt = 0.015
    default_dt_grid = [0.12, 0.06, 0.03, 0.015]
    window(geometry, m) = (geometry == :covariance && m == 3) ? 1.200 : 3.000
    paper_csv_name = "tfim_6x5x3_N90_J1_hx075_site0_all_data_side_by_side.csv"
    paper_time_column = "tJ"
elseif lattice_name == "3x3x2"
    Lx, Ly, Lz = 3, 3, 2
    paper_dt = 0.005
    default_dt_grid = [0.135, 0.045, 0.015, 0.005]
    window(geometry, m) = 1.215
    paper_csv_name = "tfim_3x3x2_J1_hx075_mu1_site0_all_data_side_by_side.csv"
    paper_time_column = "exact_tJ"
else
    error("PAPER_LATTICE must be \"6x5x3\" or \"3x3x2\", got \"$(lattice_name)\"")
end

N = Lx * Ly * Lz

# H = J sum_<ij> Z_i Z_j + h sum_i X_i, their Eq. (16). Fixed by the paper.
J = 1.0
h = 0.75

parse_list(name, fallback, convert) =
    haskey(ENV, name) ?
    [convert(strip(piece)) for piece in split(ENV[name], ',') if !isempty(strip(piece))] :
    fallback

max_weights = parse_list("PAPER_MAX_WEIGHTS", [1, 2, 3], piece -> parse(Int, piece))
dt_grid = parse_list("PAPER_DT_GRID", default_dt_grid, piece -> parse(Float64, piece))
geometries = Symbol.(parse_list("PAPER_GEOMETRIES", ["hs", "covariance"], String))
progress_every = parse(Int, get(ENV, "PAPER_PROGRESS_EVERY", "10"))
checkpoint_seconds = parse(Float64, get(ENV, "PAPER_CHECKPOINT_SECONDS", "120"))

for geometry in geometries
    geometry in (:covariance, :hs) ||
        error("PAPER_GEOMETRIES entries must be \"covariance\" or \"hs\", got \"$(geometry)\"")
end

# Trotter steps must land on the published grid, so dt is a multiple of the grid
# spacing and divides every window. Checked rather than assumed: an off-grid dt
# would silently compare against the wrong time point.
for dt in dt_grid, m in max_weights, geometry in geometries
    isapprox(window(geometry, m) / dt, round(window(geometry, m) / dt); atol = 1e-9) ||
        error("dt = $(dt) does not divide the $(geometry) m = $(m) window $(window(geometry, m))")
    isapprox(dt / paper_dt, round(dt / paper_dt); atol = 1e-9) ||
        error("dt = $(dt) is not a multiple of the published grid step $(paper_dt)")
end

site(x, y, z) = x + (y - 1) * Lx + (z - 1) * Lx * Ly

# Open boundaries. The observable sits on a corner, which only exists with open
# boundaries; the paper never states this and the release does not record it,
# but it was settled empirically at 3 x 3 x 2 -- periodic misses their exact
# curve by 25.8x while open reproduces it.
bonds = Tuple{Int,Int}[]

for z in 1:Lz, y in 1:Ly, x in 1:Lx
    current_site = site(x, y, z)

    x < Lx && push!(bonds, minmax(current_site, site(x + 1, y, z)))
    y < Ly && push!(bonds, minmax(current_site, site(x, y + 1, z)))
    z < Lz && push!(bonds, minmax(current_site, site(x, y, z + 1)))
end

sort!(bonds)

@assert length(bonds) == (Lx - 1) * Ly * Lz + Lx * (Ly - 1) * Lz + Lx * Ly * (Lz - 1)
@assert length(unique(bonds)) == length(bonds)

# Second-order Trotter for exp(-i H dt): X/2 -> ZZ -> X/2. Under the package
# convention O -> exp(i a/2 G) O exp(-i a/2 G), a term c*G propagated for time
# dt carries angle a = 2*c*dt; the transverse field is split into two halves.
function build_circuit(dt)
    generators = PauliBasis{N}[]
    angles = Float64[]

    for site_index in 1:N
        push!(generators, PauliBasis(Pauli(N; X = [site_index])))
        push!(angles, h * dt)
    end

    for (first_site, second_site) in bonds
        push!(generators, PauliBasis(Pauli(N; Z = [first_site, second_site])))
        push!(angles, 2 * J * dt)
    end

    for site_index in 1:N
        push!(generators, PauliBasis(Pauli(N; X = [site_index])))
        push!(angles, h * dt)
    end

    return generators, angles
end

# sigma_0 is simultaneously the initial state and the truncation reference --
# the paper's defining choice, and the whole content of "state-adapted".
sigma0 = ProductBlochReference(N, 0.0, 0.0, 1.0)

corner = site(1, 1, 1)
@assert corner == 1

z0_operator = PauliSum(N, Float64)
z0_operator[PauliBasis(Pauli(N; Z = [corner]))] = 1.0
observable = SparsePauliVector(z0_operator; T = Float64)

@assert isapprox(real(expectation_value(observable, sigma0)), 1.0; atol = 1e-12)

# At N = 90 the key word widens from UInt64 to UInt128. Assert it: a silent
# fallback to a narrower word would alias distant sites onto each other.
key_word = typeof(observable).parameters[2]
@assert N <= 64 ? sizeof(key_word) == 8 : sizeof(key_word) == 16

# Gate G0: the Bloch-reference and computational-basis-Ket code paths of the
# projection are different implementations of the same map, and this reference
# is the one that makes them overlap. Cheap, and it fails loudly if either
# path drifts.
let
    probe = PauliBasis(Pauli(N; Z = [2, 5, 9], X = [3]))
    bloch_projection = mean_field_factorize(probe, 1.0, sigma0, 2)
    ket_projection = mean_field_factorize(probe, 1.0, Ket(zeros(Int, N)), 2)

    @assert Set(keys(bloch_projection)) == Set(keys(ket_projection))
    @assert all(
        isapprox(bloch_projection[p], ket_projection[p]; atol = 1e-12)
        for p in keys(bloch_projection)
    )
end

announce("lattice = $(Lx) x $(Ly) x $(Lz), N = $(N), open bonds = $(length(bonds))")
announce("J = $(J), h = $(h), observable = Z_$(corner) (corner), reference = Bloch (0, 0, 1)")
announce("rotations per Trotter step = $(length(build_circuit(paper_dt)[1]))")
announce("published grid step = $(paper_dt)")
announce("body orders = $(max_weights), geometries = $(geometries)")
announce("dt grid = $(dt_grid)")
for geometry in geometries, m in max_weights
    @printf(
        "  %-4s m = %d: window tJ <= %.3f, steps at finest dt = %d\n",
        geometry == :covariance ? "cov" : "HS",
        m, window(geometry, m), round(Int, window(geometry, m) / dt_grid[end]),
    )
end
announce("gate G0 (Bloch vs Ket projection paths agree): pass")
flush(stdout)

# -----------------------------------------------------------------------------
# 2. Results file and checkpointing
# -----------------------------------------------------------------------------

results_root = get(ENV, "PAPER_RESULTS_DIR", get(ENV, "SLURM_SUBMIT_DIR", @__DIR__))
results_file = normpath(joinpath(
    results_root, "results", "paper_comparison_$(lattice_name)_alpha1.jld2",
))
mkpath(dirname(results_file))

announce("results root = $(results_root)")

saved = Dict{String,Any}(
    "parameters" => (
        lattice = lattice_name,
        Lx = Lx,
        Ly = Ly,
        Lz = Lz,
        N = N,
        J = J,
        h = h,
        boundary = "open",
        paper_dt = paper_dt,
        dt_grid = dt_grid,
        max_weights = max_weights,
        geometries = String.(geometries),
        coefficient_cutoff = nothing,
        nnz_threshold = 1e-12,
        reference = "ProductBlochReference(N, 0, 0, 1) = alpha 1 = their sigma_0",
        propagator = "second-order Trotter, X/2 -> ZZ -> X/2",
        paper = "arXiv:2609.12840v1",
    ),
    "bonds" => bonds,
    "windows" => Dict(
        (String(geometry), m) => window(geometry, m)
        for geometry in geometries, m in max_weights
    ),
)

last_checkpoint_time = Ref(time())

function checkpoint!(; quiet = false)
    # Write to a sibling then move: a walltime kill landing inside jldopen would
    # otherwise leave a truncated file where a complete earlier one used to be.
    staging_file = results_file * ".partial"

    jldopen(staging_file, "w") do file
        for (key, value) in saved
            file[key] = value
        end
    end

    mv(staging_file, results_file; force = true)
    last_checkpoint_time[] = time()
    quiet || announce("  checkpoint -> $(results_file)")
end

# JLD2 group separator is "/", so a dt like 0.015 is written with the decimal
# point replaced rather than nested into a group by accident.
function run_key(geometry, m, dt)
    geometry_tag = geometry == :covariance ? "cov" : "hs"
    dt_tag = replace(@sprintf("%.4f", dt), "." => "p")
    return @sprintf("%s_m%d_dt%s", geometry_tag, m, dt_tag)
end

checkpoint!()

# -----------------------------------------------------------------------------
# 3. Published data, if it can be found
# -----------------------------------------------------------------------------
#
# The comparison is cheap and can equally be done in the notebook afterwards, so
# a missing CSV is a warning and not an error -- the expensive part of this job
# is producing our own trajectories.

struct PaperTable
    header::Vector{String}
    rows::Vector{Vector{String}}
end

function load_paper_table(path)
    lines = readlines(path)
    header = String.(split(lines[1], ','))
    rows = Vector{Vector{String}}()

    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = String.(split(line, ','))

        # Trailing empty fields can be dropped by the writer; pad so every row
        # can be indexed by column position without a bounds check.
        while length(fields) < length(header)
            push!(fields, "")
        end

        push!(rows, fields)
    end

    return PaperTable(header, rows)
end

function paper_column(table::PaperTable, name)
    index = findfirst(==(name), table.header)
    index === nothing && return nothing
    return [
        isempty(strip(row[index])) ? NaN : parse(Float64, strip(row[index]))
        for row in table.rows
    ]
end

function locate_paper_csv()
    candidates = String[]

    haskey(ENV, "PAPER_DATA_DIR") && push!(candidates, ENV["PAPER_DATA_DIR"])
    push!(candidates, joinpath(results_root, "paper_exact"))
    push!(candidates, joinpath(@__DIR__, "paper_exact"))
    push!(candidates, normpath(joinpath(@__DIR__, "..", "code", "paper_recreate", "paper_exact")))

    for directory in candidates
        path = joinpath(directory, paper_csv_name)
        isfile(path) && return path
    end

    return nothing
end

paper_csv = locate_paper_csv()
paper_tJ = nothing
theirs = Dict{Tuple{Symbol,Int},Vector{Float64}}()
theirs_nnz = Dict{Tuple{Symbol,Int},Vector{Float64}}()

if paper_csv === nothing
    @warn "published CSV not found; trajectories will be saved without comparison" paper_csv_name
else
    table = load_paper_table(paper_csv)
    paper_tJ = paper_column(table, paper_time_column)

    for geometry in (:covariance, :hs), m in 1:3
        key = @sprintf("%s_m%02d", geometry == :covariance ? "cov" : "hs", m)
        curve = paper_column(table, key * "_z0_expectation")
        curve === nothing && continue
        theirs[(geometry, m)] = curve

        # Their representation size. At 6 x 5 x 3 there is no operator_nnz
        # column, and the tempting substitute is the wrong one: basis_b00_nnz is
        # the Krylov chart SEED, held constant between reconstructions, which
        # understates the represented operator by 2.8x at m = 3. The comparable
        # quantity is max_alpha nnz(b_alpha) -- confirmed by covariance m = 2,
        # where our 2129 strings meet their 2132.
        counts = paper_column(table, key * "_active_basis_nnz_max")
        counts === nothing && (counts = paper_column(table, key * "_operator_nnz"))
        counts === nothing || (theirs_nnz[(geometry, m)] = counts)
    end

    announce("published data = $(paper_csv)")
    announce("published runs loaded = $(length(theirs))")

    saved["paper_csv"] = paper_csv
    saved["paper_tJ"] = paper_tJ
end

flush(stdout)

# -----------------------------------------------------------------------------
# 4. Propagation
# -----------------------------------------------------------------------------

# Term count under their eps_nnz = 1e-12 counting threshold.
function nnz_count(operator; threshold = 1e-12)
    total = 0

    for (_, coefficient) in operator
        abs(coefficient) >= threshold && (total += 1)
    end

    return total
end

covariance_strategy(m) = MeanFieldTruncation(m, sigma0)
hs_strategy(m) = WeightTruncation(m)

strategy_for(geometry, m) =
    geometry == :covariance ? covariance_strategy(m) : hs_strategy(m)

# `on_progress` is what makes the multi-hour covariance m = 3 arm survivable: it
# is handed the partially filled curve every `progress_every` steps and writes
# it out, so a walltime kill costs at most that many steps.
function propagate(geometry, m, dt; on_progress = nothing)
    generators, angles = build_circuit(dt)
    strategy = strategy_for(geometry, m)

    n_steps = round(Int, window(geometry, m) / dt)
    stride = round(Int, dt / paper_dt)

    operator = deepcopy(observable)
    curve = Vector{Float64}(undef, n_steps + 1)
    counts = Vector{Int}(undef, n_steps + 1)
    step_seconds = Vector{Float64}(undef, n_steps + 1)
    grid_index = [1 + stride * step for step in 0:n_steps]

    curve[1] = real(expectation_value(operator, sigma0))
    counts[1] = nnz_count(operator)
    step_seconds[1] = 0.0

    for step in 1:n_steps
        step_seconds[step + 1] = @elapsed evolve!(
            operator, generators, angles; window = 1, truncation = strategy,
        )

        curve[step + 1] = real(expectation_value(operator, sigma0))
        counts[step + 1] = nnz_count(operator)

        if on_progress !== nothing && step % progress_every == 0
            on_progress(step, n_steps, curve, counts, step_seconds, grid_index)
        end
    end

    return (
        grid_index = grid_index,
        curve = curve,
        counts = counts,
        step_seconds = step_seconds,
        peak_terms = maximum(counts),
        final_terms = counts[end],
        range_violation = maximum(abs, curve) > 1 + 1e-12,
    )
end

# Deviation from their trajectory on the grid points both runs cover. There is
# no exact curve at 6 x 5 x 3 -- full Hilbert-space propagation at N = 90 is not
# available to anyone -- so this is the primary accuracy statement at that size,
# and the word "error" is deliberately avoided.
function deviation(result, geometry, m)
    haskey(theirs, (geometry, m)) || return nothing

    reference = theirs[(geometry, m)][result.grid_index]
    shared = .!isnan.(reference)
    any(shared) || return nothing

    return abs.(result.curve[shared] .- reference[shared])
end

function paper_peak_terms(result, geometry, m)
    haskey(theirs_nnz, (geometry, m)) || return nothing

    reference = theirs_nnz[(geometry, m)][result.grid_index]
    shared = .!isnan.(reference)
    any(shared) || return nothing

    return maximum(reference[shared])
end

# -----------------------------------------------------------------------------
# 5. Run
# -----------------------------------------------------------------------------
#
# Ordered cheapest first -- body order outermost, then dt coarse to fine, HS
# before covariance -- so that a walltime kill loses only the single most
# expensive trajectory, and every cheap result is already on disk.

announce("")
announce("Propagating $(length(max_weights) * length(dt_grid) * length(geometries)) trajectories ...")

summary_rows = NamedTuple[]

for m in sort(max_weights), dt in sort(dt_grid; rev = true), geometry in geometries
    key = run_key(geometry, m, dt)
    n_steps = round(Int, window(geometry, m) / dt)

    @printf(
        "m = %d, dt = %.4f (%d steps), %s\n",
        m, dt, n_steps, string(geometry),
    )
    flush(stdout)

    # Progress reporting is gated on elapsed time, not on step count: a run that
    # finishes in a second should say nothing, while the multi-hour covariance
    # m = 3 arm should leave a trail and a recoverable partial curve.
    function on_progress(step, total, curve, counts, step_seconds, grid_index)
        time() - last_checkpoint_time[] < checkpoint_seconds && return

        saved[key * "_partial_step"] = step
        saved[key * "_partial_curve"] = curve[1:(step + 1)]
        saved[key * "_partial_counts"] = counts[1:(step + 1)]
        saved[key * "_partial_step_seconds"] = step_seconds[1:(step + 1)]
        saved[key * "_grid_index"] = grid_index

        @printf(
            "    step %d/%d  tJ = %.3f  terms = %d  last step %.2f s  peak RSS %.2f GiB\n",
            step, total, step * dt, counts[step + 1], step_seconds[step + 1],
            Sys.maxrss() / 2^30,
        )
        flush(stdout)

        checkpoint!(quiet = true)
    end

    timing = @timed propagate(geometry, m, dt; on_progress = on_progress)
    result = timing.value

    differences = deviation(result, geometry, m)
    their_peak = paper_peak_terms(result, geometry, m)

    saved[key * "_curve"] = result.curve
    saved[key * "_counts"] = result.counts
    saved[key * "_step_seconds"] = result.step_seconds
    saved[key * "_grid_index"] = result.grid_index
    saved[key * "_peak_terms"] = result.peak_terms
    saved[key * "_final_terms"] = result.final_terms
    saved[key * "_range_violation"] = result.range_violation
    saved[key * "_wall_seconds"] = timing.time

    # Sys.maxrss is the high-water mark for the whole process, so it only ever
    # rises; recording it per run gives the envelope the job actually needed,
    # which is what the next --mem request should be based on.
    saved[key * "_peak_rss_bytes"] = Sys.maxrss()

    differences === nothing || (saved[key * "_deviation"] = differences)

    # The partial copies have served their purpose once the full run is stored.
    for suffix in ("_partial_step", "_partial_curve", "_partial_counts", "_partial_step_seconds")
        delete!(saved, key * suffix)
    end

    push!(summary_rows, (
        geometry = geometry,
        m = m,
        dt = dt,
        max_deviation = differences === nothing ? NaN : maximum(differences),
        rms_deviation = differences === nothing ? NaN : sqrt(mean(abs2, differences)),
        peak_terms = result.peak_terms,
        paper_peak_terms = their_peak === nothing ? NaN : their_peak,
        wall_seconds = timing.time,
        peak_rss_bytes = Sys.maxrss(),
        range_violation = result.range_violation,
    ))

    saved["summary"] = summary_rows
    checkpoint!()

    if differences === nothing
        @printf("peak terms = %d  (%.1f s)\n", result.peak_terms, timing.time)
    else
        @printf(
            "max|ours-paper| = %.4e  peak terms = %d  (%.1f s)\n",
            maximum(differences), result.peak_terms, timing.time,
        )
    end
    flush(stdout)
end

# -----------------------------------------------------------------------------
# 6. Summary
# -----------------------------------------------------------------------------

announce("")
announce("$(lattice_name) summary")
announce("")

println(
    rpad("run", 18), rpad("dt", 9), rpad("max dev", 13), rpad("rms dev", 13),
    rpad("peak terms", 12), rpad("paper peak", 12), rpad("ratio", 9),
    rpad("wall [s]", 11), "peak RSS [GiB]",
)
println("-"^116)

for row in summary_rows
    ratio = isnan(row.paper_peak_terms) ? NaN : row.peak_terms / row.paper_peak_terms

    println(
        rpad(string(row.geometry == :covariance ? "cov" : "HS", " m=", row.m), 18),
        rpad(@sprintf("%.4f", row.dt), 9),
        rpad(isnan(row.max_deviation) ? "-" : @sprintf("%.4e", row.max_deviation), 13),
        rpad(isnan(row.rms_deviation) ? "-" : @sprintf("%.4e", row.rms_deviation), 13),
        rpad(@sprintf("%d", row.peak_terms), 12),
        rpad(isnan(row.paper_peak_terms) ? "-" : @sprintf("%d", round(Int, row.paper_peak_terms)), 12),
        rpad(isnan(ratio) ? "-" : @sprintf("%.2fx", ratio), 9),
        rpad(@sprintf("%.1f", row.wall_seconds), 11),
        @sprintf("%.2f", row.peak_rss_bytes / 2^30),
    )
end

announce("")
@printf(
    "total wall = %.1f s (%.2f h), process peak RSS = %.2f GiB\n",
    sum(row -> row.wall_seconds, summary_rows),
    sum(row -> row.wall_seconds, summary_rows) / 3600,
    Sys.maxrss() / 2^30,
)

violations = filter(row -> row.range_violation, summary_rows)

announce("")
if isempty(violations)
    announce("expectation-range check |<Z0>| <= 1: all trajectories pass")
else
    announce("expectation-range VIOLATIONS |<Z0>| > 1:")
    for row in violations
        @printf(
            "  %s m=%d dt=%.4f\n",
            row.geometry == :covariance ? "cov" : "HS", row.m, row.dt,
        )
    end
end

# The dt scan is the load-bearing evidence: if our projection is their
# projection and only the integrator differs, the deviation must fall at the
# order of the splitting. Second-order Trotter gives a factor of 4 per halving.
if !isempty(theirs) && length(dt_grid) > 1
    announce("")
    announce("dt convergence: observed order p from log(dev ratio)/log(dt ratio)")
    announce("(the Trotter splitting is second order, so p -> 2 is the target)")

    for m in sort(max_weights), geometry in geometries
        ordered = sort(
            filter(row -> row.m == m && row.geometry == geometry, summary_rows);
            by = row -> -row.dt,
        )

        length(ordered) > 1 || continue
        any(row -> isnan(row.max_deviation), ordered) && continue

        # The dt grid need not be a halving -- at 3 x 3 x 2 it steps by three --
        # so the order is read off directly rather than compared against 4.
        orders = [
            log(ordered[i].max_deviation / ordered[i + 1].max_deviation) /
            log(ordered[i].dt / ordered[i + 1].dt)
            for i in 1:(length(ordered) - 1)
        ]

        @printf(
            "  %-4s m=%d   p = %s\n",
            geometry == :covariance ? "cov" : "HS", m,
            join([@sprintf("%.2f", p) for p in orders], "  "),
        )
    end
end

checkpoint!()

announce("")
announce("results written to $(results_file)")
