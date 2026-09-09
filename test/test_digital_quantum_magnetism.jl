using JLD2
using Test

include(joinpath(
    @__DIR__,
    "..",
    "examples",
    "digital_quantum_magnetism",
    "small_3x4_validation.jl",
))
include(joinpath(
    @__DIR__,
    "..",
    "examples",
    "digital_quantum_magnetism",
    "cluster_7x8_spv.jl",
))

const DQMSmall = DigitalQuantumMagnetismSmall
const DQMCluster = DigitalQuantumMagnetismCluster

@testset "Digital quantum magnetism examples" begin
    @testset "exact trajectory and cache" begin
        mktempdir() do temporary
            config = DQMSmall.DynamicsConfig(Lx=2, Ly=2, steps=2)
            calls = Ref(0)
            generator = function(current)
                calls[] += 1
                return DQMSmall.generate_exact_trajectory(current)
            end

            first = DQMSmall.load_or_generate_exact(
                config;
                cache_root=temporary,
                generator,
                verbose=false,
            )
            @test !first.cache_hit
            @test calls[] == 1
            @test isfile(first.path)
            @test size(first.states) == (16, 3)
            @test maximum(abs.(first.norms .- 1)) < 1e-12
            analytic =
                1 / 4 + (1 - 1 / 4) * cos(config.theta)^2
            @test first.z2[1] ≈ analytic atol=2e-13
            coefficient = DQMSmall.run_spv_method(
                config,
                :coefficient;
                threshold=4e-7,
                window=1,
            )
            @test real.(coefficient.z2) ≈ first.z2 atol=2e-12

            second = DQMSmall.load_or_generate_exact(
                config;
                cache_root=temporary,
                generator,
                verbose=false,
            )
            @test second.cache_hit
            @test calls[] == 1
            @test second.states == first.states

            changed = DQMSmall.DynamicsConfig(
                Lx=2,
                Ly=2,
                steps=2,
                h=nextfloat(config.h),
            )
            miss = DQMSmall.load_or_generate_exact(
                changed;
                cache_root=temporary,
                generator,
                verbose=false,
            )
            @test !miss.cache_hit
            @test calls[] == 2
            @test miss.path != first.path

            mismatched = DQMSmall.DynamicsConfig(
                Lx=2,
                Ly=2,
                steps=2,
                J=nextfloat(config.J),
            )
            mismatched_path =
                DQMSmall.cache_path(mismatched; cache_root=temporary)
            mkpath(dirname(mismatched_path))
            cp(first.path, mismatched_path; force=true)
            @test_throws ErrorException DQMSmall.load_or_generate_exact(
                mismatched;
                cache_root=temporary,
                generator,
                verbose=false,
            )

            open(first.path, "w") do io
                write(io, "not a JLD2 file")
            end
            @test_throws ErrorException DQMSmall.load_or_generate_exact(
                config;
                cache_root=temporary,
                generator,
                verbose=false,
            )
            forced = DQMSmall.load_or_generate_exact(
                config;
                cache_root=temporary,
                force=true,
                generator,
                verbose=false,
            )
            @test !forced.cache_hit
            @test calls[] == 3
            @test maximum(abs.(forced.norms .- 1)) < 1e-12

            incomplete_config =
                DQMSmall.DynamicsConfig(Lx=2, Ly=2, steps=1, dt=0.2)
            incomplete_path =
                DQMSmall.cache_path(incomplete_config; cache_root=temporary)
            mkpath(dirname(incomplete_path))
            open("$incomplete_path.tmp.interrupted", "w") do io
                write(io, "partial")
            end
            incomplete = DQMSmall.load_or_generate_exact(
                incomplete_config;
                cache_root=temporary,
                generator,
                verbose=false,
            )
            @test !incomplete.cache_hit
            @test calls[] == 4
        end
    end

    @testset "KetSum tilted-state phase" begin
        config = DQMSmall.DynamicsConfig(
            Lx=2,
            Ly=2,
            theta=0.41,
            phi=pi / 2,
            steps=0,
        )
        state = DQMSmall.initial_product_ketsum(config)
        reference = ProductBlochReference(
            4;
            theta=config.theta,
            phi=config.phi,
        )
        for pauli in (
            PauliBasis(Pauli(4; X=[1])),
            PauliBasis(Pauli(4; Y=[1])),
            PauliBasis(Pauli(4; Z=[1])),
        )
            @test expectation_value(PauliSum(pauli), state) ≈
                  expectation_value(pauli, reference) atol=2e-12
        end
    end

    @testset "cluster output smoke tests" begin
        mktempdir() do temporary
            for method in (:coefficient, :weight, :meanfield)
                output = joinpath(temporary, "$method.out")
                config = DQMCluster.ClusterConfig(
                    Lx=2,
                    Ly=2,
                    steps=2,
                    method=method,
                    max_weight=2,
                    output=output,
                )
                @test DQMCluster.run(config) == output
                lines = readlines(output)
                table = filter(line -> !startswith(line, '#'), lines)
                @test table[1] ==
                      "step,time,z2_real,z2_imag,nterms,operator_l2," *
                      "step_seconds,total_seconds"
                @test length(table) == 4
                for (step, line) in enumerate(table[2:end])
                    fields = split(line, ',')
                    @test length(fields) == 8
                    @test parse(Int, fields[1]) == step - 1
                    @test parse(Int, fields[5]) > 0
                    @test all(isfinite, parse.(Float64, fields[[2, 3, 4, 6, 7, 8]]))
                end
                @test_throws ArgumentError DQMCluster.run(config)
            end
        end
    end

    @testset "rows are flushed before file close" begin
        mktempdir() do temporary
            output = joinpath(temporary, "incremental.out")
            config = DQMCluster.ClusterConfig(
                Lx=2,
                Ly=2,
                steps=0,
                output=output,
            )
            open(output, "w") do io
                DQMCluster.write_header(io, config, 4)
                DQMCluster.write_row(io, 0, config, 1.0, 1, 1.0, 0.0, 0.0)
                visible = read(output, String)
                @test occursin(
                    "step,time,z2_real,z2_imag,nterms,operator_l2",
                    visible,
                )
                @test endswith(visible, ",1,1,0,0\n")
            end
        end
    end
end
