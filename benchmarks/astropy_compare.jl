#!/usr/bin/env julia

# Manual comparison against astropy.stats.sigma_clip.
#
# This file is intentionally not included by test/runtests.jl. Run it from an
# environment that has PythonCall available and whose Python can import numpy
# and astropy:
#
#     julia --project=. benchmarks/astropy_compare.jl

using Printf
using Random
using SigmaClip
using Statistics

try
    @eval import PythonCall
catch err
    if err isa ArgumentError || err isa LoadError
        println(stderr, """
        PythonCall is required for this manual benchmark.

        Run the script from a Julia environment that already has PythonCall,
        numpy, astropy, and this local package available.

        This benchmark is not part of SigmaClip's test suite and PythonCall is
        not a runtime dependency of the package.
        """)
        exit(2)
    end
    rethrow()
end

function import_python_deps()
    try
        np = PythonCall.pyimport("numpy")
        astropy_stats = PythonCall.pyimport("astropy.stats")
        warnings = PythonCall.pyimport("warnings")
        warnings.filterwarnings("ignore"; message="Input data contains invalid values.*")
        return np, astropy_stats
    catch err
        println(stderr, """
        Python dependencies are required for this manual benchmark.

        Ensure PythonCall can import both numpy and astropy, for example:

            python3 -m pip install numpy astropy

        Original error:
        $err
        """)
        exit(2)
    end
end

const NP, ASTROPY_STATS = import_python_deps()

population_std(v) = std(v; corrected=false)

struct CompareCase
    name::String
    sigma_lower::Float64
    sigma_upper::Float64
    maxiter::Int
    spread::Symbol
end

function make_data(case::CompareCase, n::Int)
    seed = if case.name == "normal_with_outliers"
        11
    elseif case.name == "nan_inf_outliers"
        17
    elseif case.name == "asymmetric_thresholds"
        23
    elseif case.name == "single_iteration"
        29
    elseif case.name == "std_spread"
        31
    else
        error("unknown comparison case: $(case.name)")
    end
    rng = MersenneTwister(seed + n)
    data = randn(rng, n)

    if case.name == "normal_with_outliers"
        data[1:10:end] .+= 8.0
        data[5:17:end] .-= 8.0
    elseif case.name == "nan_inf_outliers"
        data[1:13:end] .= NaN
        data[3:29:end] .= Inf
        data[7:31:end] .= -Inf
        data[11:23:end] .+= 9.0
    elseif case.name == "asymmetric_thresholds"
        data[1:11:end] .+= 10.0
        data[2:37:end] .-= 5.0
    elseif case.name == "single_iteration"
        data[1:9:end] .+= 7.0
        data[4:21:end] .-= 7.0
    elseif case.name == "std_spread"
        data[1:10:end] .+= 8.0
        data[5:17:end] .-= 8.0
    else
        error("unknown comparison case: $(case.name)")
    end

    return data
end

function reps_for_size(n::Int)
    n <= 1_000 && return 100
    n <= 10_000 && return 30
    return 10
end

function julia_mask!(target::AbstractVector{Bool}, data::Vector{Float64},
    ws::SigmaClipWorkspace{Float64}, case::CompareCase)

    if case.spread == :mad_std
        sigma_clip_mask!(data, target;
            workspace=ws,
            center=fast_median!,
            spread=mad_std!,
            sigma_lower=case.sigma_lower,
            sigma_upper=case.sigma_upper,
            maxiter=case.maxiter)
    elseif case.spread == :std
            sigma_clip_mask!(data, target;
            workspace=ws,
            center=fast_median!,
            spread=population_std,
            sigma_lower=case.sigma_lower,
            sigma_upper=case.sigma_upper,
            maxiter=case.maxiter)
    else
        error("unsupported spread: $(case.spread)")
    end
end

function astropy_mask(py_data, case::CompareCase)
    stdfunc = case.spread == :mad_std ? "mad_std" : "std"
    result = ASTROPY_STATS.sigma_clip(py_data;
        sigma_lower=case.sigma_lower,
        sigma_upper=case.sigma_upper,
        maxiters=case.maxiter == -1 ? nothing : case.maxiter,
        cenfunc="median",
        stdfunc=stdfunc,
        masked=true,
        copy=true)

    mask = NP.asarray(result.mask, dtype=NP.bool_)
    return PythonCall.pyconvert(Vector{Bool}, mask)
end

function elapsed_seconds(f, reps::Int)
    t0 = time_ns()
    for _ in 1:reps
        f()
    end
    (time_ns() - t0) / 1e9 / reps
end

function compare_case(case::CompareCase, n::Int)
    data = make_data(case, n)
    py_data = NP.array(data, dtype=NP.float64, copy=true)

    target = falses(n)
    ws = SigmaClipWorkspace(Float64, n)

    julia_mask!(target, data, ws, case)
    astropy_valid = .!astropy_mask(py_data, case)
    mismatches = count(i -> target[i] != astropy_valid[i], eachindex(target))

    reps = reps_for_size(n)
    julia_time = elapsed_seconds(reps) do
        julia_mask!(target, data, ws, case)
    end
    astropy_time = elapsed_seconds(reps) do
        ASTROPY_STATS.sigma_clip(py_data;
            sigma_lower=case.sigma_lower,
            sigma_upper=case.sigma_upper,
            maxiters=case.maxiter == -1 ? nothing : case.maxiter,
            cenfunc="median",
            stdfunc=case.spread == :mad_std ? "mad_std" : "std",
            masked=true,
            copy=true)
    end

    return (; case=case.name, n, reps, julia_time, astropy_time,
        speedup=astropy_time / julia_time, mismatches)
end

function print_header()
    @printf("%-24s %9s %6s %12s %12s %9s %10s\n",
        "case", "n", "reps", "SigmaClip", "Astropy", "speedup", "diffs")
    @printf("%-24s %9s %6s %12s %12s %9s %10s\n",
        repeat("-", 24), repeat("-", 9), repeat("-", 6), repeat("-", 12),
        repeat("-", 12), repeat("-", 9), repeat("-", 10))
end

function print_row(row)
    @printf("%-24s %9d %6d %9.3f ms %9.3f ms %8.2fx %10d\n",
        row.case, row.n, row.reps,
        1e3 * row.julia_time, 1e3 * row.astropy_time,
        row.speedup, row.mismatches)
end

function main()
    cases = CompareCase[
        CompareCase("normal_with_outliers", 3.0, 3.0, 5, :mad_std),
        CompareCase("nan_inf_outliers", 3.0, 3.0, 5, :mad_std),
        CompareCase("asymmetric_thresholds", 2.0, 4.0, 5, :mad_std),
        CompareCase("single_iteration", 3.0, 3.0, 1, :mad_std),
        CompareCase("std_spread", 3.0, 3.0, 5, :std),
    ]
    sizes = (1_000, 10_000, 100_000)

    println("SigmaClip.jl vs astropy.stats.sigma_clip")
    println("Astropy mask is inverted before comparison because SigmaClip uses true=valid.")
    println()
    print_header()

    failed = false
    for case in cases
        for n in sizes
            row = compare_case(case, n)
            print_row(row)
            failed |= row.mismatches != 0
        end
    end

    if failed
        println()
        println(stderr, "Result mismatch detected against Astropy.")
        exit(1)
    end

    println()
    println("All SigmaClip masks matched Astropy.")
end

main()
