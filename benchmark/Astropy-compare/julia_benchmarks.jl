#!/usr/bin/env julia

using BenchmarkTools
using Random
using Statistics

pushfirst!(LOAD_PATH, dirname(dirname(@__DIR__)))
using SigmaClip

const OUTPUT = joinpath(@__DIR__, "julia_results.csv")
const SIGMA = 3.0
const MAXITERS = 5
const SEED = 51_461_393
const VECTOR_SIZES = (128, 1_024, 10_000, 100_000)
const MATRIX_SIZES = ((32, 32), (128, 128), (512, 512))

@views function make_data(shape, seed)
    data = randn(MersenneTwister(seed), shape...)
    values = vec(data)
    values[1:101:end] .+= 8.0
    values[51:173:end] .-= 8.0
    length(values) >= 8 || return data
    values[cld(length(values), 3)] = NaN
    values[cld(2length(values), 3)] = Inf
    return data
end

function measure(benchmark, samples)
    trial = run(benchmark; samples, evals = 1)
    return median(trial.times), minimum(trial.times), trial
end

function write_row(io, family, shape, n, mode, times, seed, samples)
    median_ns, minimum_ns, trial = times
    println(
        io,
        "SigmaClip.jl,", family, ",", shape, ",", n, ",", mode, ",",
        median_ns, ",", minimum_ns, ",", samples, ",", seed, ",",
        SIGMA, ",", SIGMA, ",", MAXITERS, ",median,mad_std,",
        trial.memory, ",", trial.allocs,
    )
end

open(OUTPUT, "w") do io
    println(io, "algorithm,family,shape,elements,mode,median_ns,min_ns,samples,seed,sigma_lower,sigma_upper,maxiters,center,spread,memory_bytes,allocations")

    for n in VECTOR_SIZES
        data = make_data((n,), SEED + n)
        workspace = SigmaClipWorkspace(Vector{Float64}(undef, n), Vector{Float64}(undef, n))
        samples = n <= 10_000 ? 50 : 15

        out_of_place = @benchmarkable sigma_clip($data)
        write_row(io, "vector", string(n), n, "out_of_place", measure(out_of_place, samples), SEED + n, samples)

        in_place = @benchmarkable sigma_clip!(x; workspace = $workspace) setup = (x = copy($data)) evals = 1
        write_row(io, "vector", string(n), n, "in_place", measure(in_place, samples), SEED + n, samples)
    end

    for shape in MATRIX_SIZES
        data = make_data(shape, SEED + prod(shape))
        workspace = SigmaClipWorkspace(Vector{Float64}(undef, length(data)), Vector{Float64}(undef, length(data)))
        samples = length(data) <= 10_000 ? 50 : 15

        out_of_place = @benchmarkable sigma_clip($data)
        write_row(io, "matrix", join(shape, "x"), length(data), "out_of_place", measure(out_of_place, samples), SEED + prod(shape), samples)

        in_place = @benchmarkable sigma_clip!(x; workspace = $workspace) setup = (x = copy($data)) evals = 1
        write_row(io, "matrix", join(shape, "x"), length(data), "in_place", measure(in_place, samples), SEED + prod(shape), samples)
    end
end

println("Wrote ", OUTPUT)
