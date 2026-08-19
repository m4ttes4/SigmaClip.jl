using BenchmarkTools
using Random

const BENCHDIR = dirname(@__FILE__)
pushfirst!(LOAD_PATH, dirname(BENCHDIR))

using SigmaClip

benchmarks = [
    "sigma_clip.jl",
    "fast_median.jl",
]

const SUITE = BenchmarkGroup()

foreach(benchmarks) do bm
    include(joinpath(BENCHDIR, bm))
end
