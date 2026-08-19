using Base.Order: Forward

SUITE["fast_median"] = BenchmarkGroup()
SUITE["fast_median"]["current"] = BenchmarkGroup()
SUITE["fast_median"]["partialsort"] = BenchmarkGroup()

"""
    fast_median_partialsort!(a)

Benchmark-only alternative to `SigmaClip.fast_median!`.  The public
`Base.partialsort!` path allocates scratch space for large vectors; this calls
Julia's in-place `PartialQuickSort` entry point directly instead.
"""
function fast_median_partialsort!(a::AbstractVector{T}) where {T}
    n = length(a)
    OUT = float(T)
    n == 0 && return zero(OUT)

    lo = firstindex(a) + (n - 1) ÷ 2
    hi = firstindex(a) + n ÷ 2

    Base.Sort.sort!(a, Base.Sort.PartialQuickSort(lo), Forward)
    low = OUT(a[lo])
    lo == hi && return low

    Base.Sort.sort!(a, Base.Sort.PartialQuickSort(hi), Forward)
    return (low + OUT(a[hi])) / 2
end

function _median_reference(a::AbstractVector{T}) where {T}
    n = length(a)
    OUT = float(T)
    n == 0 && return zero(OUT)

    sorted = sort!(copy(a))
    lo = firstindex(sorted) + (n - 1) ÷ 2
    hi = firstindex(sorted) + n ÷ 2
    return lo == hi ? OUT(sorted[lo]) : (OUT(sorted[lo]) + OUT(sorted[hi])) / 2
end

function _fast_median_data(case::Symbol, n::Int, T, seed::Int)
    rng = MersenneTwister(seed + n)
    data = randn(rng, T, n)

    if case === :normal
        return data
    elseif case === :outliers
        outlier_count = max(1, n ÷ 50)
        for i in 1:outlier_count
            idx = 1 + mod((i - 1) * 37, n)
            data[idx] = isodd(i) ? T(12 + 0.1i) : T(-12 - 0.1i)
        end
    elseif case === :sorted
        sort!(data)
    elseif case === :reverse_sorted
        sort!(data, rev = true)
    elseif case === :quantized
        for i in eachindex(data)
            data[i] = T(round(data[i]; digits = 1))
        end
    elseif case === :constant
        fill!(data, T(1))
    else
        error("unknown fast_median benchmark case: $case")
    end

    return data
end

const _FAST_MEDIAN_SIZES = (
    "window-9" => 9,
    "window-49" => 49,
    "sigma-128" => 128,
    "sigma-1024" => 1_024,
    "sigma-10000" => 10_000,
)
const _FAST_MEDIAN_CASES = (
    "normal" => :normal,
    "normal-with-outliers" => :outliers,
    "already-sorted" => :sorted,
    "reverse-sorted" => :reverse_sorted,
    "quantized" => :quantized,
    "constant" => :constant,
)
const _FAST_MEDIAN_TYPES = ("Float64" => Float64, "Float32" => Float32)

# Generate all inputs once.  Copies made by `setup` below are outside the
# measured region because both implementations reorder their input in place.
const _FAST_MEDIAN_DATA = let
    datasets = Dict{Tuple{DataType, Symbol, Int}, Any}()
    for (type_index, (_, T)) in enumerate(_FAST_MEDIAN_TYPES)
        for (case_index, (_, case)) in enumerate(_FAST_MEDIAN_CASES)
            for (_, n) in _FAST_MEDIAN_SIZES
                datasets[(T, case, n)] = _fast_median_data(
                    case, n, T, 0x51A6C11 + 10_000 * type_index + 100 * case_index
                )
            end
        end
    end
    datasets
end

function _check_fast_median_alternative()
    for T in (Float32, Float64), n in (0, 9, 128)
        data = n == 0 ? T[] : _FAST_MEDIAN_DATA[(T, :normal, n)]
        expected = _median_reference(data)

        current = copy(data)
        alternative = copy(data)
        @assert fast_median!(current) == expected
        @assert fast_median_partialsort!(alternative) == expected

        # The allocation check is deliberately after warm-up and excludes the
        # copy needed to restore the mutating input for the next invocation.
        alternative = copy(data)
        @assert @allocated(fast_median_partialsort!(alternative)) == 0
    end
    return nothing
end

_check_fast_median_alternative()

for (type_label, T) in _FAST_MEDIAN_TYPES
    SUITE["fast_median"]["current"][type_label] = BenchmarkGroup()
    SUITE["fast_median"]["partialsort"][type_label] = BenchmarkGroup()

    for (case_label, case) in _FAST_MEDIAN_CASES
        SUITE["fast_median"]["current"][type_label][case_label] = BenchmarkGroup()
        SUITE["fast_median"]["partialsort"][type_label][case_label] = BenchmarkGroup()

        for (size_label, n) in _FAST_MEDIAN_SIZES
            data = _FAST_MEDIAN_DATA[(T, case, n)]

            SUITE["fast_median"]["current"][type_label][case_label][size_label] =
                @benchmarkable fast_median!(x) setup = (x = copy($data)) evals = 1
            SUITE["fast_median"]["partialsort"][type_label][case_label][size_label] =
                @benchmarkable fast_median_partialsort!(x) setup = (x = copy($data)) evals = 1
        end
    end
end
