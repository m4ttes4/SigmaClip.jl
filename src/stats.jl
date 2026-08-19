const MAD_SF = 1.4826022185056018

@inline buffer_view(ws, n::Int) = @view ws.buf[1:n]
@inline auxiliary_view(ws, n::Int) = @view ws.aux[1:n]

# ─── Quickselect median ──────────────────────────────────────────────

function _kth_smallest!(a::AbstractVector{T}, k::Int) where {T}
    l = firstindex(a)
    r = lastindex(a)
    @inbounds while l < r
        pivot = a[k]
        i, j = l, r
        while true
            while a[i] < pivot
                i += 1
            end
            while pivot < a[j]
                j -= 1
            end
            if i <= j
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
            end
            i > j && break
        end
        j < k && (l = i)
        k < i && (r = j)
    end
    return a[k]
end

# TODO for type stability, fast median should return float(T)
"""
    fast_median!(a::AbstractVector) -> float(eltype(a))

Compute the median of `a` in O(n) average time using an in-place quickselect
(Wirth's algorithm).  **Modifies the ordering of `a`** but preserves all values.
Returns `zero(eltype(a))` for empty input.
When used as `center=fast_median!` in SigmaClip, it reorders only SigmaClip's
internal workspace, not the user's input array.

Allocation-free; roughly 2–3× faster than `Statistics.median` on random data.

See also: [`mad_std!`](@ref)
"""
function fast_median!(a::AbstractVector{T}) where {T}
    n = length(a)
    OUT = float(T)
    n == 0 && return zero(OUT)
    o = firstindex(a) - 1

    if iseven(n)
        lo = OUT(_kth_smallest!(a, o + n ÷ 2))
        hi = OUT(_kth_smallest!(a, o + n ÷ 2 + 1))
        return (lo + hi) / 2
    else
        return OUT(_kth_smallest!(a, o + (n + 1) ÷ 2))
    end
end

"""
    mad_std!(a::AbstractVector)

Compute the median absolute deviation of `a` in place, scaled to match the
standard deviation of a normal distribution.
"""
mad_std!(a::AbstractVector{T}) where {T} = mad_std!(a, Vector{float(T)}(undef, length(a)))

function mad_std!(a::AbstractVector{T}, aux::AbstractVector{B}) where {T <: Number, B}
    n = length(a)
    OUT = float(T)
    n == 0 && return zero(OUT)

    m = fast_median!(a)
    @inbounds for i in eachindex(a)
        aux[i] = abs(a[i] - m)
    end
    res = fast_median!(aux) * MAD_SF
    return convert(OUT, res)
end

function mad_std!(a::AbstractVector{T}, aux::AbstractVector{B}, m) where {T <: Number, B}
    n = length(a)
    OUT = float(T)
    n == 0 && return zero(OUT)

    @inbounds for i in eachindex(a)
        aux[i] = abs(a[i] - m)
    end
    res = fast_median!(aux) * MAD_SF
    return convert(OUT, res)
end

@inline function compute_stats(
        ::typeof(fast_median!),
        ::typeof(mad_std!),
        n::Int,
        ws::WS,
    ) where {WS}
    data = buffer_view(ws, n)
    m = fast_median!(data)
    return m, mad_std!(data, auxiliary_view(ws, n), m)
end

@inline function compute_stats(
        center_f::C,
        ::typeof(mad_std!),
        n::Int,
        ws::WS,
    ) where {C, WS}
    data = buffer_view(ws, n)
    return center_f(data), mad_std!(data, auxiliary_view(ws, n))
end

@inline function compute_stats(
        center_f::C,
        spread_f::S,
        n::Int,
        ws::WS,
    ) where {C, S, WS}
    data = buffer_view(ws, n)
    return center_f(data), spread_f(data)
end

@inline need_aux(::typeof(mad_std!)) = true
@inline need_aux(_) = false
