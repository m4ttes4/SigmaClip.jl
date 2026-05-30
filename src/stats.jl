

const MAD_SF = 1.4826022185056018


@inline @views function collect_buf_from_ws(ws::WS, n::Int)where {WS}
    data = workspace_buffer(ws)
    res = data[1:n]
    return res
end
@inline @views function collect_aux_from_ws(ws::WS, n::Int) where {WS}
    data = workspace_auxbuffer(ws)
    res = data[1:n]
    return res
end



# ─── Quickselect median ───────────────────────────────────────────────────────

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
    mad_std!(a::AbstractVector) -> float(eltype(T))

Compute the Median Absolute Deviation of `a` in-place, scaled by 1.4826 to
match the standard deviation of a normal distribution.

When passed as `spread=mad_std!`, SigmaClip selects the built-in robust
dispersion estimator. When combined with `center=fast_median!` (default), the
median is computed once and shared with the MAD calculation.
When used inside SigmaClip with a workspace, `mad_std!` uses the workspace
auxiliary buffer instead of allocating its own.

See also: [`fast_median!`](@ref)
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

#Helper for fast-path with precomputed median
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


"""
    SigmaClip.statistic(f, ws, n::Int)

Compute a scalar statistic for the first `n` compacted values in `ws`.
The default method passes a mutable view of `workspace_buffer(ws)[1:n]` to
reducers that accept an `AbstractVector`.

Custom workspace-aware reducers can extend this method for their reducer and
workspace type:

```julia
SigmaClip.statistic(::MyStat, ws::MyWorkspace, n::Int)
```

Inside the method, use [`SigmaClip.workspace_buffer`](@ref) and
[`SigmaClip.workspace_auxbuffer`](@ref) to access the compacted data and
auxiliary scratch space. Custom methods should use only the first `n` elements
of each buffer.

Statistics may reorder `workspace_buffer(ws)[1:n]`, but must preserve those
values because SigmaClip compacts that buffer after computing the statistics.
The auxiliary buffer may be used freely as scratch.
"""
@inline function statistic(f::F, ws::WS, n::Int) where {F, WS}
    data = collect_buf_from_ws(ws, n)
    return f(data)
end


#TODO JETS suggest possible type inference problem with eltype(aux)
@inline function statistic(::typeof(mad_std!), ws::WS, n::Int) where {WS}
    data = collect_buf_from_ws(ws, n)
    aux = collect_aux_from_ws(ws, n)


    return mad_std!(data, aux)
end


# ─── _compute_stats ───────────────────────────────────────────────────────────
#
# Returns (centre, dispersion) for the n values packed in buf[1:n].
#
# Contract:
#   • may reorder buf[1:n] (quickselect is partial, values are preserved)
#   • must NOT overwrite buf[1:n] with unrelated data — compaction reads it next
#   • may freely read and write aux[1:n]
#
# Specialisations, resolved at compile time:
#
#   (fast_median!, mad_std!) — fully specialised; median shared between centre
#                              and MAD; two quickselects, one deviation loop
#   (fast_median!, generic)  — fast centre, workspace-aware dispersion
#   (generic,     generic)   — both reducers use the statistic protocol


# Specialisation 1 — (FastMedian, MADStd)
#
# After fast_median!(buf[1:n]) the buffer is reordered but all n values remain.
# We compute |buf[i] − m| into aux[1:n] (leaving buf intact), then run a second
# quickselect on aux to get the MAD.
#
@inline function _compute_stats(
        ::typeof(fast_median!),
        ::typeof(mad_std!),
        n::Int,
        ws::WS 
    ) where {WS}

    data = collect_buf_from_ws(ws, n)
    aux = collect_aux_from_ws(ws, n)

    m = fast_median!(data)                 # quickselect on data — data reordered
    mad = mad_std!(data, aux, m)
    return (m, mad)
end

# Specialisation 2 — (FastMedian, generic spread)
#
# Most spread functions are permutation-invariant, so calling them after
# fast_median! has reordered buf is safe.
#
@inline function _compute_stats(
        ::typeof(fast_median!), 
        spread_f::S,
        n::Int,
        ws::WS
    ) where {S, WS}

    data = collect_buf_from_ws(ws, n)
    m = fast_median!(data)
    s = statistic(spread_f, ws, n)
    return m, s
end

# Specialisation 3 — generic fallback
#
# Both reducers are plain callables.  No buffer reuse assumptions are made.
#
@inline function _compute_stats(
        center_f::C,
        spread_f::S,
        n::Int,
        ws::WS
    ) where {WS, C, S}

    c = statistic(center_f, ws, n)
    s = statistic(spread_f, ws, n)
    return c, s
end


@inline need_aux(::typeof(mad_std!)) = true
@inline need_aux(_) = false
