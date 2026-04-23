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

"""
    fast_median!(a::AbstractVector) -> eltype(a)

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
    n == 0 && return zero(T)
    o = firstindex(a) - 1
<<<<<<< Updated upstream
    iseven(n) ? (_kth_smallest!(a, o + n ÷ 2) + _kth_smallest!(a, o + n ÷ 2 + 1)) / 2 :
    _kth_smallest!(a, o + (n + 1) ÷ 2)
=======
    # Use quickselect directly so the median stays allocation-free.
    return iseven(n) ? (_kth_smallest!(a, o + n ÷ 2) + _kth_smallest!(a, o + n ÷ 2 + 1)) / 2 :
        _kth_smallest!(a, o + (n + 1) ÷ 2)
>>>>>>> Stashed changes
end

function fast_median!(a::AbstractVector{<:Integer})
    n = length(a)
    n == 0 && return 0.0
    o = firstindex(a) - 1
    return iseven(n) ? 0.5 * (_kth_smallest!(a, o + n ÷ 2) + _kth_smallest!(a, o + n ÷ 2 + 1)) :
        _kth_smallest!(a, o + (n + 1) ÷ 2)
end

"""
    mad_std!(a::AbstractVector)

Compute the Median Absolute Deviation of `a` in-place, scaled by 1.4826 to
match the standard deviation of a normal distribution.

When passed as `spread=mad_std!`, SigmaClip selects the built-in robust
dispersion estimator. When combined with `center=fast_median!` (default), the
median is computed once and shared with the MAD calculation.
When used inside SigmaClip with a workspace, `mad_std!` uses the workspace
auxiliary buffer instead of allocating its own.

See also: [`fast_median!`](@ref)
"""
function mad_std!(a::AbstractVector{T}) where {T <: Number}
    n = length(a)
    n == 0 && return zero(T)

    m = fast_median!(a)
    aux = similar(a)
    @inbounds for i in eachindex(a)
        aux[i] = abs(a[i] - m)
    end
    return fast_median!(aux) * _scale_factor(T, 1.4826022185056018)
end

function mad_std!(a::AbstractVector{<:Integer})
    n = length(a)
    n == 0 && return 0.0

    m = fast_median!(a)
    aux = Vector{Float64}(undef, n)
    @inbounds for i in eachindex(a)
        aux[i] = abs(a[i] - m)
    end
    return fast_median!(aux) * 1.4826022185056018
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
@inline function statistic(f, ws, n::Int)
    data = @inbounds view(workspace_buffer(ws), 1:n)
    return f(data)
end

@inline function statistic(::typeof(mad_std!), ws, n::Int)
    data = @inbounds view(workspace_buffer(ws), 1:n)
    aux = @inbounds view(workspace_auxbuffer(ws), 1:n)
    T = eltype(aux)

    m = fast_median!(data)
    @inbounds for i in eachindex(data)
        aux[i] = abs(data[i] - m)
    end

    return fast_median!(aux) * _scale_factor(T, 1.4826022185056018)
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
        ::typeof(fast_median!), ::typeof(mad_std!),
        n::Int,
        ws
    )
    data = @inbounds view(workspace_buffer(ws), 1:n)
    aux = @inbounds view(workspace_auxbuffer(ws), 1:n)
    T = eltype(aux)
    m = fast_median!(data)                 # quickselect on data — data reordered

    @inbounds for i in eachindex(data)      # write deviations to aux, data intact
        aux[i] = abs(data[i] - m)
    end
    mad = fast_median!(aux)                 # quickselect on aux

    return m, mad * _scale_factor(T, 1.4826022185056018)
end

# Specialisation 2 — (FastMedian, generic spread)
#
# Most spread functions are permutation-invariant, so calling them after
# fast_median! has reordered buf is safe.
#
@inline function _compute_stats(
        ::typeof(fast_median!), spread_f,
        n::Int,
        ws
    )
    data = @inbounds view(workspace_buffer(ws), 1:n)
    m = fast_median!(data)
    s = statistic(spread_f, ws, n)
    return m, s
end

# Specialisation 3 — generic fallback
#
# Both reducers are plain callables.  No buffer reuse assumptions are made.
#
@inline function _compute_stats(
        center_f, spread_f,
        n::Int,
        ws
    )
    c = statistic(center_f, ws, n)
    s = statistic(spread_f, ws, n)
    return c, s
end
