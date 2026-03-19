module SigmaClip



export sigma_clip_mask, sigma_clip_mask!, sigma_clip!, sigma_clip
export SigmaClipWorkspace
export fast_median!, fast_median, mad_std

const BAD_PIXEL = true
const GOOD_PIXEL = false


# ─── Reducer sentinels ────────────────────────────────────────────────────────
#
# Sentinels let the compiler specialise _compute_stats at compile time via
# multiple dispatch.  Users pass the constant (e.g. cent_reducer=fast_median)
# to opt into the fast path; arbitrary callables fall through to the generic
# fallback with no runtime penalty other than one extra function call.

"""
    fast_median

Sentinel for `cent_reducer`.  Selects the allocation-free quickselect median.
When combined with `std_reducer=mad_std`, the median is computed once and
shared with the MAD calculation (no redundant quickselect).

See also: [`fast_median!`](@ref), [`mad_std`](@ref)
"""
struct _FastMedian end
const fast_median = _FastMedian()

"""
    mad_std

Sentinel for `std_reducer`.  Selects the Median Absolute Deviation scaled by
1.4826 as the dispersion estimator — more robust than `std` for heavy-tailed
distributions or data with many outliers.

When `cent_reducer=fast_median` (default), the median is shared between the
centre and the deviation calculation: only two quickselects per iteration
instead of three.

See also: [`fast_median`](@ref)
"""
struct _MADStd end
const mad_std = _MADStd()


# ─── Workspace ────────────────────────────────────────────────────────────────

"""
    SigmaClipWorkspace{T <: AbstractFloat}

Pre-allocated scratch space for sigma clipping.  Pass one instance via the
`workspace` keyword to eliminate all dynamic allocations in hot loops.

# Fields
- `buf` — working copy of the valid elements; compacted in-place each iteration.
- `aux` — auxiliary buffer; used only when `std_reducer=mad_std` to hold
  `|buf[i] − median|` without overwriting `buf` (which is still needed for the
  in-place compaction step that follows statistics computation).

# Constructors

    SigmaClipWorkspace(T, n)     # explicit floating-point type and capacity
    SigmaClipWorkspace(x)        # T and n inferred from the input array

# Example
```julia
ws = SigmaClipWorkspace(Float64, size(image, 2))
for row in eachrow(image)
    sigma_clip!(row; workspace=ws)
end
```
"""
struct SigmaClipWorkspace{T<:AbstractFloat}
    buf::Vector{T}
    aux::Vector{T}
end

SigmaClipWorkspace(T::Type{<:AbstractFloat}, n::Int) =
    SigmaClipWorkspace{T}(Vector{T}(undef, n), Vector{T}(undef, n))

SigmaClipWorkspace(x::AbstractArray{T}) where {T<:AbstractFloat} =
    SigmaClipWorkspace(T, length(x))

SigmaClipWorkspace(x::AbstractArray{<:Integer}) =
    SigmaClipWorkspace(Float64, length(x))

@inline function _ensure_workspace(::Type{T}, n::Int, ::Nothing) where T
    SigmaClipWorkspace(T, n)
end
@inline function _ensure_workspace(::Type{T}, n::Int, ws::SigmaClipWorkspace) where T
    length(ws.buf) >= n || throw(ArgumentError(
        "workspace buffer too short: length $(length(ws.buf)) < required $n"))
    ws
end


# ─── _compute_stats ───────────────────────────────────────────────────────────
#
# Returns (centre, dispersion) for the n values packed in buf[1:n].
#
# Contract:
#   • may reorder buf[1:n] (quickselect is partial, values are preserved)
#   • must NOT overwrite buf[1:n] with unrelated data — compaction reads it next
#   • may freely read and write ws.aux[1:n]
#
# Three specialisations, resolved at compile time:
#
#   (_FastMedian, _MADStd)   — fully specialised; median shared between centre
#                              and MAD; two quickselects, one deviation loop
#   (_FastMedian, generic)   — fast centre, user-supplied dispersion
#   (generic,     generic)   — both reducers are plain callables (fallback)


# Specialisation 1 — (FastMedian, MADStd)
#
# After fast_median!(buf[1:n]) the buffer is reordered but all n values remain.
# We compute |buf[i] − m| into aux[1:n] (leaving buf intact), then run a second
# quickselect on aux to get the MAD.
#
@inline function _compute_stats(::_FastMedian, ::_MADStd,
    n::Int,
    ws::SigmaClipWorkspace{T}) where {T}
    
    buf = ws.buf
    bv = @inbounds view(buf, 1:n)
    m = fast_median!(bv)                   # quickselect on buf — buf reordered

    @inbounds for i in 1:n                  # write deviations to aux, buf intact
        ws.aux[i] = abs(buf[i] - m)
    end
    av = @inbounds view(ws.aux, 1:n)
    mad = fast_median!(av)                  # quickselect on aux

    return m, mad * T(1.4826022185056018)
end

# Specialisation 2 — (FastMedian, generic std)
#
# std (and most other dispersion functions) are permutation-invariant, so
# calling them after fast_median! has reordered buf is safe.
#
@inline function _compute_stats(::_FastMedian, std_f,
    n::Int,
    ws::SigmaClipWorkspace{T}) where {T}
    buf = ws.buf
    bv = @inbounds view(buf, 1:n)
    m = fast_median!(bv)
    s = std_f(bv)
    return m, s
end

# Specialisation 3 — generic fallback
#
# Both reducers are plain callables.  No buffer reuse assumptions are made.
#
@inline function _compute_stats(cent_f, std_f,
    n::Int,
    ws::SigmaClipWorkspace{T}) where T
    buf = ws.buf
    bv = @inbounds view(buf, 1:n)
    return cent_f(bv), std_f(bv)
end


# ─── Core bounds algorithm ────────────────────────────────────────────────────

function _sigma_clip_bounds(
    x::AbstractArray{T},
    mask::M,
    ws::SigmaClipWorkspace,
    sigma_lower::L,
    sigma_upper::L,
    cent_reducer,
    std_reducer,
    maxiter::Int,
) where {T,M,L}

    have_mask = !isnothing(mask)

    # Pack valid (finite, unmasked) elements into ws.buf
    n = 0
    @inbounds for i in eachindex(x)
        if (!have_mask || mask[i] != BAD_PIXEL) && isfinite(x[i])
            n += 1
            ws.buf[n] = x[i]
        end
    end

    n == 0 && return (T(0), T(0))

    current = n
    lower_bound = T(0)
    upper_bound = T(0)
    iter = 0

    while true
        c, s = _compute_stats(cent_reducer, std_reducer, current, ws)

        lower_bound = c - s * sigma_lower
        upper_bound = c + s * sigma_upper

        # In-place compaction — write index <= read index always holds
        new_count = 0
        @inbounds for i in 1:current
            val = ws.buf[i]
            if val >= lower_bound && val <= upper_bound
                new_count += 1
                ws.buf[new_count] = val
            end
        end

        new_count == current && return (lower_bound, upper_bound)
        current = new_count
        iter += 1

        (maxiter != -1 && iter >= maxiter) && return (lower_bound, upper_bound)
        current < 2 && return (lower_bound, upper_bound)
    end
end


# ─── Public API ───────────────────────────────────────────────────────────────

"""
    sigma_clip_mask(x; kwargs...) -> BitArray

Identify outliers in `x` via iterative sigma clipping.  Returns a `BitArray`
where `true` marks a non-finite or clipped value.  `x` is never modified.

# Keyword Arguments
- `workspace=nothing`  — [`SigmaClipWorkspace`](@ref) for allocation-free operation.
- `sigma_lower=3`      — lower rejection threshold (multiples of dispersion).
- `sigma_upper=3`      — upper rejection threshold.
- `maxiter=5`          — maximum iterations; `-1` means run until convergence.
- `cent_reducer=fast_median` — centre estimator; any `f(v::AbstractVector) -> scalar`.
- `std_reducer=std`    — dispersion estimator; pass [`mad_std`](@ref) for robustness.
- `mask=nothing`       — initial boolean mask (`true` = exclude when computing bounds).

# Example
```julia
data = randn(1000); data[1] = 99.0
clean = data[.!sigma_clip_mask(data)]                       # default
clean = data[.!sigma_clip_mask(data; std_reducer=mad_std)]  # robust MAD
```
"""
function sigma_clip_mask(x::AbstractArray{T};
    workspace::Union{Nothing,SigmaClipWorkspace}=nothing,
    kwargs...) where T
    target = trues(size(x))
    sigma_clip_mask!(x, target; workspace, kwargs...)
end

"""
    sigma_clip_mask!(x, target; kwargs...) -> target

In-place mask variant: writes outlier flags into the pre-allocated `BitArray`
`target` (same shape as `x`).  Same keyword arguments as [`sigma_clip_mask`](@ref).
"""
function sigma_clip_mask!(x::AbstractArray{T},
    target::AbstractArray{Bool};
    workspace::Union{Nothing,SigmaClipWorkspace}=nothing,
    mask::M=nothing,
    sigma_lower=3,
    sigma_upper=3,
    cent_reducer::C=fast_median,
    std_reducer::S=mad_std,
    maxiter::Int=5) where {T,M,C,S}

    ws = _ensure_workspace(float(T), length(x), workspace)
    lb, ub = _sigma_clip_bounds(x, mask, ws,
        float(sigma_lower), float(sigma_upper),
        cent_reducer, std_reducer, maxiter)
    @inbounds for i in eachindex(x)
        val = x[i]
        target[i] = !isfinite(val) || val < lb || val > ub
    end
    return target
end

"""
    sigma_clip!(x; kwargs...) -> x

In-place sigma clipping: replaces outliers in `x` with `NaN`.
Requires `x <: AbstractArray{<:AbstractFloat}`.

Same keyword arguments as [`sigma_clip_mask`](@ref).

# Example
```julia
data = randn(500); data[end] = 1e6
sigma_clip!(data)                         # fast_median + std (default)
sigma_clip!(data; std_reducer=mad_std)    # robust: median + MAD shared computation
sigma_clip!(data; cent_reducer=mean, std_reducer=std)  # fully custom
```
"""
function sigma_clip!(x::AbstractArray{T};
    workspace::Union{Nothing,SigmaClipWorkspace}=nothing,
    mask::Union{Nothing,AbstractArray{Bool}}=nothing,
    sigma_lower=3,
    sigma_upper=3,
    cent_reducer::C=fast_median,
    std_reducer::S=mad_std,
    maxiter::Int=5) where {T<:AbstractFloat,C,S}

    ws = _ensure_workspace(T, length(x), workspace)
    lb, ub = _sigma_clip_bounds(x, mask, ws,
        float(sigma_lower), float(sigma_upper),
        cent_reducer, std_reducer, maxiter)
    @inbounds for i in eachindex(x)
        val = x[i]
        if !isfinite(val) || val < lb || val > ub
            x[i] = T(NaN)
        end
    end
    return x
end

"""
    sigma_clip(x; kwargs...) -> Array{<:AbstractFloat}

Out-of-place variant.  Returns a copy of `x` with outliers replaced by `NaN`.
Integer arrays are promoted to `Float64`.
Same keyword arguments as [`sigma_clip!`](@ref).
"""
sigma_clip(x::AbstractArray{<:AbstractFloat}; kw...) = sigma_clip!(copy(x); kw...)
sigma_clip(x::AbstractArray{<:Integer}; kw...) = sigma_clip!(float.(x); kw...)

"""
    SigmaClip.sigma_clip_bounds(x; kwargs...) -> (lb, ub)

Return the final convergence bounds without modifying `x` or producing a mask.
Accepts the same keyword arguments as [`sigma_clip_mask`](@ref).

```julia
lb, ub = SigmaClip.sigma_clip_bounds(data; sigma_lower=2.5, std_reducer=mad_std)
println("outliers: x < \$lb  or  x > \$ub")
```
"""
function sigma_clip_bounds(x::AbstractArray{T};
    workspace::Union{Nothing,SigmaClipWorkspace}=nothing,
    mask=nothing,
    sigma_lower=3,
    sigma_upper=3,
    cent_reducer::C=fast_median,
    std_reducer::S=mad_std,
    maxiter::Int=5) where {T,C,S}

    ws = _ensure_workspace(float(T), length(x), workspace)
    _sigma_clip_bounds(x, mask, ws,
        float(sigma_lower), float(sigma_upper),
        cent_reducer, std_reducer, maxiter)
end


# ─── Quickselect median ───────────────────────────────────────────────────────

function _kth_smallest!(a::AbstractVector{T}, k::Int) where T
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

Allocation-free; roughly 2–3× faster than `Statistics.median` on random data.

See also: [`fast_median`](@ref) (sentinel for use as `cent_reducer`)
"""
function fast_median!(a::AbstractVector{T}) where T
    n = length(a)
    n == 0 && return zero(T)
    o = firstindex(a) - 1
    iseven(n) ? T(0.5) * (_kth_smallest!(a, o + n ÷ 2) + _kth_smallest!(a, o + n ÷ 2 + 1)) :
    _kth_smallest!(a, o + (n + 1) ÷ 2)
end

end # module SigmaClip