module SigmaClip



export sigma_clip_mask, sigma_clip_mask!, sigma_clip!, sigma_clip
export SigmaClipWorkspace
export fast_median!, mad_std!

const BAD_PIXEL = false
const GOOD_PIXEL = true


# ─── Reducer specialisations ──────────────────────────────────────────────────
#
# Reducer functions are singleton objects, so _compute_stats can still
# specialise on `typeof(fast_median!)` / `typeof(mad_std!)` at compile time.
# Users pass the public function directly (e.g. `center=fast_median!`) to opt
# into the fast path; arbitrary callables fall through to the generic fallback.

# ─── Workspace ────────────────────────────────────────────────────────────────

"""
    SigmaClipWorkspace{T <: AbstractFloat}

Pre-allocated scratch space for sigma clipping.  Pass one instance via the
`workspace` keyword to eliminate all dynamic allocations in hot loops.
`SigmaClipWorkspace` is the built-in implementation of SigmaClip's public
workspace protocol; external packages may pass their own workspace types by
implementing [`SigmaClip.workspace_buffer`](@ref) and
[`SigmaClip.workspace_auxbuffer`](@ref).

# Fields
- `buf` — working copy of the valid elements; compacted in-place each iteration.
- `aux` — auxiliary buffer; used only when `spread=mad_std!` to hold
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

"""
    SigmaClip.workspace_buffer(ws) -> AbstractVector

Return the main mutable scratch buffer used by SigmaClip for packed valid data.
Custom workspace types can participate in the allocation-free API by
implementing this method and [`SigmaClip.workspace_auxbuffer`](@ref).
"""
workspace_buffer(ws) = throw(ArgumentError(
    "unsupported workspace $(typeof(ws)); implement SigmaClip.workspace_buffer(::$(typeof(ws))) and SigmaClip.workspace_auxbuffer(::$(typeof(ws)))"))

"""
    SigmaClip.workspace_auxbuffer(ws) -> AbstractVector

Return the auxiliary mutable scratch buffer used by SigmaClip's specialised
`mad_std!` path. Custom workspace types can participate in the allocation-free
API by implementing this method and [`SigmaClip.workspace_buffer`](@ref).
"""
workspace_auxbuffer(ws) = throw(ArgumentError(
    "unsupported workspace $(typeof(ws)); implement SigmaClip.workspace_buffer(::$(typeof(ws))) and SigmaClip.workspace_auxbuffer(::$(typeof(ws)))"))

workspace_buffer(ws::SigmaClipWorkspace) = ws.buf
workspace_auxbuffer(ws::SigmaClipWorkspace) = ws.aux

SigmaClipWorkspace(T::Type{<:AbstractFloat}, n::Int) =
    SigmaClipWorkspace{T}(Vector{T}(undef, n), Vector{T}(undef, n))

SigmaClipWorkspace(x::AbstractArray{T}) where {T<:AbstractFloat} =
    SigmaClipWorkspace(T, length(x))

SigmaClipWorkspace(x::AbstractArray{<:Integer}) =
    SigmaClipWorkspace(Float64, length(x))

@inline function _ensure_workspace(::Type{T}, n::Int, ::Nothing) where {T<:AbstractFloat}
    SigmaClipWorkspace(T, n)
end

@inline function _validate_workspace_buffer(buf, ::Type{T}, n::Int, role::AbstractString) where {T<:AbstractFloat}
    buf isa AbstractVector || throw(ArgumentError(
        "workspace $role must be an AbstractVector, got $(typeof(buf))"))
    eltype(buf) === T || throw(ArgumentError(
        "workspace $role type mismatch: expected AbstractVector{$T}, got $(typeof(buf))"))
    Base.ismutable(buf) || throw(ArgumentError(
        "workspace $role must be mutable, got $(typeof(buf))"))
    length(buf) >= n || throw(ArgumentError(
        "workspace $role too short: length $(length(buf)) < required $n"))
    nothing
end

@inline function _ensure_workspace(::Type{T}, n::Int, ws) where {T<:AbstractFloat}
    buf = workspace_buffer(ws)
    aux = workspace_auxbuffer(ws)
    _validate_workspace_buffer(buf, T, n, "buffer")
    _validate_workspace_buffer(aux, T, n, "aux buffer")
    ws
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

See also: [`mad_std!`](@ref)
"""
function fast_median!(a::AbstractVector{T}) where T
    n = length(a)
    n == 0 && return zero(T)
    o = firstindex(a) - 1
    iseven(n) ? T(0.5) * (_kth_smallest!(a, o + n ÷ 2) + _kth_smallest!(a, o + n ÷ 2 + 1)) :
    _kth_smallest!(a, o + (n + 1) ÷ 2)
end

function fast_median!(a::AbstractVector{<:Integer})
    n = length(a)
    n == 0 && return 0.0
    o = firstindex(a) - 1
    iseven(n) ? 0.5 * (_kth_smallest!(a, o + n ÷ 2) + _kth_smallest!(a, o + n ÷ 2 + 1)) :
    _kth_smallest!(a, o + (n + 1) ÷ 2)
end

"""
    mad_std!(a::AbstractVector)

Compute the Median Absolute Deviation of `a` in-place, scaled by 1.4826 to
match the standard deviation of a normal distribution.

When passed as `spread=mad_std!`, SigmaClip selects the built-in robust
dispersion estimator. When combined with `center=fast_median!` (default), the
median is computed once and shared with the MAD calculation.

See also: [`fast_median!`](@ref)
"""
function mad_std!(a::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(a)
    n == 0 && return zero(T)

    m = fast_median!(a)
    aux = similar(a)
    @inbounds for i in eachindex(a)
        aux[i] = abs(a[i] - m)
    end
    fast_median!(aux) * T(1.4826022185056018)
end

function mad_std!(a::AbstractVector{<:Integer})
    n = length(a)
    n == 0 && return 0.0

    m = fast_median!(a)
    aux = Vector{Float64}(undef, n)
    @inbounds for i in eachindex(a)
        aux[i] = abs(a[i] - m)
    end
    fast_median!(aux) * 1.4826022185056018
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
# Three specialisations, resolved at compile time:
#
#   (fast_median!, mad_std!) — fully specialised; median shared between centre
#                              and MAD; two quickselects, one deviation loop
#   (fast_median!, generic)  — fast centre, user-supplied dispersion
#   (generic,     generic)   — both reducers are plain callables (fallback)


# Specialisation 1 — (FastMedian, MADStd)
#
# After fast_median!(buf[1:n]) the buffer is reordered but all n values remain.
# We compute |buf[i] − m| into aux[1:n] (leaving buf intact), then run a second
# quickselect on aux to get the MAD.
#
@inline function _compute_stats(::typeof(fast_median!), ::typeof(mad_std!),
    n::Int,
    ws)
    buf = workspace_buffer(ws)
    aux = workspace_auxbuffer(ws)
    T = eltype(buf)
    bv = @inbounds view(buf, 1:n)
    m = fast_median!(bv)                   # quickselect on buf — buf reordered

    @inbounds for i in 1:n                  # write deviations to aux, buf intact
        aux[i] = abs(buf[i] - m)
    end
    av = @inbounds view(aux, 1:n)
    mad = fast_median!(av)                  # quickselect on aux

    return m, mad * T(1.4826022185056018)
end

# Specialisation 2 — (FastMedian, generic spread)
#
# Most spread functions are permutation-invariant, so calling them after
# fast_median! has reordered buf is safe.
#
@inline function _compute_stats(::typeof(fast_median!), spread_f,
    n::Int,
    ws)
    buf = workspace_buffer(ws)
    bv = @inbounds view(buf, 1:n)
    m = fast_median!(bv)
    s = spread_f(bv)
    return m, s
end

# Specialisation 3 — generic fallback
#
# Both reducers are plain callables.  No buffer reuse assumptions are made.
#
@inline function _compute_stats(center_f, spread_f,
    n::Int,
    ws)
    buf = workspace_buffer(ws)
    bv = @inbounds view(buf, 1:n)
    return center_f(bv), spread_f(bv)
end


# ─── Core bounds algorithm ────────────────────────────────────────────────────

function _sigma_clip_bounds_impl(
    x::AbstractArray{T},
    exclude::M,
    ws,
    sigma_lower,
    sigma_upper,
    center::C,
    spread::S,
    maxiter::Int,
) where {T,M,C,S}

    have_exclude = !isnothing(exclude)
    W = eltype(workspace_buffer(ws))
    sigma_lower = convert(W, sigma_lower)
    sigma_upper = convert(W, sigma_upper)
    buf = workspace_buffer(ws)

    # Pack valid (finite, not excluded) elements into the workspace buffer
    n = 0
    @inbounds for i in eachindex(x)
        if (!have_exclude || !exclude[i]) && isfinite(x[i])
            n += 1
            buf[n] = x[i]
        end
    end

    n == 0 && return (zero(W), zero(W))

    current = n
    lower_bound = zero(W)
    upper_bound = zero(W)
    iter = 0

    while true
        c, s = _compute_stats(center, spread, current, ws)
        c = convert(W, c)
        s = convert(W, s)

        lower_bound = c - s * sigma_lower
        upper_bound = c + s * sigma_upper

        # In-place compaction — write index <= read index always holds
        new_count = 0
        @inbounds for i in 1:current
            val = buf[i]
            if val >= lower_bound && val <= upper_bound
                new_count += 1
                buf[new_count] = val
            end
        end

        new_count == current && return (lower_bound, upper_bound)
        current = new_count
        iter += 1

        (maxiter != -1 && iter >= maxiter) && return (lower_bound, upper_bound)
        current < 2 && return (lower_bound, upper_bound)
    end
end

@inline function _sigma_clip_bounds_checked(
    x::AbstractArray{T},
    workspace,
    exclude::Union{Nothing,AbstractArray{Bool}},
    sigma_lower,
    sigma_upper,
    center::C,
    spread::S,
    maxiter::Int,
) where {T,C,S}
    ws = _ensure_workspace(float(T), length(x), workspace)
    _sigma_clip_bounds_impl(x, exclude, ws,
        float(sigma_lower), float(sigma_upper),
        center, spread, maxiter)
end


# ─── Public API ───────────────────────────────────────────────────────────────

"""
    sigma_clip_mask(x; kwargs...) -> BitArray

Identify valid pixels in `x` via iterative sigma clipping.  Returns a `BitArray`
where `true` marks a finite, non-clipped value.  `x` is never modified.

# Keyword Arguments
- `workspace=nothing`  — pre-allocated workspace for allocation-free operation;
                         accepts [`SigmaClipWorkspace`](@ref) or any custom type
                         implementing [`SigmaClip.workspace_buffer`](@ref) and
                         [`SigmaClip.workspace_auxbuffer`](@ref).
- `sigma_lower=3`      — lower rejection threshold (multiples of dispersion).
- `sigma_upper=3`      — upper rejection threshold.
- `maxiter=5`          — maximum iterations; `-1` means run until convergence.
- `center=fast_median!` — centre estimator; any `f(v::AbstractVector) -> scalar`.
- `spread=mad_std!`     — dispersion estimator; any `f(v::AbstractVector) -> scalar`.
- `exclude=nothing`    — boolean array selecting points excluded from bound computation;
                         unlike the returned mask, here `true` means "exclude".

# Example
```julia
data = randn(1000); data[1] = 99.0
clean = data[sigma_clip_mask(data)]                        # default
clean = data[sigma_clip_mask(data; spread=mad_std!)]      # robust MAD
```
"""
function sigma_clip_mask(x::AbstractArray{T};
    workspace=nothing,
    exclude::Union{Nothing,AbstractArray{Bool}}=nothing,
    sigma_lower=3,
    sigma_upper=3,
    center::C=fast_median!,
    spread::S=mad_std!,
    maxiter::Int=5) where {T,C,S}
    target = falses(size(x))
    sigma_clip_mask!(x, target;
        workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter)
end

"""
    sigma_clip_mask!(x, target; kwargs...) -> target

In-place mask variant: writes pixel-validity flags into the pre-allocated `BitArray`
`target` (same shape as `x`).  Same keyword arguments as [`sigma_clip_mask`](@ref).
"""
function sigma_clip_mask!(x::AbstractArray{T},
    target::AbstractArray{Bool};
    workspace=nothing,
    exclude::Union{Nothing,AbstractArray{Bool}}=nothing,
    sigma_lower=3,
    sigma_upper=3,
    center::C=fast_median!,
    spread::S=mad_std!,
    maxiter::Int=5) where {T,C,S}

    lb, ub = _sigma_clip_bounds_checked(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter)
    @inbounds for i in eachindex(x)
        val = x[i]
        target[i] = isfinite(val) && val >= lb && val <= ub
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
sigma_clip!(data)                    # fast_median! + mad_std! (default)
sigma_clip!(data; spread=std)        # median + standard deviation
sigma_clip!(data; center=mean, spread=std)  # fully custom
```
"""
function sigma_clip!(x::AbstractArray{T};
    workspace=nothing,
    exclude::Union{Nothing,AbstractArray{Bool}}=nothing,
    sigma_lower=3,
    sigma_upper=3,
    center::C=fast_median!,
    spread::S=mad_std!,
    maxiter::Int=5) where {T<:AbstractFloat,C,S}

    lb, ub = _sigma_clip_bounds_checked(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter)
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
lb, ub = SigmaClip.sigma_clip_bounds(data; sigma_lower=2.5, spread=mad_std!)
println("outliers: x < \$lb  or  x > \$ub")
```
"""
function sigma_clip_bounds(x::AbstractArray{T};
    workspace=nothing,
    exclude::Union{Nothing,AbstractArray{Bool}}=nothing,
    sigma_lower=3,
    sigma_upper=3,
    center::C=fast_median!,
    spread::S=mad_std!,
    maxiter::Int=5) where {T,C,S}

    _sigma_clip_bounds_checked(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter)
end

end # module SigmaClip
