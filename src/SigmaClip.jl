module SigmaClip



export sigma_clip_mask, sigma_clip_mask!, sigma_clip!, sigma_clip
export SigmaClipWorkspace
export fast_median!, mad_std!

const BAD_PIXEL = false
const GOOD_PIXEL = true


# ─── Statistics reducers ─────────────────────────────────────────────────────

include("stats.jl")


# ─── Reducer specialisations ──────────────────────────────────────────────────
#
# Reducer functions are singleton objects, so _compute_stats can still
# specialise on `typeof(fast_median!)` / `typeof(mad_std!)` at compile time.
# Users pass the public function directly (e.g. `center=fast_median!`) to opt
# into the fast path; arbitrary callables fall through to the generic fallback.

# ─── Workspace ────────────────────────────────────────────────────────────────

"""
    SigmaClipWorkspace{T <: Number}

Pre-allocated scratch space for sigma clipping.  Pass one instance via the
`workspace` keyword to eliminate all dynamic allocations in hot loops.
`SigmaClipWorkspace` is the built-in implementation of SigmaClip's public
workspace protocol; external packages may pass their own workspace types by
implementing [`SigmaClip.workspace_buffer`](@ref) and
[`SigmaClip.workspace_auxbuffer`](@ref).

# Fields
- `buf` — working copy of the valid elements; compacted in-place each iteration.
- `aux` — auxiliary buffer; used by `mad_std!` and available to workspace-aware
  reducers through [`SigmaClip.workspace_auxbuffer`](@ref).

# Constructors

    SigmaClipWorkspace(T, n)     # explicit numeric type and capacity
    SigmaClipWorkspace(x)        # T and n inferred from the input array

# Example
```julia
ws = SigmaClipWorkspace(Float64, size(image, 2))
for row in eachrow(image)
    sigma_clip!(row; workspace=ws)
end
```
"""
struct SigmaClipWorkspace{T<:Number}
    buf::Vector{T}
    aux::Vector{T}
end

"""
    SigmaClip.workspace_buffer(ws) -> AbstractVector

Return the main mutable scratch buffer used by SigmaClip for packed valid data.
Custom workspace types can participate in the allocation-free API by returning
a writable, 1-indexed `AbstractVector` and implementing
[`SigmaClip.workspace_auxbuffer`](@ref).
"""
workspace_buffer(ws) = throw(ArgumentError(
    "unsupported workspace $(typeof(ws)); implement SigmaClip.workspace_buffer(::$(typeof(ws))) and SigmaClip.workspace_auxbuffer(::$(typeof(ws)))"))

"""
    SigmaClip.workspace_auxbuffer(ws) -> AbstractVector

Return the auxiliary mutable scratch buffer used by SigmaClip's specialised
`mad_std!` path and workspace-aware statistics. Custom workspace types can
participate in the allocation-free API by returning a writable, 1-indexed
`AbstractVector` and implementing [`SigmaClip.workspace_buffer`](@ref).
"""
workspace_auxbuffer(ws) = throw(ArgumentError(
    "unsupported workspace $(typeof(ws)); implement SigmaClip.workspace_buffer(::$(typeof(ws))) and SigmaClip.workspace_auxbuffer(::$(typeof(ws)))"))

workspace_buffer(ws::SigmaClipWorkspace) = ws.buf
workspace_auxbuffer(ws::SigmaClipWorkspace) = ws.aux

SigmaClipWorkspace(T::Type{<:Number}, n::Int) =
    SigmaClipWorkspace{T}(Vector{T}(undef, n), Vector{T}(undef, n))

SigmaClipWorkspace(x::AbstractArray{T}) where {T<:Number} =
    SigmaClipWorkspace(T, length(x))

SigmaClipWorkspace(x::AbstractArray{<:Integer}) =
    SigmaClipWorkspace(Float64, length(x))

@inline _workspace_eltype(::Type{T}) where {T<:AbstractFloat} = T
@inline _workspace_eltype(::Type{<:Integer}) = Float64
@inline _workspace_eltype(::Type{T}) where {T<:Number} = T

@inline _scale_factor(::Type{T}, x) where {T<:AbstractFloat} = convert(T, x)
@inline _scale_factor(::Type{<:Number}, x) = float(x)

@inline _nan_value(::Type{T}) where {T<:AbstractFloat} = T(NaN)
@inline _nan_value(::Type{T}) where {T<:Number} = convert(T, NaN * oneunit(T))

@inline _same_axes(a, b) = axes(a) == axes(b)
# @inline _is_one_indexed(v) = firstindex(v) == 1
@inline _is_nonnegative_finite(x) = isfinite(x) && x >= zero(x)
@inline _validate_axes(name, a, x) = _same_axes(a, x) || throw(ArgumentError("$name axes mismatch: expected $(axes(x)), got $(axes(a))"))
@inline _validate_sigma(name, value) = _is_nonnegative_finite(value) || throw(ArgumentError("$name must be finite and non-negative, got $value"))

@inline function _ensure_workspace(::Type{T}, n::Int, ::Nothing) where {T<:Number}
    SigmaClipWorkspace(T, n)
end

@inline function _validate_workspace_buffer(buf, ::Type{T}, n::Int, role::AbstractString) where {T<:Number}
    buf isa AbstractVector || throw(ArgumentError(
        "workspace $role must be an AbstractVector, got $(typeof(buf))"))
    eltype(buf) === T || throw(ArgumentError(
        "workspace $role type mismatch: expected AbstractVector{$T}, got $(typeof(buf))"))
    length(buf) >= n || throw(ArgumentError(
        "workspace $role too short: length $(length(buf)) < required $n"))
    # _is_one_indexed(buf) || throw(ArgumentError(
    #     "workspace $role must be 1-indexed, got firstindex $(firstindex(buf))"))
    # if n > 0
    #     try
    #         buf[firstindex(buf)] = zero(T)
    #     catch err
    #         throw(ArgumentError(
    #             "workspace $role must support setindex!, got $(typeof(buf)): $err"))
    #     end
    # end
    nothing
end

@inline function _ensure_workspace(::Type{T}, n::Int, ws) where {T<:Number}
    buf = workspace_buffer(ws)
    aux = workspace_auxbuffer(ws)
    _validate_workspace_buffer(buf, T, n, "buffer")
    _validate_workspace_buffer(aux, T, n, "aux buffer")
    ws
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
    sigma_lower = _scale_factor(W, sigma_lower)
    sigma_upper = _scale_factor(W, sigma_upper)
    _validate_sigma("sigma_lower", sigma_lower)
    _validate_sigma("sigma_upper", sigma_upper)
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
    !isnothing(exclude) && _validate_axes("exclude", exclude, x)
    ws = _ensure_workspace(_workspace_eltype(T), length(x), workspace)
    _sigma_clip_bounds_impl(x, exclude, ws,
        sigma_lower, sigma_upper,
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
- `sigma_lower=3`      — finite, non-negative lower rejection threshold.
- `sigma_upper=3`      — finite, non-negative upper rejection threshold.
- `maxiter=5`          — maximum iterations; `-1` means run until convergence.
- `center=fast_median!` — centre estimator; any `f(v::AbstractVector) -> scalar`,
                          or a workspace-aware reducer implementing
                          `SigmaClip.statistic(f, ws, n)`.
- `spread=mad_std!`     — dispersion estimator; any `f(v::AbstractVector) -> scalar`,
                          or a workspace-aware reducer implementing
                          `SigmaClip.statistic(f, ws, n)`.
- `exclude=nothing`    — boolean array with the same axes as `x`; `true` excludes
                         a value from bound estimation only. Excluded values are
                         still classified against the final bounds.

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

In-place mask variant: writes pixel-validity flags into the pre-allocated boolean
`target` with the same axes as `x`. Same keyword arguments as [`sigma_clip_mask`](@ref).
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

    _validate_axes("target", target, x)
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
Requires an array whose element type can represent `NaN`. Use [`sigma_clip`](@ref)
for integer arrays, or convert integer input to floating point before calling
`sigma_clip!`.

Same keyword arguments as [`sigma_clip_mask`](@ref).

# Example
```julia
using Statistics

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
    maxiter::Int=5) where {T<:Number,C,S}

    lb, ub = _sigma_clip_bounds_checked(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter)
    nan = _nan_value(T)
    @inbounds for i in eachindex(x)
        val = x[i]
        if !isfinite(val) || val < lb || val > ub
            x[i] = nan
        end
    end
    return x
end

sigma_clip!(x::AbstractArray{<:Integer}; kw...) =
    throw(ArgumentError(
        "sigma_clip! requires an array whose element type can represent NaN; use sigma_clip(x) for integer arrays or convert the input to floating point"))

"""
    sigma_clip(x; kwargs...) -> Array{<:Number}

Out-of-place variant.  Returns a copy of `x` with outliers replaced by `NaN`.
Integer arrays are promoted to `Float64`; numeric arrays with units keep their
element type when it can represent `NaN`.
Same keyword arguments as [`sigma_clip!`](@ref).
"""
sigma_clip(x::AbstractArray{<:Number}; kw...) = sigma_clip!(copy(x); kw...)
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
