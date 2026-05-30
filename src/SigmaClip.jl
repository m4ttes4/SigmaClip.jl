module SigmaClip

# ─── Statistics reducers ─────────────────────────────────────────────────────

include("workspace.jl")
include("stats.jl")

export sigma_clip_mask, sigma_clip_mask!, sigma_clip!, sigma_clip
export WorkSpace
export fast_median!, mad_std!

const BAD_PIXEL = false
const GOOD_PIXEL = true


function validate_sigma(s::T) where {T <: Real}
    if !isfinite(s) || s <= zero(T)
        throw(ArgumentError("Sigma must be finite and strictly > 0"))
    end
    return nothing
end

function validate_axes(a::AbstractArray{Bool}, x::AbstractArray)
    if axes(a) != axes(x)
        throw(DimensionMismatch("Helper array must have the same axes as input array"))
    end
    return nothing
end

function validate_maxiter(a::Integer)
    if a < -1 || a == 0
        throw(ArgumentError("Max iters must be > 0, else pass -1 to run until convergence"))
    end
    return nothing
end



#TODO we can write a version that includes missings
@inline function pack_valid!(buf::AbstractVector, x::AbstractArray, ::Nothing)
    n = 0
    @inbounds for i in eachindex(x)
        val = x[i]
        if isfinite(val)
            n += 1
            buf[n] = val
        end
    end
    return n
end

@inline function pack_valid!(buf::AbstractVector, x::AbstractArray, exclude::AbstractArray{Bool})
    n = 0
    @inbounds for i in eachindex(x, exclude)
        val = x[i]
        if !exclude[i] && isfinite(val)
            n += 1
            buf[n] = val
        end
    end
    return n
end



@inline function is_valid_data(x, up, low)
    if low <= x <= up
        return GOOD_PIXEL
    end
    return BAD_PIXEL
end
# ─── Core bounds algorithm ────────────────────────────────────────────────────

function _sigma_clip_bounds_impl(
        x::AbstractArray{T},
        exclude::M,
        ws::WS,
        sigma_lower::Real,
        sigma_upper::Real,
        center::C,
        spread::S,
        maxiter::Int,
    ) where {T, M, C, S, WS}

    

    buf = workspace_buffer(ws)

    n = pack_valid!(buf, x, exclude)
    n == 0 && throw(ArgumentError("No valid data to clip found"))

    current = n
    iter = 0

    while true
        c, s = _compute_stats(center, spread, current, ws)

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
    return
end

@inline function sigma_clip_bounds_checked(
        x::AbstractArray{T},
        workspace::WS,
        exclude::Union{Nothing, AbstractArray{Bool}},
        sigma_lower,
        sigma_upper,
        center::C,
        spread::S,
        maxiter::Int,
    ) where {T, C, S, WS}

    !isnothing(exclude) && validate_axes(exclude, x)

    ws = prepare_ws(x, spread, workspace)

    validate_maxiter(maxiter)

    validate_sigma(sigma_lower)
    validate_sigma(sigma_upper)

    return _sigma_clip_bounds_impl(
        x, exclude, ws,
        sigma_lower, sigma_upper,
        center, spread, maxiter
    )
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
clean = data[sigma_clip_mask(data; spread = mad_std!)]      # robust MAD
```
"""
function sigma_clip_mask(
        x::AbstractArray{T};
        workspace = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower = 3,
        sigma_upper = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5
    ) where {T, C, S}

    target = fill(GOOD_PIXEL, size(x))

    return sigma_clip_mask!(
        x, target;
        workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )
end

"""
    sigma_clip_mask!(x, target; kwargs...) -> target

In-place mask variant: writes pixel-validity flags into the pre-allocated boolean
`target` with the same axes as `x`. Same keyword arguments as [`sigma_clip_mask`](@ref).
"""
function sigma_clip_mask!(
        x::AbstractArray{T},
        target::AbstractArray{Bool};
        workspace = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower = 3,
        sigma_upper = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5
    ) where {T, C, S}
    
    validate_axes(target, x)
    
    lb, ub = sigma_clip_bounds_checked(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )

    @inbounds for i in eachindex(x)
        val = x[i]
        target[i] = is_valid_data(val, ub, lb)#isfinite(val) && val >= lb && val <= ub
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

data = randn(500); data[end] = 1.0e6
sigma_clip!(data)                    # fast_median! + mad_std! (default)
sigma_clip!(data; spread = std)        # median + standard deviation
sigma_clip!(data; center = mean, spread = std)  # fully custom
```
"""
function sigma_clip!(
        x::AbstractArray{T};
        workspace = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower = 3,
        sigma_upper = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5
    ) where {T <: Number, C, S}

    lb, ub = sigma_clip_bounds_checked(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )
    nan = T(NaN)


    @inbounds for i in eachindex(x)
        val = x[i]
        if !is_valid_data(val, ub, lb)
            x[i] = nan
        end
    end
    return x
end

sigma_clip!(::AbstractArray{<:Integer}; _kw...) =
    throw(
    ArgumentError(
        "sigma_clip! requires an array whose element type can represent NaN; use sigma_clip(x) for integer arrays or convert the input to floating point"
    )
)

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
lb, ub = SigmaClip.sigma_clip_bounds(data; sigma_lower = 2.5, spread = mad_std!)
println("outliers: x < \$lb  or  x > \$ub")
```
"""
function sigma_clip_bounds(
        x::AbstractArray{T};
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower::Real = 3,
        sigma_upper::Real = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5
    ) where {T, C, S, WS}

    return sigma_clip_bounds_checked(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )
end

end # module SigmaClip
