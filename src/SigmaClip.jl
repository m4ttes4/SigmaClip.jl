module SigmaClip

# ─── Statistics reducers ─────────────────────────────────────────────────────

include("workspace.jl")
include("stats.jl")

export sigma_clip_mask, sigma_clip_mask!, sigma_clip!, sigma_clip, sigma_clip_bounds
export SigmaClipWorkspace
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

function sigma_clip_compact_unsafe(
        x::AbstractArray{T},
        exclude::M,
        ws::WS,
        sigma_lower::Real,
        sigma_upper::Real,
        center::C,
        spread::S,
        maxiter::Int,
    ) where {T, M, C, S, WS}

    

    # TODO better to specialize on this?
    buf = workspace_buffer(ws)

    n = pack_valid!(buf, x, exclude)
    n == 0 && throw(ArgumentError("No valid data to clip found"))

    current = n
    iter = 0

    while true
        c, s = compute_stats(center, spread, current, ws)

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

        new_count == current && return (lower_bound, upper_bound, current)
        current = new_count
        iter += 1

        (maxiter != -1 && iter >= maxiter) && return (lower_bound, upper_bound, current)
        current < 2 && return (lower_bound, upper_bound, current)
    end
end


function sigma_clip_compact(x::AbstractArray{T},
        exclude::M,
        workspace::WS,
        sigma_lower::Real,
        sigma_upper::Real,
        center::C,
        spread::S,
        maxiter::Int,
    ) where {T, M, C, S, WS}
    !isnothing(exclude) && validate_axes(exclude, x)

    ws = prepare_ws(x, spread, workspace)

    validate_maxiter(maxiter)

    validate_sigma(sigma_lower)
    validate_sigma(sigma_upper)

    return sigma_clip_compact_unsafe(x, exclude, ws,
        sigma_lower, sigma_upper,
        center, spread, maxiter)
end

@inline function sigma_clip_bounds(
        x::AbstractArray{T},
        workspace::WS,
        exclude::Union{Nothing, AbstractArray{Bool}},
        sigma_lower,
        sigma_upper,
        center::C,
        spread::S,
        maxiter::Int,
    ) where {T, C, S, WS}

    lb, up, _ = sigma_clip_compact(
        x, exclude, workspace,
        sigma_lower, sigma_upper,
        center, spread, maxiter
    )
    return (lb, up)
end

"""
    sigma_clip_bounds(x::AbstractArray; kwargs...) -> (lower, upper)

Run iterative sigma clipping on `x` and return the final lower and upper bounds.

This function computes the same convergence bounds used by [`sigma_clip_mask`](@ref),
[`sigma_clip_mask!`](@ref), [`sigma_clip`](@ref), and [`sigma_clip!`](@ref), but
does not allocate or return a validity mask and does not modify `x`.

Non-finite values are ignored while estimating the bounds. Values marked by
`exclude` are also ignored while estimating the bounds.

# Arguments
- `x::AbstractArray`: numeric input data used to estimate sigma-clipping bounds.

# Keywords
- `workspace=nothing`: optional pre-allocated workspace. Pass a
  [`SigmaClipWorkspace`](@ref) or a custom workspace implementing
  `SigmaClip.workspace_buffer` and `SigmaClip.workspace_auxbuffer`.
- `exclude::Union{Nothing, AbstractArray{Bool}}=nothing`: optional boolean array
  with the same axes as `x`. Entries set to `true` are excluded from bound
  estimation.
- `sigma_lower::Real=3`: lower sigma threshold. Must be finite and strictly
  positive.
- `sigma_upper::Real=3`: upper sigma threshold. Must be finite and strictly
  positive.
- `center=fast_median!`: center estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `spread=mad_std!`: dispersion estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `maxiter::Int=5`: maximum number of sigma-clipping iterations. Use `-1` to run
  until convergence.

# Returns
- `(lower, upper)`: tuple containing the final lower and upper clipping bounds.

# Throws
- `ArgumentError`: if no finite, non-excluded data are available, if a sigma
  threshold is not finite and strictly positive, or if `maxiter` is `0` or less
  than `-1`.
- `DimensionMismatch`: if `exclude` does not have the same axes as `x`.

# Examples
```julia
using SigmaClip

data = [1.0, 1.0, 1.0, 50.0]
lower, upper = sigma_clip_bounds(data)

(lower, upper)
# (1.0, 1.0)
```

Use the bounds to classify values manually:

```julia
data = [0.0, 0.0, 0.0, 10.0]
lower, upper = sigma_clip_bounds(data)

valid = isfinite.(data) .& (lower .<= data .<= upper)
# valid == Bool[true, true, true, false]
```

See also: [`sigma_clip_mask`](@ref), [`sigma_clip`](@ref),
[`sigma_clip!`](@ref).
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

    return sigma_clip_bounds(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )
end
# ─── Public API ───────────────────────────────────────────────────────────────

"""
    sigma_clip_mask(x::AbstractArray; kwargs...) -> BitArray

Run iterative sigma clipping on `x` and return a validity mask.

The returned mask has the same shape as `x`. Each `true` entry marks a finite
value retained by the final sigma-clipping bounds; each `false` entry marks a
non-finite value or an outlier. The input array `x` is not modified.

# Arguments
- `x::AbstractArray`: numeric input data to classify.

# Keywords
- `workspace=nothing`: optional pre-allocated workspace. Pass a
  [`SigmaClipWorkspace`](@ref) or a custom workspace implementing
  `SigmaClip.workspace_buffer` and `SigmaClip.workspace_auxbuffer`.
- `exclude::Union{Nothing, AbstractArray{Bool}}=nothing`: optional boolean array
  with the same axes as `x`. Entries set to `true` are excluded from bound
  estimation, but are still classified against the final bounds.
- `sigma_lower=3`: lower sigma threshold. Must be finite and strictly positive.
- `sigma_upper=3`: upper sigma threshold. Must be finite and strictly positive.
- `center=fast_median!`: center estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `spread=mad_std!`: dispersion estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `maxiter::Int=5`: maximum number of sigma-clipping iterations. Use `-1` to run
  until convergence.

# Returns
- `BitArray`: validity mask where `true` means finite and retained by the final
  bounds.

# Throws
- `ArgumentError`: if no finite, non-excluded data are available, if a sigma
  threshold is not finite and strictly positive, or if `maxiter` is `0` or less
  than `-1`.
- `DimensionMismatch`: if `exclude` does not have the same axes as `x`.

# Examples
```julia
using SigmaClip

data = [1.0, 1.0, 1.0, 50.0, NaN]
mask = sigma_clip_mask(data)
# mask == Bool[true, true, true, false, false]

clean = data[mask]
# clean == [1.0, 1.0, 1.0]
```

Exclude known calibration samples from bound estimation while still classifying
them against the final bounds:

```julia
data = [-100.0, 0.0, 0.0, 0.0, 50.0]
exclude = Bool[true, false, false, false, true]

mask = sigma_clip_mask(data; exclude)
# mask == Bool[false, true, true, true, false]
```

See also: [`sigma_clip_mask!`](@ref), [`sigma_clip`](@ref),
[`sigma_clip_bounds`](@ref).
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

    target = trues(size(x))

    return sigma_clip_mask!(
        x, target;
        workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )
end

"""
    sigma_clip_mask!(x::AbstractArray, target::AbstractArray{Bool}; kwargs...) -> target

Run iterative sigma clipping on `x` and write the validity mask into `target`.

`target` must have the same axes as `x`. Each `true` entry marks a finite value
retained by the final sigma-clipping bounds; each `false` entry marks a
non-finite value or an outlier. The input array `x` is not modified. The
pre-allocated `target` array is overwritten and returned.

# Arguments
- `x::AbstractArray`: numeric input data to classify.
- `target::AbstractArray{Bool}`: output validity mask. Must have the same axes
  as `x`.

# Keywords
- `workspace=nothing`: optional pre-allocated workspace. Pass a
  [`SigmaClipWorkspace`](@ref) or a custom workspace implementing
  `SigmaClip.workspace_buffer` and `SigmaClip.workspace_auxbuffer`.
- `exclude::Union{Nothing, AbstractArray{Bool}}=nothing`: optional boolean array
  with the same axes as `x`. Entries set to `true` are excluded from bound
  estimation, but are still classified against the final bounds.
- `sigma_lower=3`: lower sigma threshold. Must be finite and strictly positive.
- `sigma_upper=3`: upper sigma threshold. Must be finite and strictly positive.
- `center=fast_median!`: center estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `spread=mad_std!`: dispersion estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `maxiter::Int=5`: maximum number of sigma-clipping iterations. Use `-1` to run
  until convergence.

# Returns
- `target`: the same boolean array passed by the caller, filled as a validity
  mask.

# Throws
- `ArgumentError`: if no finite, non-excluded data are available, if a sigma
  threshold is not finite and strictly positive, or if `maxiter` is `0` or less
  than `-1`.
- `DimensionMismatch`: if `target` or `exclude` does not have the same axes as
  `x`.

# Examples
```julia
using SigmaClip

data = [2.0, 2.0, 2.0, 20.0]
target = falses(length(data))

result = sigma_clip_mask!(data, target)

result === target
# true

target
# 4-element BitVector:
#  1
#  1
#  1
#  0
```

Use a pre-allocated workspace in repeated calls:

```julia
data = [1.0, 1.0, 1.0, 99.0]
target = falses(length(data))
workspace = SigmaClipWorkspace(Vector{Float64}(undef, length(data)),
                               Vector{Float64}(undef, length(data)))

sigma_clip_mask!(data, target; workspace)
# target == Bool[true, true, true, false]
```

See also: [`sigma_clip_mask`](@ref), [`sigma_clip!`](@ref),
[`sigma_clip_bounds`](@ref).
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
    
    lb, ub = sigma_clip_bounds(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )

    @inbounds for i in eachindex(x)
        val = x[i]
        target[i] = is_valid_data(val, ub, lb)#isfinite(val) && val >= lb && val <= ub
    end
    return target
end

"""
    sigma_clip!(x::AbstractArray{<:Number}; kwargs...) -> x

Run iterative sigma clipping in place, replacing invalid entries with `NaN`.

Each non-finite value and each outlier outside the final sigma-clipping bounds is
replaced with `NaN`. Values retained by the final bounds are left unchanged. The
input array `x` is modified and returned.

`x` must have an element type that can represent `NaN`. For integer arrays, use
[`sigma_clip`](@ref), which returns a floating-point copy.

# Arguments
- `x::AbstractArray{<:Number}`: numeric input data to modify in place.

# Keywords
- `workspace=nothing`: optional pre-allocated workspace. Pass a
  [`SigmaClipWorkspace`](@ref) or a custom workspace implementing
  `SigmaClip.workspace_buffer` and `SigmaClip.workspace_auxbuffer`.
- `exclude::Union{Nothing, AbstractArray{Bool}}=nothing`: optional boolean array
  with the same axes as `x`. Entries set to `true` are excluded from bound
  estimation, but are still classified against the final bounds.
- `sigma_lower=3`: lower sigma threshold. Must be finite and strictly positive.
- `sigma_upper=3`: upper sigma threshold. Must be finite and strictly positive.
- `center=fast_median!`: center estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `spread=mad_std!`: dispersion estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `maxiter::Int=5`: maximum number of sigma-clipping iterations. Use `-1` to run
  until convergence.

# Returns
- `x`: the same array passed by the caller, with non-finite values and outliers
  replaced by `NaN`.

# Throws
- `ArgumentError`: if `x` has an integer element type, if no finite,
  non-excluded data are available, if a sigma threshold is not finite and
  strictly positive, or if `maxiter` is `0` or less than `-1`.
- `DimensionMismatch`: if `exclude` does not have the same axes as `x`.

# Examples
```julia
using SigmaClip

data = [1.0, 1.0, 1.0, 40.0]
result = sigma_clip!(data)

result === data
# true

isnan(data[end])
# true
```

Use asymmetric sigma thresholds:

```julia
data = [-20.0, 0.0, 0.0, 0.0, 5.0]

sigma_clip!(data; sigma_lower = 2, sigma_upper = 4)
```

Use standard deviation instead of the default MAD-based spread:

```julia
using Statistics
using SigmaClip

data = [1.0, 1.0, 1.0, 8.0]
sigma_clip!(data; center = mean, spread = std)
```

See also: [`sigma_clip`](@ref), [`sigma_clip_mask!`](@ref),
[`sigma_clip_bounds`](@ref).
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

    lb, ub = sigma_clip_bounds(
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
    sigma_clip(x::AbstractArray{<:Number}; kwargs...) -> AbstractArray

Run iterative sigma clipping on a copy of `x`, replacing invalid entries with
`NaN` in the returned array.

The input array `x` is not modified. Each non-finite value and each outlier
outside the final sigma-clipping bounds is replaced with `NaN` in the result.
Values retained by the final bounds are copied unchanged.

For integer inputs, `sigma_clip` first converts `x` with `float.(x)` so the
returned array can represent `NaN`.

# Arguments
- `x::AbstractArray{<:Number}`: numeric input data to copy and sigma-clip.

# Keywords
- `workspace=nothing`: optional pre-allocated workspace. Pass a
  [`SigmaClipWorkspace`](@ref) or a custom workspace implementing
  `SigmaClip.workspace_buffer` and `SigmaClip.workspace_auxbuffer`.
- `exclude::Union{Nothing, AbstractArray{Bool}}=nothing`: optional boolean array
  with the same axes as `x`. Entries set to `true` are excluded from bound
  estimation, but are still classified against the final bounds.
- `sigma_lower=3`: lower sigma threshold. Must be finite and strictly positive.
- `sigma_upper=3`: upper sigma threshold. Must be finite and strictly positive.
- `center=fast_median!`: center estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `spread=mad_std!`: dispersion estimator used at each iteration. It may be any
  callable accepting an `AbstractVector`, or a workspace-aware reducer with a
  `SigmaClip.statistic` method.
- `maxiter::Int=5`: maximum number of sigma-clipping iterations. Use `-1` to run
  until convergence.

# Returns
- `AbstractArray`: a clipped copy of `x`. Integer inputs are returned as a
  floating-point array; non-integer numeric inputs keep the element type of
  `copy(x)`.

# Throws
- `ArgumentError`: if no finite, non-excluded data are available, if a sigma
  threshold is not finite and strictly positive, or if `maxiter` is `0` or less
  than `-1`.
- `DimensionMismatch`: if `exclude` does not have the same axes as `x`.

# Examples
```julia
using SigmaClip

data = [1.0, 1.0, 1.0, 100.0]
clipped = sigma_clip(data)

data
# [1.0, 1.0, 1.0, 100.0]

isnan(clipped[end])
# true
```

Integer inputs are promoted so outliers can be represented as `NaN`:

```julia
data = [2, 2, 2, 50]
clipped = sigma_clip(data)

eltype(clipped)
# Float64

isnan(clipped[end])
# true
```

See also: [`sigma_clip!`](@ref), [`sigma_clip_mask`](@ref),
[`sigma_clip_bounds`](@ref).
"""
sigma_clip(x::AbstractArray{<:Number}; kw...) = sigma_clip!(copy(x); kw...)
sigma_clip(x::AbstractArray{<:Integer}; kw...) = sigma_clip!(float.(x); kw...)

function sigma_clip_stats(
        x::AbstractArray{T},
        functions...;
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower::Real = 3,
        sigma_upper::Real = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5
    ) where {T, C, S, WS}

    !isnothing(exclude) && validate_axes(exclude, x)

    ws = prepare_ws(x, spread, workspace)

    validate_maxiter(maxiter)

    validate_sigma(sigma_lower)
    validate_sigma(sigma_upper)

    _, _, n = sigma_clip_compact_unsafe(
        x, exclude, ws,
        sigma_lower, sigma_upper,
        center, spread, maxiter
    )

    data = collect_buf_from_ws(ws, n)
    stats_functions = isempty(functions) ? (fast_median!, mad_std!) : functions

    res = map(stats_functions) do f
        f(data)
    end
    return res
end


end # module SigmaClip
