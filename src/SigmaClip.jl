module SigmaClip

include("workspace.jl")
include("stats.jl")

export sigma_clip_mask, sigma_clip_mask!, sigma_clip!, sigma_clip, sigma_clip_bounds
export sigma_clipped_stats
export SigmaClipWorkspace
export fast_median!, mad_std!

validate_sigma(s::Real) =
    isfinite(s) && s > zero(s) ? nothing : throw(ArgumentError("sigma must be finite and positive"))
validate_axes(a::AbstractArray{Bool}, x::AbstractArray) =
    axes(a) == axes(x) ? nothing : throw(DimensionMismatch("arrays must have the same axes"))
validate_maxiter(n::Integer) =
    n == -1 || n > 0 ? nothing : throw(ArgumentError("maxiter must be -1 or positive"))

@inline function pack_valid!(buf::AbstractVector, x::AbstractArray, ::Nothing)
    n = 0
    @inbounds for value in x
        if isfinite(value)
            n += 1
            buf[n] = value
        end
    end
    return n
end

@inline function pack_valid!(buf::AbstractVector, x::AbstractArray, exclude::AbstractArray{Bool})
    n = 0
    @inbounds for i in eachindex(x, exclude)
        value = x[i]
        if !exclude[i] && isfinite(value)
            n += 1
            buf[n] = value
        end
    end
    return n
end

function sigma_clip_compact_unsafe(
        x::AbstractArray{T},
        exclude::M,
        ws::WS,
        sigma_lower::Real,
        sigma_upper::Real,
        center::C,
        spread::S,
        maxiter::Int,
    ) where {T, M, WS, C, S}
    buf = workspace_buffer(ws)
    current = pack_valid!(buf, x, exclude)
    current == 0 && throw(ArgumentError("no valid data to clip"))
    iter = 0

    while true
        c, s = compute_stats(center, spread, current, ws)
        lower_bound = c - s * sigma_lower
        upper_bound = c + s * sigma_upper

        new_count = 0
        @inbounds for i in 1:current
            value = buf[i]
            if lower_bound <= value <= upper_bound
                new_count += 1
                buf[new_count] = value
            end
        end

        new_count == current && return lower_bound, upper_bound, current
        current = new_count
        iter += 1

        (maxiter != -1 && iter >= maxiter) && return lower_bound, upper_bound, current
        current < 2 && return lower_bound, upper_bound, current
    end
    return
end

function prepare_clip_workspace(
        x::AbstractArray,
        exclude,
        workspace,
        sigma_lower::Real,
        sigma_upper::Real,
        spread,
        maxiter::Int,
    )
    !isnothing(exclude) && validate_axes(exclude, x)
    validate_maxiter(maxiter)
    validate_sigma(sigma_lower)
    validate_sigma(sigma_upper)
    return prepare_ws(x, spread, workspace)
end

function sigma_clip_compact(
        x::AbstractArray{T},
        exclude::M,
        workspace::WS,
        sigma_lower::Real,
        sigma_upper::Real,
        center::C,
        spread::S,
        maxiter::Int,
    ) where {T, M, WS, C, S}
    ws = prepare_clip_workspace(
        x, exclude, workspace, sigma_lower, sigma_upper, spread, maxiter
    )
    return sigma_clip_compact_unsafe(
        x, exclude, ws, sigma_lower, sigma_upper, center, spread, maxiter
    )
end

"""
    sigma_clipped_stats(x, sigma; kwargs...) -> NamedTuple
    sigma_clipped_stats(x, sigma_lower, sigma_upper; kwargs...) -> NamedTuple

Clip `x` and calculate statistics on the values retained in the internal
workspace buffer.  The `center` and `spread` callables used for clipping are
also returned as the `center` and `spread` fields.  Additional statistics can
be requested with symbol-keyed pairs, for example:

```julia
sigma_clipped_stats(
    data, 3;
    :median => fast_median!,
    :madstd => mad_std!,
)
```

The result is a `NamedTuple` with `center` and `spread` fields followed by the
requested pair fields.  Each pair callable receives the compacted buffer view
as its only argument.  The input array is not modified.

The thresholds are required and must be finite and positive.
`sigma_clipped_stats(x, sigma)` is shorthand for
`sigma_clipped_stats(x, sigma, sigma)`.  Keywords: `workspace=nothing`,
`exclude=nothing`, `center=fast_median!`, `spread=mad_std!`, and `maxiter=5`.
Use `maxiter=-1` to run until convergence.  Any other keyword is taken as a
statistic pair.
"""
function sigma_clipped_stats(
        x::AbstractArray{T},
        sigma_lower::Real,
        sigma_upper::Real;
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5,
        statistics...,
    ) where {T, WS, C, S}
    ws = prepare_clip_workspace(
        x, exclude, workspace, sigma_lower, sigma_upper, spread, maxiter
    )
    _, _, n = sigma_clip_compact_unsafe(
        x, exclude, ws, sigma_lower, sigma_upper, center, spread, maxiter
    )

    center_value, spread_value = compute_stats(center, spread, n, ws)
    data = buffer_view(ws, n)
    stat_values = Tuple(f(data) for f in Base.values(statistics))
    extra = NamedTuple{keys(statistics)}(stat_values)
    return merge((center = center_value, spread = spread_value), extra)
end

sigma_clipped_stats(x::AbstractArray, sigma::Real; kw...) =
    sigma_clipped_stats(x, sigma, sigma; kw...)

sigma_clipped_stats(::AbstractArray; _kw...) = throw(
    ArgumentError(
        "sigma_clipped_stats requires sigma thresholds: " *
            "sigma_clipped_stats(x, sigma) or sigma_clipped_stats(x, sigma_lower, sigma_upper)"
    )
)

"""
    sigma_clip_bounds(x, sigma; kwargs...) -> (lower, upper)
    sigma_clip_bounds(x, sigma_lower, sigma_upper; kwargs...) -> (lower, upper)

Return the final iterative sigma-clipping bounds without modifying `x`.
Non-finite values and entries marked by `exclude` are ignored while estimating
the bounds.

The thresholds are required and must be finite and positive.
`sigma_clip_bounds(x, sigma)` is shorthand for
`sigma_clip_bounds(x, sigma, sigma)`.  Keywords: `workspace=nothing`,
`exclude=nothing`, `center=fast_median!`, `spread=mad_std!`, and `maxiter=5`.
Use `maxiter=-1` to run until convergence.
"""
function sigma_clip_bounds(
        x::AbstractArray{T},
        sigma_lower::Real,
        sigma_upper::Real;
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5,
    ) where {T, WS, C, S}
    lower, upper, _ = sigma_clip_compact(
        x, exclude, workspace, sigma_lower, sigma_upper, center, spread, maxiter
    )
    return lower, upper
end

sigma_clip_bounds(x::AbstractArray, sigma::Real; kw...) =
    sigma_clip_bounds(x, sigma, sigma; kw...)

sigma_clip_bounds(::AbstractArray; _kw...) = throw(
    ArgumentError(
        "sigma_clip_bounds requires sigma thresholds: " *
            "sigma_clip_bounds(x, sigma) or sigma_clip_bounds(x, sigma_lower, sigma_upper)"
    )
)

"""
    sigma_clip_mask(x, sigma; kwargs...) -> BitArray
    sigma_clip_mask(x, sigma_lower, sigma_upper; kwargs...) -> BitArray

Return a mask where `true` marks finite values inside the final clipping bounds.
`sigma_clip_mask(x, sigma)` is shorthand for `sigma_clip_mask(x, sigma, sigma)`.
Accepts the same thresholds and keywords as [`sigma_clip_bounds`](@ref).
"""
sigma_clip_mask(x::AbstractArray, sigma_lower::Real, sigma_upper::Real; kw...) =
    # Preserve the input axes while retaining the documented packed mask type.
    sigma_clip_mask!(x, similar(BitArray, axes(x)), sigma_lower, sigma_upper; kw...)

sigma_clip_mask(x::AbstractArray, sigma::Real; kw...) =
    sigma_clip_mask(x, sigma, sigma; kw...)

sigma_clip_mask(::AbstractArray; _kw...) = throw(
    ArgumentError(
        "sigma_clip_mask requires sigma thresholds: " *
            "sigma_clip_mask(x, sigma) or sigma_clip_mask(x, sigma_lower, sigma_upper)"
    )
)

"""
    sigma_clip_mask!(x, target, sigma; kwargs...) -> target
    sigma_clip_mask!(x, target, sigma_lower, sigma_upper; kwargs...) -> target

Write the validity mask into `target`, which must have the same axes as `x`.
`sigma_clip_mask!(x, target, sigma)` is shorthand for
`sigma_clip_mask!(x, target, sigma, sigma)`. Accepts the same thresholds and
keywords as [`sigma_clip_bounds`](@ref).
"""
function sigma_clip_mask!(
        x::AbstractArray{T},
        target::AbstractArray{Bool},
        sigma_lower::Real,
        sigma_upper::Real;
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5,
    ) where {T, WS, C, S}
    validate_axes(target, x)
    lower, upper = sigma_clip_bounds(
        x, sigma_lower, sigma_upper; workspace, exclude, center, spread, maxiter
    )

    @inbounds for i in eachindex(x, target)
        target[i] = lower <= x[i] <= upper
    end
    return target
end

sigma_clip_mask!(x::AbstractArray, target::AbstractArray{Bool}, sigma::Real; kw...) =
    sigma_clip_mask!(x, target, sigma, sigma; kw...)

sigma_clip_mask!(::AbstractArray, ::AbstractArray{Bool}; _kw...) = throw(
    ArgumentError(
        "sigma_clip_mask! requires sigma thresholds: " *
            "sigma_clip_mask!(x, target, sigma) or sigma_clip_mask!(x, target, sigma_lower, sigma_upper)"
    )
)

"""
    sigma_clip!(x, sigma; kwargs...) -> x
    sigma_clip!(x, sigma_lower, sigma_upper; kwargs...) -> x

Replace non-finite values and outliers in `x` with `NaN`. The element type must
represent `NaN`. `sigma_clip!(x, sigma)` is shorthand for
`sigma_clip!(x, sigma, sigma)`. Accepts the same thresholds and keywords as
[`sigma_clip_bounds`](@ref).
"""
function sigma_clip!(
        x::AbstractArray{T},
        sigma_lower::Real,
        sigma_upper::Real;
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5,
    ) where {T <: Number, WS, C, S}
    lower, upper = sigma_clip_bounds(
        x, sigma_lower, sigma_upper; workspace, exclude, center, spread, maxiter
    )
    nan = T(NaN)

    @inbounds for i in eachindex(x)
        lower <= x[i] <= upper || (x[i] = nan)
    end
    return x
end

sigma_clip!(::AbstractArray{<:Integer}, ::Real, ::Real; _kw...) = throw(
    ArgumentError("sigma_clip! cannot write NaN to an integer array; use sigma_clip instead")
)

sigma_clip!(x::AbstractArray, sigma::Real; kw...) = sigma_clip!(x, sigma, sigma; kw...)

sigma_clip!(::AbstractArray; _kw...) = throw(
    ArgumentError(
        "sigma_clip! requires sigma thresholds: " *
            "sigma_clip!(x, sigma) or sigma_clip!(x, sigma_lower, sigma_upper)"
    )
)

"""
    sigma_clip(x, sigma; kwargs...)
    sigma_clip(x, sigma_lower, sigma_upper; kwargs...)

Return a clipped copy of `x`; integer inputs are converted to floating point.
`sigma_clip(x, sigma)` is shorthand for `sigma_clip(x, sigma, sigma)`.
Accepts the same thresholds and keywords as [`sigma_clip_bounds`](@ref).
"""
sigma_clip(x::AbstractArray{<:Number}, sigma_lower::Real, sigma_upper::Real; kw...) =
    sigma_clip!(copy(x), sigma_lower, sigma_upper; kw...)
sigma_clip(x::AbstractArray{<:Integer}, sigma_lower::Real, sigma_upper::Real; kw...) =
    sigma_clip!(float.(x), sigma_lower, sigma_upper; kw...)

sigma_clip(x::AbstractArray, sigma::Real; kw...) = sigma_clip(x, sigma, sigma; kw...)

sigma_clip(::AbstractArray; _kw...) = throw(
    ArgumentError(
        "sigma_clip requires sigma thresholds: " *
            "sigma_clip(x, sigma) or sigma_clip(x, sigma_lower, sigma_upper)"
    )
)

end
