module SigmaClip

include("workspace.jl")
include("stats.jl")

export sigma_clip_mask, sigma_clip_mask!, sigma_clip!, sigma_clip, sigma_clip_bounds
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
    ) where {T, M, WS <: SigmaClipWorkspace, C, S}
    buf = ws.buf
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
    !isnothing(exclude) && validate_axes(exclude, x)
    validate_maxiter(maxiter)
    validate_sigma(sigma_lower)
    validate_sigma(sigma_upper)

    ws = prepare_ws(x, spread, workspace)
    return sigma_clip_compact_unsafe(
        x, exclude, ws, sigma_lower, sigma_upper, center, spread, maxiter
    )
end

function sigma_clip_bounds(
        x::AbstractArray{T},
        workspace::WS,
        exclude::Union{Nothing, AbstractArray{Bool}},
        sigma_lower::Real,
        sigma_upper::Real,
        center::C,
        spread::S,
        maxiter::Int,
    ) where {T, WS, C, S}
    lower, upper, _ = sigma_clip_compact(
        x, exclude, workspace, sigma_lower, sigma_upper, center, spread, maxiter
    )
    return lower, upper
end

"""
    sigma_clip_bounds(x; kwargs...) -> (lower, upper)

Return the final iterative sigma-clipping bounds without modifying `x`.
Non-finite values and entries marked by `exclude` are ignored while estimating
the bounds.

Keywords: `workspace=nothing`, `exclude=nothing`, `sigma_lower=3`,
`sigma_upper=3`, `center=fast_median!`, `spread=mad_std!`, and `maxiter=5`.
Use `maxiter=-1` to run until convergence.
"""
function sigma_clip_bounds(
        x::AbstractArray{T};
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower::Real = 3,
        sigma_upper::Real = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5,
    ) where {T, WS, C, S}
    return sigma_clip_bounds(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )
end

"""
    sigma_clip_mask(x; kwargs...) -> BitArray

Return a mask where `true` marks finite values inside the final clipping bounds.
Accepts the same keywords as [`sigma_clip_bounds`](@ref).
"""
function sigma_clip_mask(
        x::AbstractArray{T};
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower::Real = 3,
        sigma_upper::Real = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5,
    ) where {T, WS, C, S}
    target = trues(size(x))
    return sigma_clip_mask!(
        x, target; workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )
end

"""
    sigma_clip_mask!(x, target; kwargs...) -> target

Write the validity mask into `target`, which must have the same axes as `x`.
Accepts the same keywords as [`sigma_clip_bounds`](@ref).
"""
function sigma_clip_mask!(
        x::AbstractArray{T},
        target::AbstractArray{Bool};
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower::Real = 3,
        sigma_upper::Real = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5,
    ) where {T, WS, C, S}
    validate_axes(target, x)
    lower, upper = sigma_clip_bounds(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )

    @inbounds for i in eachindex(x, target)
        target[i] = lower <= x[i] <= upper
    end
    return target
end

"""
    sigma_clip!(x; kwargs...) -> x

Replace non-finite values and outliers in `x` with `NaN`. The element type must
represent `NaN`. Accepts the same keywords as [`sigma_clip_bounds`](@ref).
"""
function sigma_clip!(
        x::AbstractArray{T};
        workspace::WS = nothing,
        exclude::Union{Nothing, AbstractArray{Bool}} = nothing,
        sigma_lower::Real = 3,
        sigma_upper::Real = 3,
        center::C = fast_median!,
        spread::S = mad_std!,
        maxiter::Int = 5,
    ) where {T <: Number, WS, C, S}
    lower, upper = sigma_clip_bounds(
        x, workspace, exclude, sigma_lower, sigma_upper, center, spread, maxiter
    )
    nan = T(NaN)

    @inbounds for i in eachindex(x)
        lower <= x[i] <= upper || (x[i] = nan)
    end
    return x
end

sigma_clip!(::AbstractArray{<:Integer}; _kw...) = throw(
    ArgumentError("sigma_clip! cannot write NaN to an integer array; use sigma_clip instead")
)

"""
    sigma_clip(x; kwargs...)

Return a clipped copy of `x`; integer inputs are converted to floating point.
Accepts the same keywords as [`sigma_clip_bounds`](@ref).
"""
sigma_clip(x::AbstractArray{<:Number}; kw...) = sigma_clip!(copy(x); kw...)
sigma_clip(x::AbstractArray{<:Integer}; kw...) = sigma_clip!(float.(x); kw...)

end
