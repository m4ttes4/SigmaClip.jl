module SigmaClip

# TODO mask only affects bounds
# TODO make buffer args only 
using Statistics

const BAD_PIXEL = true #it marks a flagged pixel
const GOOD_PIXEL = false

export sigma_clip_mask, sigma_clip_mask!, sigma_clip!, sigma_clip

"""
    sigma_clip_mask(x::AbstractArray; kwargs...) -> BitArray

Performs iterative sigma clipping on array `x` to identify outliers. 
Returns a boolean mask (`BitArray`) where `bad` (default true) indicates that the corresponding value is an outlier 
(or was already masked in the input).

The algorithm iteratively calculates central and dispersion statistics on a subset of "good" data, 
narrowing the bounds at each step until convergence or until `maxiter` is reached.

# Positional Arguments
- `x`: The input data array (not modified).
- `buffer` (optional): A pre-allocated vector of length `length(x)` used for internal calculations to reduce memory allocations.

# Keyword Arguments
- `mask` (optional): An initial boolean mask of the same size as `x`.
- `sigma_lower::Real`: Number of standard deviations below the central value for clipping (default: 3).
- `sigma_upper::Real`: Number of standard deviations above the central value for clipping (default: 3).
- `cent_reducer::Function`: Function to compute central tendency (default: `fast_median!`).
- `std_reducer::Function`: Function to compute dispersion (default: `std`).
- `maxiter::Int`: Maximum number of clipping iterations (default: 5).

## NOTE: 
mask is used only to mask some values when computing the bounds!

# Example
```julia
data = randn(100)
data[50] = 100.0 # Artificial outlier
mask = sigma_clip_mask(data, sigma_lower=3.0, sigma_upper=3.0)
clean_data = data[.!mask]
```
"""
sigma_clip_mask(x, args...; kwargs...) = begin 
    
    target = trues(size(x))
    
    sigma_clip_mask!(x, target, args...; kwargs...)
end
    

function sigma_clip_mask!(x::AbstractArray{T, N},
    target::AbstractArray{Bool},
    buffer::B=nothing;
    mask::M = nothing,
    sigma_lower=3,
    sigma_upper=3,
    cent_reducer::F1=fast_median!,
    std_reducer::F2=std,
    maxiter::Int=5) where {T,F1,F2,B, M, N}


    if isnothing(buffer)
        buffer = Vector{T}(undef, length(x))
    end

    
    lb, up = sigma_clip_bounds(x, mask, buffer, float(sigma_lower), float(sigma_upper), cent_reducer, std_reducer, maxiter)

    @inbounds for i in eachindex(x)
        val = x[i]

        is_outlier = ismissing(val) || !isfinite(val) || val < lb || val > up

        if is_outlier
            target[i] = BAD_PIXEL
        else
            target[i] = GOOD_PIXEL
        end
    end

    return target
end

# TODO add support for units as extension via unitless
""" 
sigma_clip!(x::AbstractArray{<:AbstractFloat}; kwargs...) -> x

In-place version of sigma_clip.

Performs sigma clipping and directly modifies the array x, replacing identified outliers with NaN. Unlike the standard version, this function requires x to be an array of floating-point numbers (AbstractFloat).
Notes

    Values that are NaN in the input are ignored during statistics calculation but remain NaN.

    Values identified as outliers are overwritten with NaN.

Arguments and Keywords

Accepts the same arguments as sigma_clip.
"""
function sigma_clip!(x::AbstractArray{T,N},
    buffer::B=nothing;
    mask::Union{Nothing,AbstractArray{Bool, N}}=nothing,
    sigma_lower=3,
    sigma_upper=3,
    cent_reducer::F1=fast_median!,
    std_reducer::F2=std,
    maxiter::Int=5) where {T<:AbstractFloat,F1,F2,B, N} 

    if isnothing(buffer)
        buffer = Vector{T}(undef, length(x))
    end

    lb, up = sigma_clip_bounds(x, mask, buffer, float(sigma_lower), float(sigma_upper), cent_reducer, std_reducer, maxiter)


    @inbounds for i in eachindex(x)
        val = x[i]

        is_outlier = ismissing(val) || !isfinite(val) || val < lb || val > up


        if is_outlier
            x[i] = T(NaN)
        end
    end

    return x
end



sigma_clip(x::AbstractArray{T}, args...; kwargs...) where {T <: AbstractFloat} = sigma_clip!(copy(x), args...; kwargs...)
sigma_clip(x::AbstractArray{T}, args...; kwargs...) where {T <: Integer} = sigma_clip!(float.(x), args...; kwargs...)


sigma_clip_bounds(x::AbstractArray{T}; 
mask = nothing,
buffer = Vector{T}(undef, length(x)),
sigma_lower = 3,
sigma_upper = 3,
cent_reducer = fast_median!,
std_reducer = std,
maxiter = 5) where T = sigma_clip_bounds(x, mask, buffer,
float(sigma_lower), 
float(sigma_upper), 
cent_reducer,
std_reducer,
maxiter)

# TODO make this function return the buffer so we can compute mean, median and std ?

function sigma_clip_bounds(
    x::AbstractArray{T},
    mask::Union{Nothing,AbstractArray{Bool}},
    buffer::AbstractVector{T},
    sigma_lower::L,
    sigma_upper::L,
    cent_reducer::F1,
    std_reducer::F2,
    maxiter::Int,
) where {T,L,F1,F2}

    N = 0
    have_mask = !isnothing(mask)


    @inbounds for i in eachindex(x)
        is_valid = (!have_mask || mask[i] != BAD_PIXEL) && isfinite(x[i]) && !ismissing(x[i])

        if is_valid
            N += 1
            buffer[N] = x[i]
        end
    end

    if N == 0
        return T(0), T(0)
    end

    iter = 0
    current_count = N


    lower_bound = T(0)
    upper_bound = T(0)

    while true
        working_view = view(buffer, 1:current_count)

        c = cent_reducer(working_view)
        s = std_reducer(working_view)

        lower_bound = c - s * sigma_lower
        upper_bound = c + s * sigma_upper

        new_count = 0
        @inbounds for i in 1:current_count
            val = buffer[i]
            if val >= lower_bound && val <= upper_bound
                new_count += 1
                buffer[new_count] = val
            end
        end


        if new_count == current_count
            return lower_bound, upper_bound
        end

        current_count = new_count
        iter += 1


        if maxiter != -1 && iter >= maxiter
            return lower_bound, upper_bound
        end


        if current_count == 0
            return lower_bound, upper_bound
        end
    end
end



function _kth_smallest!(a::AbstractVector{T}, k::Int) where T
    n = length(a)
    l = 1
    m = n

    @inbounds while l < m
        x = a[k]
        i = l
        j = m
        while true
            while a[i] < x
                i += 1
            end
            while x < a[j]
                j -= 1
            end

            if i <= j
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
            end
            if i > j
                break
            end
        end
        if j < k
            l = i
        end
        if k < i
            m = j
        end
    end
    return a[k]
end

function fast_median!(a::AbstractVector{T}) where T
    n = length(a)

    if n == 0
        return T(0)
    end 

    if iseven(n)
        m1 = _kth_smallest!(a, n ÷ 2)
        m2 = _kth_smallest!(a, (n ÷ 2) + 1)
        return 0.5 * (m1 + m2)
    else
        return _kth_smallest!(a, (n + 1) ÷ 2)
    end
end

end # module SigmaClip