SigmaClip.jl

    Note: The algorithmic logic implemented in this module is inspired by the sigma_clip implementation found in the Astropy Python library.

SigmaClip.jl provides a lightweight, highly efficient, and robust toolset for identifying and rejecting outliers in data arrays using iterative sigma clipping. It is designed to be numerically stable, handling NaN and Inf values gracefully while offering performance optimizations for high-throughput data processing.
Overview

The algorithm works by calculating robust statistics (defaulting to the median and standard deviation) on the data. It rejects values that are more than a specified number of standard deviations from the center, re-calculates the statistics on the remaining data, and repeats the process until convergence or a maximum number of iterations is reached.
Key Features

    Iterative Refinement: Automatically tightens bounds over multiple passes.

    Robustness: Explicitly handles and ignores non-finite values (NaN, Inf) during statistical calculation.

    Dual Modes: Supports both non-destructive masking and in-place modification.

    Memory Efficient: Supports pre-allocated buffers to minimize garbage collection overhead in hot loops.

Usage
1. Basic Clipping (Mask Generation)

Use sigma_clip to identify outliers without altering the original dataset. This returns a BitArray mask.
Julia

using SigmaClip, Statistics

# Generate data with artificial outliers
data = randn(100)
data[50] = 100.0  # High outlier
data[51] = -50.0  # Low outlier

# Perform clipping (default: 3-sigma, 5 iterations)
mask = sigma_clip(data, sigma_lower=3, sigma_upper=3)

# Filter the valid data
clean_data = data[.!mask]

println("Original size: $(length(data))")
println("Cleaned size:  $(length(clean_data))")

2. In-Place Modification

Use sigma_clip! to modify the array directly. Identified outliers are overwritten with NaN. Requirement: The input array must be of type AbstractFloat.
Julia

data = Float64[1.0, 2.0, 1.1, 500.0, 0.9]

# Replace outliers with NaN in-place
sigma_clip!(data)

# 'data' is now: [1.0, 2.0, 1.1, NaN, 0.9]

3. Custom Statistics

The default behavior uses the median for the center and standard deviation for dispersion. You can provide custom reducers (e.g., using mean for faster but less robust centering).
Julia

# Use Mean centering instead of Median
mask = sigma_clip(data, cent_reducer=mean, std_reducer=std)

API Reference
Functions
sigma_clip(x; kwargs...) -> BitArray

Performs sigma clipping and returns a boolean mask where true indicates an outlier.
sigma_clip!(x; kwargs...) -> Vector

Performs sigma clipping and modifies x in-place, setting outliers to NaN.
Configuration (Keywords)
Argument	Default	Description
sigma_lower	3	Tolerance (number of std devs) below the central value.
sigma_upper	3	Tolerance (number of std devs) above the central value.
maxiter	5	Maximum number of iterations.
cent_reducer	fast_median!	Function to calculate the central tendency.
std_reducer	std	Function to calculate the dispersion.
mask	nothing	Optional initial mask (true indicates a value to ignore).
bad	true	The boolean value representing a "bad" datum in the input mask.
buffer	nothing	A pre-allocated vector of size length(x) for internal calculations.
Performance Optimization

When applying sigma clipping to thousands of arrays (e.g., rows of an image or time-series data), memory allocation can become a bottleneck. You can eliminate allocations by pre-allocating a reuseable buffer:
Julia

# Allocate the buffer once
buf = Vector{Float64}(undef, length(data_row))

for i in 1:nrows
    # Pass the buffer to avoid re-allocating memory every iteration
    mask = sigma_clip(matrix[i, :], buffer=buf)
end

License

This project is licensed under the MIT License.