[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
# SigmaClip.jl

  
> [!NOTE]
> The algorithmic logic implemented in this module is inspired by the sigma_clip implementation found in the Astropy Python library.
  
SigmaClip.jl provides a lightweight, highly efficient, and robust toolset for identifying and rejecting outliers in generic data arrays using iterative sigma clipping. It is designed to be numerically stable, handling NaN and Inf values gracefully while offering performance optimizations for high-throughput data processing.

  

## Overview

The algorithm works by calculating robust statistics (defaulting to the median and standard deviation) on the data. It rejects values that are more than a specified number of standard deviations from the center, re-calculates the statistics on the remaining data, and repeats the process until convergence or a maximum number of iterations is reached.

  
  

## Usage


## 1. Basic Clipping (Mask Generation)

Use sigma_clip_mask to identify outliers without altering the original dataset. This returns a BitArray mask.

```Julia

using SigmaClip

data = randn(100)

data[50] = 100.0 # High outlier
data[51] = -50.0 # Low outlier

mask = sigma_clip_mask(data, sigma_lower=3, sigma_upper=3)

clean_data = data[.!mask]

println("Original size: $(length(data))")
println("Cleaned size: $(length(clean_data))")

```

  
  

## 2. In-Place Modification

Use sigma_clip! to modify the array directly. Identified outliers are overwritten with NaN. Requirement: 

> [!WARNING]
> **Data Types**
> The input array must be of type AbstractFloat.
```Julia

data = randn(100)

data[50] = 100.0 # High outlier
data[51] = -50.0 # Low outlier

sigma_clip!(data)

data[50]#NaN
```

  

## 3. Custom Statistics

The default behavior uses the fast_median for the center and standard deviation for dispersion. You can provide custom reducers (e.g., using mean for faster but less robust centering).

```Julia
# Use Mean centering instead of Median
mask = sigma_clip(data, cent_reducer=mean, std_reducer=std)
#use any other stats function you want
```

  
## 4. Compute bounds directly

If you only need to know the final convergence thresholds (lower and upper limits) without modifying the data or generating a full mask, you can use the internal function `sigma_clip_bounds`.

```julia
data = randn(100)
data[50] = 100.0 

# Returns a tuple: (lower_bound, upper_bound)
lb, ub = SigmaClip.sigma_clip_bounds(data, sigma_lower=3, sigma_upper=3)
# just like the other function, it support masks and buffers via keywords arguments

println("Outliers are values < $lb or > $ub")
```

### Performance Optimization

When applying sigma clipping to thousands of arrays (e.g., rows of an image or time-series data), memory allocation can become a bottleneck. You can eliminate allocations by pre-allocating a reusable buffer.

The core algorithms in `SigmaClip.jl` are designed to be **allocation-free** when a pre-allocated buffer is provided. Ideally, if a buffer is passed, all internal calculations (including sorting for median determination and statistical reductions) reuse this memory. This ensures **zero dynamic memory allocation** during execution, eliminating Garbage Collection (GC) overhead. This is particularly critical when processing large datasets, such as iterating over thousands of image rows or time-series segments.

  

```Julia

# 1. Allocate the buffer once (must be same length as the data being processed) 
buf = Vector{Float64}(undef, length(my_data_rows[1])) 
# 2. Hot loop 
for i in eachindex(my_data_rows) 
# Pass the buffer 'buf' to avoid re-allocating memory every iteration. 
# The function reuses 'buf' for internal sorting and stats. 
	sigma_clip!(my_data_rows[i], buf) 
end

```

  

### Configuration (Keywords)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `sigma_lower` | `3` | Tolerance (number of std devs) below the central value. |
| `sigma_upper` | `3` | Tolerance (number of std devs) above the central value. |
| `maxiter` | `5` | Maximum number of iterations. |
| `cent_reducer` | `fast_median!` | Function to calculate the central statistic. |
| `std_reducer` | `std` | Function to calculate the dispersion. |
| `mask` | `nothing` | Optional initial mask (`true` indicates a value to ignore). |
  
  
# License

This project is licensed under the MIT License.