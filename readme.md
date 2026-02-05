# SigmaClip.jl

> [!NOTE]
> The algorithmic logic implemented in this module is inspired by the sigma_clip implementation found in the Astropy Python library.

  
SigmaClip.jl provides a lightweight, highly efficient, and robust toolset for identifying and rejecting outliers in data arrays using iterative sigma clipping. It is designed to be numerically stable, handling NaN and Inf values gracefully while offering performance optimizations for high-throughput data processing.

## Overview

The algorithm works by calculating robust statistics (defaulting to the median and standard deviation) on the data. It rejects values that are more than a specified number of standard deviations from the center, re-calculates the statistics on the remaining data, and repeats the process until convergence or a maximum number of iterations is reached.

#### Key Features

- Iterative Refinement: Automatically tightens bounds over multiple passes.
- Robustness: Explicitly handles and ignores non-finite values (NaN, Inf) during statistical calculation.
- Dual Modes: Supports both non-destructive masking and in-place modification.
- Memory Efficient: Supports pre-allocated buffers to minimize garbage collection overhead in hot loops.
  

## Usage

## 1. Basic Clipping (Mask Generation)

Use sigma_clip to identify outliers without altering the original dataset. This returns a BitArray mask.
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

Use sigma_clip! to modify the array directly. Identified outliers are overwritten with NaN. Requirement: The input array must be of type AbstractFloat.
```Julia

data = randn(100)

data[50] = 100.0 # High outlier

data[51] = -50.0 # Low outlier

sigma_clip!(data)

```

## 3. Custom Statistics

The default behavior uses the fast_median for the center and standard deviation for dispersion. You can provide custom reducers (e.g., using mean for faster but less robust centering).
```Julia

# Use Mean centering instead of Median
mask = sigma_clip(data, cent_reducer=mean, std_reducer=std)

#use any other stats function you want
```

  
### Performance Optimization

When applying sigma clipping to thousands of arrays (e.g., rows of an image or time-series data), memory allocation can become a bottleneck. You can eliminate allocations by pre-allocating a reuseable buffer:

```Julia
# Allocate the buffer once
buf = Vector{Float64}(undef, length(my_data))

for i in 1:length(my_data)
	# Pass the buffer to avoid re-allocating memory every iteration
	mask = sigma_clip!(my_data[i], buf)
	
end
```

### Configuration (Keywords)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `sigma_lower` | `3` | Tolerance (number of std devs) below the central value. |
| `sigma_upper` | `3` | Tolerance (number of std devs) above the central value. |
| `maxiter` | `5` | Maximum number of iterations. |
| `cent_reducer` | `fast_median!` | Function to calculate the central tendency. |
| `std_reducer` | `std` | Function to calculate the dispersion. |
| `mask` | `nothing` | Optional initial mask (`true` indicates a value to ignore). |
| `bad` | `true` | The boolean value representing a "bad" datum in the input mask. |



# License

This project is licensed under the MIT License.