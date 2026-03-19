[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# SigmaClip.jl

> [!NOTE]
> The algorithmic logic implemented in this module is inspired by the `sigma_clip`
> implementation found in the Astropy Python library.

SigmaClip.jl provides a lightweight, highly efficient, and robust toolset for
identifying and rejecting outliers using iterative sigma clipping. It is
primarily designed for **astrophysical data processing** — cleaning light
curves, removing cosmic rays from CCD frames, rejecting bad pixels in image
stacks — but works equally well on any numeric array: 1-D time series, 2-D
images, or higher-dimensional data.

The library is designed to be numerically stable, handling `NaN` and `Inf`
values gracefully, and offers both a convenient one-liner API and a
zero-allocation path for high-throughput workloads such as iterating over every
row of a multi-extension FITS image.

## Overview

The algorithm works by computing a centre estimate and a dispersion estimate on
the current "good" data, rejecting values that fall outside
`[centre − σ_lower × dispersion, centre + σ_upper × dispersion]`,
then repeating on the surviving values until convergence or until `maxiter`
iterations have been reached.

By default the centre is the **median** (via an allocation-free quickselect)
and the dispersion is the **MAD-std** (Median Absolute Deviation scaled by
1.4826). Both are swappable via keyword arguments.

SigmaClip.jl has **no external dependencies**.

---

## Installation

```julia
using Pkg
Pkg.add("SigmaClip")
```

---

## Quick start

```julia
using SigmaClip

data = randn(1000)
data[42]  = 50.0    # high outlier
data[123] = -30.0   # low outlier

# Out-of-place: returns a copy with outliers replaced by NaN
clean = sigma_clip(data)

# In-place: modifies data directly
sigma_clip!(data)

# Mask only: returns a BitArray (true = outlier or non-finite)
mask = sigma_clip_mask(data)
clean = data[.!mask]
```

---

## API reference

### `sigma_clip(x; kwargs...) -> Array{<:AbstractFloat}`

Out-of-place sigma clipping. Returns a copy of `x` with outliers replaced by
`NaN`. Integer arrays are promoted to `Float64`.

### `sigma_clip!(x; kwargs...) -> x`

In-place version. Replaces outliers in `x` with `NaN`. Requires
`x <: AbstractArray{<:AbstractFloat}`.

### `sigma_clip_mask(x; kwargs...) -> BitArray`

Returns a boolean mask where `true` marks a clipped or non-finite value. The
input array is never modified.

### `sigma_clip_mask!(x, target; kwargs...) -> target`

Writes the outlier flags into a pre-allocated `BitArray` `target` of the same
shape as `x`.

### `SigmaClip.sigma_clip_bounds(x; kwargs...) -> (lb, ub)`

Returns the final convergence bounds `(lower, upper)` without modifying `x` or
producing a mask. Useful when you only need to know the thresholds.

```julia
lb, ub = SigmaClip.sigma_clip_bounds(data; sigma_lower=2.5)
println("outliers: x < $lb  or  x > $ub")
```

### Keyword arguments (all functions)

| Keyword | Default | Description |
| :--- | :--- | :--- |
| `sigma_lower` | `3` | Rejection threshold below the centre (in units of dispersion). |
| `sigma_upper` | `3` | Rejection threshold above the centre. |
| `maxiter` | `5` | Maximum number of clipping iterations. Pass `-1` to run until convergence. |
| `cent_reducer` | `fast_median` | Centre estimator. Any callable `f(v::AbstractVector) -> scalar`. |
| `std_reducer` | `mad_std` | Dispersion estimator. Any callable `f(v::AbstractVector) -> scalar`. |
| `mask` | `nothing` | Initial boolean mask (`true` = exclude from bound computation). |
| `workspace` | `nothing` | Pre-allocated [`SigmaClipWorkspace`](#zero-allocation-hot-loops). |

---

## Reducer sentinels

SigmaClip exports two sentinel values that unlock specialised, faster code paths
resolved entirely at compile time via Julia's multiple dispatch — there is no
runtime overhead compared to calling the functions directly.

### `fast_median`

Pass as `cent_reducer=fast_median` (the default) to select the built-in
allocation-free O(n) quickselect median. When combined with `std_reducer=mad_std`,
the median is computed **once** and shared with the MAD calculation (two
quickselects per iteration instead of three).

`fast_median!` is also exported as a standalone utility if you need a fast
in-place median elsewhere:

```julia
buf = [3.0, 1.0, 4.0, 1.0, 5.0]
m = fast_median!(buf)   # reorders buf, returns 3.0
```

> [!WARNING]
> `fast_median!` modifies the order of elements in the input vector.
> All values are preserved, but the original ordering is lost.

### `mad_std`

Pass as `std_reducer=mad_std` to use the **Median Absolute Deviation** (scaled
by 1.4826 to match the standard deviation of a normal distribution) as the
dispersion estimator:

```
MAD-std = median(|xᵢ − median(x)|) × 1.4826
```

This is significantly more robust than `std` when the data contains many
outliers or follows a heavy-tailed distribution, at the cost of one extra
quickselect per iteration.

```julia
# Robust clipping with shared median computation
sigma_clip!(data; std_reducer=mad_std)

# Explicit (same as above — fast_median is the default cent_reducer)
sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
```

> [!NOTE]
> The sentinel `fast_median` (used as `cent_reducer=fast_median`) is distinct
> from the function `fast_median!`. Passing the function directly as
> `cent_reducer=fast_median!` also works but falls through to the generic
> dispatch path and will **not** share the median with a `mad_std` calculation.

---

## Custom statistics

Any callable that accepts an `AbstractVector` and returns a scalar can be used
as `cent_reducer` or `std_reducer`. SigmaClip has no dependency on
`Statistics.jl`, but you can freely pull functions from it (or anywhere else)
as custom reducers by adding `Statistics` to your own project.

```julia
using Statistics   # your project's dependency, not SigmaClip's

# Mean centre + standard deviation (less robust, but faster than MAD)
sigma_clip!(data; cent_reducer=mean, std_reducer=std)

# Median centre + IQR-based spread (fully custom)
iqr_spread(v) = (quantile(v, 0.75) - quantile(v, 0.25)) / 1.349
sigma_clip!(data; std_reducer=iqr_spread)
```

---

## Zero-allocation hot loops

When applying sigma clipping to thousands of arrays (e.g. every row of a 2-D
image), the internal buffer allocations can become a bottleneck. Allocate a
`SigmaClipWorkspace` once and pass it via the `workspace` keyword to make every
subsequent call allocation-free.

```julia
# Allocate once — must be at least as long as each array being processed
ws = SigmaClipWorkspace(Float64, size(image, 2))

for row in eachrow(image)
    sigma_clip!(row; workspace=ws)
end
```

The workspace holds two internal buffers:

- `buf` — working copy of the valid elements, compacted in-place each iteration.
- `aux` — auxiliary buffer used only when `std_reducer=mad_std` to hold the
  absolute deviations `|xᵢ − median|` without overwriting `buf`.

Both buffers are the same length. When `mad_std` is not used, `aux` is never
written.

### Constructors

```julia
SigmaClipWorkspace(Float64, n)   # explicit type and capacity
SigmaClipWorkspace(my_array)     # T and length inferred from the array
                                 # (integers are promoted to Float64)
```

If you pass a workspace whose buffer is shorter than the input array, an
`ArgumentError` is thrown immediately before any computation begins.

---

## Performance summary

| Configuration | Quickselects / iteration | Notes |
| :--- | :---: | :--- |
| `fast_median` + `mad_std` (**default**) | 2 | median shared; `aux` written once per iteration |
| `fast_median` + `std` | 1 | `std` is a single O(n) pass; less robust |
| custom `cent_reducer` + `mad_std` | 2–3 | median computed independently |
| custom `cent_reducer` + custom `std_reducer` | — | fully generic, no fast path |

The quickselect used internally (Wirth's algorithm) has O(n) average time and
O(n²) worst case. On typical scientific data the average case dominates.

---

## License

SigmaClip.jl is licensed under the MIT License. See [LICENSE](LICENSE) for details.