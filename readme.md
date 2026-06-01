[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

# SigmaClip.jl

> [!NOTE]
> The algorithmic logic implemented in this module is inspired by the
> `sigma_clip` implementation in the Astropy Python library.

SigmaClip.jl identifies outliers in numeric arrays with iterative sigma
clipping. It is designed for scientific workloads such as light-curve cleaning,
cosmic-ray rejection, and image-stack processing, but it works on any numeric
array.

The package provides:

- an out-of-place API that returns a clipped copy,
- an in-place API that writes `NaN` into rejected entries,
- validity masks where `true` means retained,
- final clipping bounds without building a mask,
- a workspace path for allocation-sensitive loops.

SigmaClip.jl has no external runtime dependencies.

---

## Installation

```julia
using Pkg
Pkg.add("SigmaClip")
```

---

## How It Works

Sigma clipping repeatedly:

1. estimates a center and dispersion from the currently retained finite values,
2. computes bounds
   `[center - sigma_lower * dispersion, center + sigma_upper * dispersion]`,
3. keeps values inside those bounds,
4. stops at convergence or after `maxiter` iterations.

Defaults:

- `center = fast_median!`
- `spread = mad_std!`
- `sigma_lower = 3`
- `sigma_upper = 3`
- `maxiter = 5`
- `exclude = nothing`
- `workspace = nothing`

`fast_median!` uses in-place quickselect. `mad_std!` computes the median
absolute deviation scaled by `1.4826`.

---

## Quick Start

```julia
using SigmaClip

data = [1.0, 1.0, 1.0, 50.0, NaN]

# Out-of-place: returns a copy with outliers and non-finite values set to NaN.
clipped = sigma_clip(data)

# Mask only: true means finite and retained by the final bounds.
mask = sigma_clip_mask(data)
retained = data[mask]

# Bounds only: useful if you want to classify values yourself.
lower, upper = sigma_clip_bounds(data)
```

`data` is unchanged by `sigma_clip`, `sigma_clip_mask`, and
`sigma_clip_bounds`.

---

## Common Recipes

### Return a clipped copy

Use `sigma_clip` when you want to keep the original array unchanged.

```julia
using SigmaClip

data = [2.0, 2.0, 2.0, 100.0]
clipped = sigma_clip(data)

isnan(clipped[end])
# true

data[end]
# 100.0
```

Integer inputs are converted with `float.(x)` so the result can represent
`NaN`.

```julia
clipped = sigma_clip([1, 1, 1, 99])

eltype(clipped)
# Float64
```

### Modify a floating-point array in place

Use `sigma_clip!` when the input array can store `NaN`.

```julia
data = [0.0, 0.0, 0.0, 30.0]

sigma_clip!(data)

isnan(data[end])
# true
```

Integer arrays cannot store `NaN`; use `sigma_clip(x)` for integer input, or
convert the input before calling `sigma_clip!`.

### Build a validity mask

Use `sigma_clip_mask` when you want to select retained values yourself.

```julia
data = [1.0, 1.0, 1.0, 50.0, Inf]
mask = sigma_clip_mask(data)

mask
# 5-element BitVector:
#  1
#  1
#  1
#  0
#  0

retained = data[mask]
# [1.0, 1.0, 1.0]
```

For repeated calls, write into a pre-allocated boolean array:

```julia
target = falses(length(data))
sigma_clip_mask!(data, target)
```

### Get the final bounds

Use `sigma_clip_bounds` when you only need the thresholds.

```julia
data = [0.0, 0.0, 0.0, 10.0]
lower, upper = sigma_clip_bounds(data)

valid = isfinite.(data) .& (lower .<= data .<= upper)
# Bool[true, true, true, false]
```

### Use mean and standard deviation

SigmaClip does not depend on `Statistics`, but you can pass reducers from it.

```julia
using Statistics
using SigmaClip

data = [1.0, 1.0, 1.0, 8.0]
sigma_clip!(data; center = mean, spread = std)
```

### Use asymmetric thresholds

Set lower and upper thresholds independently.

```julia
data = [-20.0, 0.0, 0.0, 0.0, 5.0]
sigma_clip!(data; sigma_lower = 2, sigma_upper = 4)
```

### Exclude values from bound estimation

Use `exclude` for values that should not influence the estimated center and
spread. Excluded values are still classified against the final bounds.

```julia
data = [-100.0, 0.0, 0.0, 0.0, 50.0]
exclude = Bool[true, false, false, false, true]

mask = sigma_clip_mask(data; exclude)
# Bool[false, true, true, true, false]
```

---

## API Reference

### `sigma_clip(x; kwargs...) -> AbstractArray`

Run sigma clipping on a copy of `x`. The result contains `NaN` for non-finite
values and outliers. Integer inputs are promoted with `float.(x)`.

### `sigma_clip!(x; kwargs...) -> x`

Run sigma clipping in place. Non-finite values and outliers are replaced with
`NaN`. The element type of `x` must be able to represent `NaN`.

### `sigma_clip_mask(x; kwargs...) -> BitArray`

Return a validity mask with the same shape as `x`. `true` means finite and
retained by the final bounds.

### `sigma_clip_mask!(x, target; kwargs...) -> target`

Write the validity mask into the pre-allocated boolean array `target`.
`target` must have the same axes as `x`.

### `sigma_clip_bounds(x; kwargs...) -> (lower, upper)`

Return the final lower and upper sigma-clipping bounds. This does not modify
`x` and does not return a mask.

### Keywords

| Keyword | Default | Description |
| :--- | :--- | :--- |
| `workspace` | `nothing` | Optional pre-allocated workspace. |
| `exclude` | `nothing` | Boolean array with the same axes as `x`; `true` excludes a value from bound estimation. |
| `sigma_lower` | `3` | Lower sigma threshold. Must be finite and strictly positive. |
| `sigma_upper` | `3` | Upper sigma threshold. Must be finite and strictly positive. |
| `center` | `fast_median!` | Center estimator used at each iteration. |
| `spread` | `mad_std!` | Dispersion estimator used at each iteration. |
| `maxiter` | `5` | Maximum number of iterations. Use `-1` to run until convergence. |

All public clipping functions accept the same keywords.

---

## Built-In Statistics

SigmaClip exports two reducer functions:

- `fast_median!`
- `mad_std!`

They can be used directly, and they also select optimized paths when passed as
`center = fast_median!` and `spread = mad_std!`.

### `fast_median!`

```julia
buf = [3.0, 1.0, 4.0, 1.0, 5.0]
m = fast_median!(buf)

m
# 3.0
```

`fast_median!` mutates the order of `buf`. When used through `sigma_clip`, it
only mutates SigmaClip's internal workspace, not the user's input array.

### `mad_std!`

```julia
buf = [1.0, 1.0, 1.0, 10.0]
s = mad_std!(buf)
```

`mad_std!` mutates its input. The one-argument form allocates an auxiliary
buffer. Inside SigmaClip, the workspace path provides that auxiliary buffer.

---

## Zero-Allocation Hot Loops

When applying sigma clipping many times, allocate workspace once and pass it
with the `workspace` keyword.

```julia
using SigmaClip

image = randn(128, 1024)
workspace = SigmaClipWorkspace(
    Vector{Float64}(undef, size(image, 2)),
    Vector{Float64}(undef, size(image, 2)),
)

for row in eachrow(image)
    sigma_clip!(row; workspace)
end
```

`SigmaClipWorkspace` stores two buffers:

- `buf`: packed finite values retained during sigma clipping,
- `aux`: auxiliary scratch space used by `mad_std!` and workspace-aware
  reducers.

For custom workflows that do not need auxiliary scratch space, `aux` may be
`nothing`, but the default `spread = mad_std!` requires it.

```julia
workspace = SigmaClipWorkspace(Vector{Float64}(undef, length(data)), nothing)
sigma_clip!(data; workspace, spread = x -> 1.0)
```

If the workspace buffers are shorter than the input array, SigmaClip throws
before running the algorithm.

---

## Custom Workspace Types

External workspace types can participate by implementing:

```julia
SigmaClip.workspace_buffer(ws)
SigmaClip.workspace_auxbuffer(ws)
```

`workspace_buffer(ws)` must return a writable, 1-indexed `AbstractVector` with
the same element type as the input and length at least `length(x)`.

`workspace_auxbuffer(ws)` may return another writable vector or `nothing`. It
must return a vector when the selected spread function needs auxiliary scratch
space, including the built-in `mad_std!`.

```julia
struct ExternalWorkspace
    buf::Vector{Float64}
    aux::Vector{Float64}
end

SigmaClip.workspace_buffer(ws::ExternalWorkspace) = ws.buf
SigmaClip.workspace_auxbuffer(ws::ExternalWorkspace) = ws.aux

workspace = ExternalWorkspace(
    Vector{Float64}(undef, length(data)),
    Vector{Float64}(undef, length(data)),
)

sigma_clip!(data; workspace)
```

---

## Custom Statistics

Any callable that accepts an `AbstractVector` and returns a scalar can be used
as `center` or `spread`.

```julia
using Statistics
using SigmaClip

iqr_spread(v) = (quantile(v, 0.75) - quantile(v, 0.25)) / 1.349

data = [1.0, 1.0, 1.0, 10.0]
sigma_clip!(data; spread = iqr_spread)
```

Reducers receive a mutable view of SigmaClip's internal workspace. They may
reorder it, but they must preserve the values. Do not overwrite the data with
derived quantities.

Reducers that need direct workspace access can extend
`SigmaClip.statistic(f, ws, n)`.

```julia
struct MeanAbsDeviation end

function SigmaClip.statistic(::MeanAbsDeviation, ws, n::Int)
    data = @view SigmaClip.workspace_buffer(ws)[1:n]
    aux = @view SigmaClip.workspace_auxbuffer(ws)[1:n]
    c = sum(data) / length(data)

    @inbounds for i in eachindex(data)
        aux[i] = abs(data[i] - c)
    end

    return sum(aux) / length(aux)
end

data = [1.0, 1.0, 1.0, 10.0]
sigma_clip!(data; spread = MeanAbsDeviation())
```

---

## Performance Notes

| Configuration | Notes |
| :--- | :--- |
| `fast_median!` + `mad_std!` | Default robust path; median is shared with MAD. |
| `fast_median!` + `std` | One quickselect plus standard deviation. |
| custom `center` + custom `spread` | Uses the generic statistic protocol. |

The quickselect used by `fast_median!` has O(n) average time and O(n²) worst
case. On typical scientific arrays, the average case dominates.

---

## License

SigmaClip.jl is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
