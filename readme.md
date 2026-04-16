[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# SigmaClip.jl

> [!NOTE]
> The algorithmic logic implemented in this module is inspired by the `sigma_clip`
> implementation found in the Astropy Python library.

SigmaClip.jl provides a lightweight, highly efficient, and robust toolset for
identifying and rejecting outliers using iterative sigma clipping. It is
primarily designed for **astrophysical data processing** — cleaning light
curves, removing cosmic rays from CCD frames, rejecting bad pixels in image
stacks — but works equally well on any numeric array: 1-D, 2-D
images, or higher-dimensional data.

The library is designed to be numerically stable, handling `NaN` and `Inf`
values gracefully, and offers both a convenient one-liner API and a
zero-allocation path for high-throughput workloads.

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

# Mask only: returns a BitArray (true = good / retained pixel)
mask = sigma_clip_mask(data)
clean = data[mask]
```

---

## Common recipes

### Use a mask

Use `sigma_clip_mask` when you want to keep the original data unchanged and
select the retained values yourself.

```julia
mask = sigma_clip_mask(data)
clean = data[mask]
```

The mask uses `true` for finite values retained by the final clipping bounds.

### Modify a floating-point array in-place

Use `sigma_clip!` when the input array can store `NaN`.

```julia
data = randn(500)
data[end] = 1e6

sigma_clip!(data)
```

Integer arrays cannot store `NaN`; use `sigma_clip(data)` for integer input, or
convert the input to a floating-point array before calling `sigma_clip!`.

### Use mean and standard deviation

SigmaClip does not depend on `Statistics`, but your project can use it for
custom centre and spread functions.

```julia
using Statistics

sigma_clip!(data; center=mean, spread=std)
```

### Use asymmetric thresholds

Set the lower and upper thresholds independently when only one side of the
distribution should be clipped aggressively.

```julia
sigma_clip!(data; sigma_lower=5, sigma_upper=2)
```

### Exclude values from bound estimation

Use `exclude` for values that should not influence the estimated centre and
spread. Excluded values are still classified against the final bounds; they are
not automatically forced to `false` in the mask or `NaN` in the clipped output.

```julia
exclude = falses(size(data))
exclude[known_bad_index] = true

mask = sigma_clip_mask(data; exclude=exclude)
```

---

## API reference

### `sigma_clip(x; kwargs...) -> Array{<:Number}`

Out-of-place sigma clipping. Returns a copy of `x` with outliers replaced by
`NaN`. Integer arrays are promoted to `Float64`; numeric arrays with units keep
their element type when it can represent `NaN`.

### `sigma_clip!(x; kwargs...) -> x`

In-place version. Replaces outliers in `x` with `NaN`. Requires an array whose
element type can represent `NaN`.

### `sigma_clip_mask(x; kwargs...) -> BitArray`

Returns a boolean mask where `true` marks a finite, retained value. The input
array is never modified.

### `sigma_clip_mask!(x, target; kwargs...) -> target`

Writes pixel-validity flags into a pre-allocated boolean `target` with the same
axes as `x`.

### `SigmaClip.sigma_clip_bounds(x; kwargs...) -> (lb, ub)`

Returns the final convergence bounds `(lower, upper)` without modifying `x` or
producing a mask. Useful when you only need to know the thresholds.

```julia
lb, ub = SigmaClip.sigma_clip_bounds(data; sigma_lower=2.5)
println("outliers: x < $lb  or  x > $ub")
```

### Keyword arguments

| Keyword | Default | Description |
| :--- | :--- | :--- |
| `sigma_lower` | `3` | Finite, non-negative rejection threshold below the centre. |
| `sigma_upper` | `3` | Finite, non-negative rejection threshold above the centre. |
| `maxiter` | `5` | Maximum number of clipping iterations. Pass `-1` to run until convergence. |
| `center` | `fast_median!` | Centre estimator. Any callable `f(v::AbstractVector) -> scalar`, or a workspace-aware reducer via `SigmaClip.statistic(f, ws, n)`. |
| `spread` | `mad_std!` | Dispersion estimator. Any callable `f(v::AbstractVector) -> scalar`, or a workspace-aware reducer via `SigmaClip.statistic(f, ws, n)`. |
| `exclude` | `nothing` | Boolean array with the same axes as `x`; `true` excludes a value from bound estimation only. |
| `workspace` | `nothing` | Pre-allocated workspace for allocation-free operation; accepts [`SigmaClipWorkspace`](#zero-allocation-hot-loops) or a custom type implementing `SigmaClip.workspace_buffer` and `SigmaClip.workspace_auxbuffer`. |

---

## Built-in statistics

SigmaClip exports two reducer functions that also unlock specialised, faster
code paths when passed directly as `center=fast_median!` and `spread=mad_std!`.
The dispatch is resolved entirely at compile time via Julia's multiple
dispatch.

### `fast_median!`

Pass as `center=fast_median!` (the default) to select the built-in
allocation-free O(n) quickselect median. When combined with `spread=mad_std!`,
the median is computed once and shared with the MAD calculation.

`fast_median!` is also exported as a standalone utility if you need a fast
in-place median elsewhere:

```julia
buf = [3.0, 1.0, 4.0, 1.0, 5.0]
m = fast_median!(buf)   # reorders buf, returns 3.0
```

When `fast_median!` is used through `sigma_clip`, it reorders SigmaClip's
internal workspace, not the user's input array.

### `mad_std!`

Pass as `spread=mad_std!` to use the Median Absolute Deviation scaled by
1.4826 to match the standard deviation of a normal distribution:

```julia
# Robust clipping with shared median computation
sigma_clip!(data; spread=mad_std!)
```

The standalone `mad_std!(v)` utility mutates `v` and allocates an auxiliary
buffer. Inside SigmaClip, `mad_std!` uses the provided workspace and can run
without allocations.

---

## Zero-allocation hot loops

When applying sigma clipping to thousands of arrays (e.g. every row of a 2-D
image), the internal buffer allocations can become a bottleneck. Allocate a
`SigmaClipWorkspace` once and pass it via the `workspace` keyword to make every
subsequent call allocation-free. External packages may also pass a custom
workspace type, as long as it exposes the two buffers SigmaClip needs.

```julia
# Allocate once — must be at least as long as each array being processed
ws = SigmaClipWorkspace(Float64, size(image, 2))

for row in eachrow(image)
    sigma_clip!(row; workspace=ws)
end
```

The workspace holds two internal buffers:

- `buf` — working copy of the valid elements, compacted in-place each iteration.
- `aux` — auxiliary buffer used by `mad_std!` and available to workspace-aware
  reducers through `SigmaClip.workspace_auxbuffer(ws)`.

Both buffers are the same length. The workspace itself does not store the
current compacted length; workspace-aware reducers receive it as the `n`
argument to `SigmaClip.statistic(f, ws, n)`.

### Constructors

```julia
SigmaClipWorkspace(Float64, n)   # explicit type and capacity
SigmaClipWorkspace(my_array)     # T and length inferred from the array
                                 # (integers are promoted to Float64)
```

If you pass a workspace whose buffer is shorter than the input array, an
`ArgumentError` is thrown immediately before any computation begins.

### Custom workspace protocol

Custom workspace types participate in the same API by implementing two methods:

```julia
SigmaClip.workspace_buffer(ws)    # main packed-data buffer
SigmaClip.workspace_auxbuffer(ws) # auxiliary MAD buffer
```

Both accessors must return writable, 1-indexed `AbstractVector`s with the exact
numeric type SigmaClip requires for the input being processed (`Float32` for
`Float32` input, the quantity type for unitful input, `Float64` for integer
input, etc.) and length at least `length(x)`.

```julia
struct ExternalWorkspace{T}
    tmp1::Vector{T}
    tmp2::Vector{T}
    tmp3::Vector{T}
end

SigmaClip.workspace_buffer(ws::ExternalWorkspace) = ws.tmp2
SigmaClip.workspace_auxbuffer(ws::ExternalWorkspace) = ws.tmp3

ws = ExternalWorkspace(zeros(Float64, 1024), zeros(Float64, 1024), zeros(Float64, 1024))
sigma_clip!(data; workspace=ws)
```

---

## Custom statistics

Any callable that accepts an `AbstractVector` and returns a scalar can be used
as `center` or `spread`.

```julia
using Statistics

iqr_spread(v) = (quantile(v, 0.75) - quantile(v, 0.25)) / 1.349
sigma_clip!(data; spread=iqr_spread)
```

Custom reducers receive a mutable view of SigmaClip's internal workspace
buffer. They may reorder that view, but they must preserve the data values.
Overwriting values with derived quantities can lead to incorrect clipping in
later iterations.

---

## Advanced extension hooks

Reducers that need direct workspace access can extend
`SigmaClip.statistic(f, ws, n)`. SigmaClip passes the concrete workspace and
the number of compacted values currently stored in `workspace_buffer(ws)`.
The default method is equivalent to:

```julia
SigmaClip.statistic(f, ws, n) = f(@view SigmaClip.workspace_buffer(ws)[1:n])
```

Custom methods can use the auxiliary buffer without allocating:

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

sigma_clip!(data; spread=MeanAbsDeviation())
```

Dispatch on your workspace type when a reducer depends on fields beyond the
two standard buffers:

```julia
struct CalibratedWorkspace{T}
    buf::Vector{T}
    aux::Vector{T}
    gain::T
end

SigmaClip.workspace_buffer(ws::CalibratedWorkspace) = ws.buf
SigmaClip.workspace_auxbuffer(ws::CalibratedWorkspace) = ws.aux

struct GainCorrectedMean end

function SigmaClip.statistic(::GainCorrectedMean, ws::CalibratedWorkspace, n::Int)
    data = @view SigmaClip.workspace_buffer(ws)[1:n]
    return ws.gain * sum(data) / length(data)
end
```

Workspace-aware reducers may reorder `workspace_buffer(ws)[1:n]`, but they
must preserve those values because SigmaClip compacts the same buffer after
computing the statistics. `workspace_auxbuffer(ws)[1:n]` and any extra
workspace fields may be used as scratch.

---

## Performance summary

| Configuration | Quickselects / iteration | Notes |
| :--- | :---: | :--- |
| `fast_median!` + `mad_std!` (**default**) | 2 | median shared; `aux` written once per iteration |
| `fast_median!` + `std` | 1 | `std` is a single O(n) pass; less robust |
| custom `center` + `mad_std!` | 2 + centre | MAD uses `aux` without allocating; median computed independently |
| custom `center` + custom `spread` | — | generic statistic protocol |

The quickselect used internally (Wirth's algorithm) has O(n) average time and
O(n²) worst case. On typical scientific data the average case dominates.

---

## License

SigmaClip.jl is licensed under the MIT License. See [LICENSE](LICENSE) for details.
