[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

# SigmaClip.jl

SigmaClip.jl removes outliers from numeric arrays with iterative sigma
clipping. It works on vectors, matrices, and higher-dimensional arrays.

Use it when you need to:

- replace rejected values with `NaN`,
- build a mask of retained values,
- compute clipping bounds and apply them yourself,
- reuse scratch buffers in repeated calls,
- plug in custom center or spread statistics.

The default algorithm follows the robust sigma-clipping approach used by
Astropy's `sigma_clip`: median as center, MAD-based standard deviation as
spread, and repeated clipping until convergence or `maxiter`.

SigmaClip.jl has no runtime dependencies.

## Contents

1. [Installation](#installation)
2. [Usage and API](#usage-and-api)
   - [Quick start](#quick-start)
   - [How sigma clipping runs](#how-sigma-clipping-runs)
   - [Choose the right function](#choose-the-right-function)
   - [Clipped copies and in-place clipping](#clipped-copies-and-in-place-clipping)
   - [Masks and bounds](#masks-and-bounds)
   - [Keyword arguments](#keyword-arguments)
   - [Exported methods](#exported-methods)
   - [Built-in statistics](#built-in-statistics)
3. [Extend API with Custom Buffer and Statistics](#extend-api-with-custom-buffer-and-statistics)
   - [Reuse buffers in hot loops](#reuse-buffers-in-hot-loops)
   - [Hook 1: custom workspace buffers](#hook-1-custom-workspace-buffers)
   - [Hook 2: plain custom statistics](#hook-2-plain-custom-statistics)
   - [Hook 3: workspace-aware statistics](#hook-3-workspace-aware-statistics)
   - [Hook contracts](#hook-contracts)
4. [Performance Notes](#performance-notes)
5. [License](#license)

## Installation

Install SigmaClip.jl from the Julia package manager:

```julia
using Pkg
Pkg.add("SigmaClip")
```

Load it with:

```julia
using SigmaClip
```

## Usage and API

### Quick start

```julia
using SigmaClip

data = [1.0, 1.0, 1.0, 50.0, NaN]

clipped = sigma_clip(data)
mask = sigma_clip_mask(data)
lower, upper = sigma_clip_bounds(data)

clipped
# 5-element Vector{Float64}:
#    1.0
#    1.0
#    1.0
#  NaN
#  NaN

mask
# 5-element BitVector:
#  1
#  1
#  1
#  0
#  0

(lower, upper)
# (1.0, 1.0)
```

`sigma_clip`, `sigma_clip_mask`, and `sigma_clip_bounds` leave `data`
unchanged.

### How sigma clipping runs

SigmaClip repeats the same loop until the retained set stops changing or
`maxiter` runs out:

1. Pack finite values into a workspace buffer.
2. Estimate the center and spread of the retained values.
3. Compute bounds:

   ```julia
   lower = center - sigma_lower * spread
   upper = center + sigma_upper * spread
   ```

4. Reject values outside `[lower, upper]`.

The defaults are:

| Setting | Default |
| :--- | :--- |
| `center` | `fast_median!` |
| `spread` | `mad_std!` |
| `sigma_lower` | `3` |
| `sigma_upper` | `3` |
| `maxiter` | `5` |
| `exclude` | `nothing` |
| `workspace` | `nothing` |

`fast_median!` computes a median with in-place quickselect. `mad_std!` computes
the median absolute deviation and scales it by `1.4826022185056018`.

### Choose the right function

| Function | Use it when |
| :--- | :--- |
| `sigma_clip(x)` | You want a clipped copy and want to keep `x` unchanged. |
| `sigma_clip!(x)` | You want to modify a floating-point array in place. |
| `sigma_clip_mask(x)` | You want a `BitArray` where `true` means retained. |
| `sigma_clip_mask!(x, target)` | You already allocated the boolean mask. |
| `sigma_clip_bounds(x)` | You only need the final lower and upper bounds. |

All clipping functions use the same keyword arguments.

### Clipped copies and in-place clipping

Use `sigma_clip` when you want a new array:

```julia
data = [2.0, 2.0, 2.0, 100.0]
clipped = sigma_clip(data)

isnan(clipped[end])
# true

data[end]
# 100.0
```

Integer inputs work with `sigma_clip` because the function converts them with
`float.(x)` before it writes `NaN`:

```julia
clipped = sigma_clip([1, 1, 1, 99])

eltype(clipped)
# Float64
```

Use `sigma_clip!` when the input array can store `NaN`:

```julia
data = [0.0, 0.0, 0.0, 30.0]
sigma_clip!(data)

isnan(data[end])
# true
```

`sigma_clip!` rejects integer arrays because they cannot represent `NaN`.
Convert the input first or use `sigma_clip(x)`.

### Masks and bounds

Use `sigma_clip_mask` when you want to select retained values yourself:

```julia
data = [1.0, 1.0, 1.0, 50.0, Inf]
mask = sigma_clip_mask(data)
retained = data[mask]

retained
# [1.0, 1.0, 1.0]
```

For repeated calls, allocate the mask once:

```julia
target = falses(length(data))
sigma_clip_mask!(data, target)
```

Use `sigma_clip_bounds` when another part of your code applies the threshold:

```julia
data = [0.0, 0.0, 0.0, 10.0]
lower, upper = sigma_clip_bounds(data)

valid = isfinite.(data) .& (lower .<= data .<= upper)
# Bool[true, true, true, false]
```

### Keyword arguments

| Keyword | Default | Meaning |
| :--- | :--- | :--- |
| `workspace` | `nothing` | Scratch buffers used during clipping. Pass a workspace to reduce allocations. |
| `exclude` | `nothing` | Boolean array with the same axes as `x`. `true` removes a value from bound estimation. |
| `sigma_lower` | `3` | Lower sigma threshold. Must be finite and positive. |
| `sigma_upper` | `3` | Upper sigma threshold. Must be finite and positive. |
| `center` | `fast_median!` | Reducer used to estimate the center at each iteration. |
| `spread` | `mad_std!` | Reducer used to estimate dispersion at each iteration. |
| `maxiter` | `5` | Maximum number of iterations. Use `-1` to run until convergence. |

Use `exclude` when some values should not influence the center or spread but
should still be classified by the final bounds:

```julia
data = [-100.0, 0.0, 0.0, 0.0, 50.0]
exclude = Bool[true, false, false, false, true]

sigma_clip_mask(data; exclude)
# Bool[false, true, true, true, false]
```

Set asymmetric thresholds when low and high outliers need different treatment:

```julia
data = [-20.0, 0.0, 0.0, 0.0, 5.0]
sigma_clip!(data; sigma_lower = 2, sigma_upper = 4)
```

Pass statistics from `Statistics` or your own reducers:

```julia
using Statistics
using SigmaClip

data = [1.0, 1.0, 1.0, 8.0]
sigma_clip!(data; center = mean, spread = std)
```

### Exported methods

#### `sigma_clip(x; kwargs...) -> AbstractArray`

Return a clipped copy of `x`. The result contains `NaN` for non-finite values
and outliers. Integer inputs return a floating-point array.

#### `sigma_clip!(x; kwargs...) -> x`

Clip `x` in place by writing `NaN` into non-finite values and outliers. The
element type of `x` must support `NaN`.

#### `sigma_clip_mask(x; kwargs...) -> BitArray`

Return a mask with the same shape as `x`. `true` marks finite values retained
by the final bounds.

#### `sigma_clip_mask!(x, target; kwargs...) -> target`

Write the mask into `target`. `target` must have the same axes as `x` and an
element type of `Bool`.

#### `sigma_clip_bounds(x; kwargs...) -> (lower, upper)`

Return the final clipping bounds. This function does not modify `x` and does
not build a mask.

#### `SigmaClipWorkspace(buf, aux)`

Store scratch buffers for repeated clipping calls. `buf` stores packed finite
values. `aux` stores scratch space for `mad_std!` and custom workspace-aware
statistics.

#### `fast_median!(a) -> Number`

Compute the median of `a` with in-place quickselect. The function may reorder
`a`, but it preserves the values.

#### `mad_std!(a) -> Number`

Compute the median absolute deviation scaled to match the standard deviation of
a normal distribution. The one-argument form allocates an auxiliary buffer.

### Built-in statistics

`fast_median!` and `mad_std!` can be called directly:

```julia
buf = [3.0, 1.0, 4.0, 1.0, 5.0]
fast_median!(buf)
# 3.0
```

```julia
buf = [1.0, 1.0, 1.0, 10.0]
mad_std!(buf)
```

Both functions mutate their input. When you pass them through SigmaClip's
clipping API, they mutate only SigmaClip's workspace buffer.

## Extend API with Custom Buffer and Statistics

SigmaClip has three extension points:

1. `SigmaClip.workspace_buffer(ws)` gives SigmaClip a main scratch buffer.
2. `SigmaClip.workspace_auxbuffer(ws)` gives SigmaClip auxiliary scratch space.
3. `SigmaClip.statistic(f, ws, n)` lets a reducer read the workspace directly.

Use the first two hooks to connect an external workspace type. Use the third
hook when a custom statistic needs scratch memory, specialized dispatch, or
direct access to the compacted values.

### Reuse buffers in hot loops

Allocate a `SigmaClipWorkspace` once when you clip many arrays with the same
maximum length:

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

`SigmaClipWorkspace` stores:

- `buf`, the compacted finite values retained during the current iteration,
- `aux`, scratch storage used by `mad_std!` and workspace-aware statistics.

If your `spread` reducer does not need auxiliary storage, use `nothing` for the
second buffer:

```julia
workspace = SigmaClipWorkspace(Vector{Float64}(undef, length(data)), nothing)
sigma_clip!(data; workspace, spread = x -> 1.0)
```

The main buffer must have at least `length(x)` slots. The auxiliary buffer must
also have at least `length(x)` slots when the selected statistic needs it.

### Hook 1: custom workspace buffers

Implement `workspace_buffer` and `workspace_auxbuffer` when another object owns
the scratch memory:

```julia
struct ExternalWorkspace
    buf::Vector{Float64}
    aux::Vector{Float64}
end

SigmaClip.workspace_buffer(ws::ExternalWorkspace) = ws.buf
SigmaClip.workspace_auxbuffer(ws::ExternalWorkspace) = ws.aux

data = [1.0, 1.0, 1.0, 10.0]
workspace = ExternalWorkspace(
    Vector{Float64}(undef, length(data)),
    Vector{Float64}(undef, length(data)),
)

sigma_clip!(data; workspace)
```

`workspace_auxbuffer(ws)` may return `nothing` only when the selected `spread`
does not need auxiliary storage.

### Hook 2: plain custom statistics

Pass any callable as `center` or `spread` if it accepts an `AbstractVector` and
returns one scalar:

```julia
using Statistics
using SigmaClip

iqr_spread(v) = (quantile(v, 0.75) - quantile(v, 0.25)) / 1.349

data = [1.0, 1.0, 1.0, 10.0]
sigma_clip!(data; spread = iqr_spread)
```

SigmaClip passes a mutable view of its internal buffer to the reducer. Your
reducer may reorder the values. It must preserve them because the clipping loop
uses the same buffer after computing the statistic.

### Hook 3: workspace-aware statistics

Extend `SigmaClip.statistic(f, ws, n)` when a reducer needs direct workspace
access:

```julia
struct MeanAbsDeviation end

function SigmaClip.statistic(::MeanAbsDeviation, ws, n::Int)
    data = @view SigmaClip.workspace_buffer(ws)[1:n]
    aux = @view SigmaClip.workspace_auxbuffer(ws)[1:n]
    center = sum(data) / length(data)

    @inbounds for i in eachindex(data)
        aux[i] = abs(data[i] - center)
    end

    return sum(aux) / length(aux)
end

data = [1.0, 1.0, 1.0, 10.0]
sigma_clip!(data; spread = MeanAbsDeviation())
```

This hook gives the reducer both buffers and the number of valid compacted
entries. Use only `1:n`.

### Hook contracts

| Hook | Contract |
| :--- | :--- |
| `workspace_buffer(ws)` | Return a writable, 1-indexed `AbstractVector` with the same element type as `x` and length at least `length(x)`. |
| `workspace_auxbuffer(ws)` | Return a writable vector with length at least `length(x)`, or `nothing` when the selected statistics do not need auxiliary storage. |
| `statistic(f, ws, n)` | Return one scalar from `workspace_buffer(ws)[1:n]`. Preserve the values in the main buffer. Use the auxiliary buffer only as scratch. |

Common combinations:

| Configuration | Typical use |
| :--- | :--- |
| Plain callable `center` or `spread` | The reducer only needs the compacted vector. |
| Custom workspace hooks | Another type owns reusable scratch memory. |
| Custom `statistic(f, ws, n)` | The reducer needs scratch space or specialized workspace access. |
| `SigmaClipWorkspace(buf, nothing)` | The selected `spread` does not need auxiliary storage. |
| `SigmaClipWorkspace(buf, aux)` | The selected `spread` is `mad_std!` or a custom statistic uses `aux`. |

## Performance Notes

The default configuration uses `fast_median!` and `mad_std!`.

| Configuration | Notes |
| :--- | :--- |
| `fast_median!` + `mad_std!` | Robust default. SigmaClip shares the median with the MAD calculation. |
| `fast_median!` + `std` | Uses quickselect for the center and standard deviation for spread. |
| Custom `center` + custom `spread` | Uses the generic `statistic` protocol. |
| Reused `SigmaClipWorkspace` | Avoids allocating scratch buffers on each call. |

`fast_median!` uses quickselect. It has O(n) average time and O(n^2) worst-case
time.

## License

SigmaClip.jl is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
