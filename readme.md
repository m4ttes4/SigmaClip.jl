# SigmaClip.jl

SigmaClip.jl removes outliers from numeric arrays with iterative sigma clipping.
It supports clipped copies, in-place clipping, masks, bounds, reusable buffers,
and custom center or spread functions. There are no runtime dependencies.

## Installation

```julia
using Pkg
Pkg.add("SigmaClip")
```

## Quick start

```julia
using SigmaClip

data = [0, 1, 2, 3, 4, 5, 6, 50, NaN, Inf]

sigma_clip(data)
# [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, NaN, NaN, NaN]

sigma_clip_mask(data)
# Bool[true, true, true, true, true, true, true, false, false, false]

# Bounds from the final clipping iteration.
sigma_clip_bounds(data)
# (-5.89561331103361, 11.89561331103361)

sigma_clipped_stats(data; :median => fast_median!, :madstd => mad_std!)
# (center = 3.0, spread = 2.9652044370112036, median = 3.0, madstd = 2.9652044370112036)
```

`sigma_clip`, `sigma_clip_mask`, `sigma_clip_bounds`, and `sigma_clipped_stats`
leave `data` unchanged.
Use `sigma_clip!` to replace rejected values in a floating-point array with
`NaN`, or `sigma_clip_mask!` to reuse a boolean target array.

## API

| Function | Result |
| :--- | :--- |
| `sigma_clip(x)` | Clipped copy; integer inputs become floating point. |
| `sigma_clip!(x)` | Modifies `x` by replacing rejected values with `NaN`. |
| `sigma_clip_mask(x)` | `BitArray` where `true` means retained. |
| `sigma_clip_mask!(x, target)` | Writes the mask into `target`. |
| `sigma_clip_bounds(x)` | Final `(lower, upper)` clipping bounds. |
| `sigma_clipped_stats(x; pairs...)` | Named tuple of final statistics. |

All clipping functions accept these keywords:

| Keyword | Default | Meaning |
| :--- | :--- | :--- |
| `workspace` | `nothing` | Reusable `SigmaClipWorkspace`. |
| `exclude` | `nothing` | Boolean array; `true` excludes a value from bound estimation. |
| `sigma_lower` | `3` | Positive lower rejection threshold. |
| `sigma_upper` | `3` | Positive upper rejection threshold. |
| `center` | `fast_median!` | Center function. |
| `spread` | `mad_std!` | Spread function. |
| `maxiter` | `5` | Iteration limit; `-1` runs until convergence. |

> [!WARNING]
> `exclude` only removes selected values from the calculation of the clipping
> bounds. It does not protect them from the final clipping step: excluded
> values are still compared with the final bounds and may be classified as
> outliers.

Excluded values are still classified against the final bounds:

```julia
data = [-100.0, 0.0, 0.0, 0.0, 50.0]
exclude = Bool[true, false, false, false, true]

sigma_clip_mask(data; exclude)
# Bool[false, true, true, true, false]
```

Lower and upper thresholds can differ:

```julia
sigma_clip!(data; sigma_lower = 2, sigma_upper = 4)
```

`sigma_clipped_stats` returns `center` and `spread`, computed from the values
retained after clipping. Additional statistics can be requested with
symbol-keyed pairs; each callable receives the compacted internal buffer view:

```julia
using Statistics

sigma_clipped_stats(
    data;
    center = mean,
    spread = std,
    :mean => mean,
    :minimum => minimum,
)
# (center = ..., spread = ..., mean = ..., minimum = ...)
```

Use `:name => function` for pair keys. The unquoted form `name => function`
requires `name` to be a variable defined in the calling scope.

## Reusing workspace

Allocate buffers once when clipping many arrays of the same maximum size:

```julia
image = randn(128, 1024)
workspace = SigmaClipWorkspace(
    Vector{Float64}(undef, size(image, 2)),
    Vector{Float64}(undef, size(image, 2)),
)

for row in eachrow(image)
    sigma_clip!(row; workspace)
end
```

The main buffer must have the same element type as the input and at least as
many elements. The auxiliary buffer is required by `mad_std!`; pass `nothing`
when using a spread function that does not need it.

```julia
using Statistics

workspace = SigmaClipWorkspace(similar(data), nothing)
sigma_clip_bounds(data; workspace, spread = std)
```

Custom workspace types can implement the following contract:

```julia
SigmaClip.workspace_buffer(ws)    # -> writable AbstractVector
SigmaClip.workspace_auxbuffer(ws) # -> AbstractVector or nothing
```

The main buffer must have the same element type as the input array and at
least as many elements. It is used to pack the finite, non-excluded values.
The auxiliary buffer is required when `spread = mad_std!`; otherwise it may be
`nothing`. The methods must be defined in the `SigmaClip` namespace:

```julia
struct CustomWorkspace{B, A}
    buf::B
    aux::A
end

SigmaClip.workspace_buffer(ws::CustomWorkspace) = ws.buf
SigmaClip.workspace_auxbuffer(ws::CustomWorkspace) = ws.aux

workspace = CustomWorkspace(
    Vector{Float64}(undef, 1024),
    Vector{Float64}(undef, 1024),
)
sigma_clip_bounds(data; workspace)
```

## Custom statistics

Any callable that accepts an `AbstractVector` and returns a scalar can be used
as `center` or `spread`:

```julia
using Statistics

iqr_spread(v) = (quantile(v, 0.75) - quantile(v, 0.25)) / 1.349
sigma_clip(data; center = mean, spread = iqr_spread)
```

`fast_median!` and `mad_std!` are also exported for direct use. Both may reorder
their input. Calls through SigmaClip operate on the workspace, not the user's
array.

## License

SigmaClip.jl is licensed under the MIT License. See [LICENSE](LICENSE).

## Performance

Benchmark comparison with [Astropy](https://www.astropy.org/):

![Sigma clipping performance comparison](benchmark/Astropy-compare/benchmark_plot.png)
