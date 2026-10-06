```@meta
DocTestSetup = :(using SigmaClip)
```

# Guide

## Choosing an output

| Function | Result |
| :--- | :--- |
| [`sigma_clip(x, sigma)`](@ref sigma_clip) | Clipped copy; integer inputs become floating point. |
| [`sigma_clip!(x, sigma)`](@ref sigma_clip!) | Replaces outliers and non-finite values in `x` with `NaN`. |
| [`sigma_clip_mask(x, sigma)`](@ref sigma_clip_mask) | Validity mask as a `BitArray`. |
| [`sigma_clip_mask!(x, target, sigma)`](@ref sigma_clip_mask!) | Writes the validity mask into `target`. |
| [`sigma_clip_bounds(x, sigma)`](@ref sigma_clip_bounds) | Final `(lower, upper)` bounds. |
| [`sigma_clipped_stats(x, sigma; pairs...)`](@ref sigma_clipped_stats) | Named tuple of statistics on the retained values. |

The thresholds are positional and have no default. A single `sigma` sets both
bounds; `sigma_lower, sigma_upper` set them separately, as in
`sigma_clip(x, 2, 4)`. Both are in units of the spread and must be finite and
positive.

All of them accept the same keywords:

| Keyword | Default | Meaning |
| :--- | :--- | :--- |
| `workspace` | `nothing` | Reusable scratch buffers; see [Reusing a workspace](@ref). |
| `exclude` | `nothing` | Boolean array; `true` removes a value from bound estimation. |
| `center` | `fast_median!` | Center estimator. |
| `spread` | `mad_std!` | Spread estimator. |
| `maxiter` | `5` | Iteration limit; `-1` runs until convergence. |

## Clipping in place

`sigma_clip!` writes `NaN`, so it needs a floating-point array. Here the lower
threshold is 2 and the upper threshold is 4:

```jldoctest
julia> x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -10.0, 12.0];

julia> sigma_clip!(x, 2, 4);

julia> x'
1×9 adjoint(::Vector{Float64}) with eltype Float64:
 0.0  1.0  2.0  3.0  4.0  5.0  6.0  NaN  12.0
```

For integer arrays, use `sigma_clip`, which returns a floating-point copy.

## Excluding values from bound estimation

`exclude` keeps values out of the center and spread estimates. The excluded
values are still compared with the final bounds, so it does not protect them
from being classified as outliers:

```jldoctest
julia> data = [-100.0, 0.0, 0.1, -0.1, 50.0, 0.05];

julia> exclude = Bool[true, false, false, false, true, false];

julia> sigma_clip_mask(data, 3; exclude)'
1×6 adjoint(::BitVector) with eltype Bool:
 0  1  1  1  0  1
```

## Statistics on the retained values

`sigma_clipped_stats` always returns the final `center` and `spread`. Extra
statistics are requested with symbol-keyed pairs; each function receives a
vector of the retained values:

```jldoctest
julia> using Statistics

julia> data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0];

julia> sigma_clipped_stats(data, 3; :mean => mean, :n => length, :max => maximum)
(center = 3.0, spread = 1.4826022185056018, mean = 3.0, n = 5, max = 5.0)
```

Write pair keys as `:name => f`. Without the colon, `name` is looked up as a
variable in the calling scope. Every keyword other than `workspace`, `exclude`,
`center`, `spread` and `maxiter` is taken as a statistic, so a misspelled
keyword fails when its value is called on the retained values.

## Custom center and spread

Any callable that takes an `AbstractVector` and returns a scalar can be used
as `center` or `spread`. It may reorder its argument, which is SigmaClip's
internal buffer and not the user's array:

```jldoctest
julia> using Statistics

julia> data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0];

julia> iqr_spread(v) = (quantile(v, 0.75) - quantile(v, 0.25)) / 1.349;

julia> sigma_clip(data, 3; spread = iqr_spread)'
1×9 adjoint(::Vector{Float64}) with eltype Float64:
 1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  NaN
```

[`fast_median!`](@ref) and [`mad_std!`](@ref) can also be called directly.
Both reorder their input.

## Reusing a workspace

Every call needs scratch space for a copy of the finite values. When clipping
many arrays, allocate it once with [`SigmaClipWorkspace`](@ref) and pass it
through the `workspace` keyword:

```jldoctest
julia> image = repeat([0.0 1.0 2.0 3.0 4.0 5.0 6.0 50.0], 3);

julia> n = size(image, 2);

julia> workspace = SigmaClipWorkspace(Vector{Float64}(undef, n), Vector{Float64}(undef, n));

julia> for row in eachrow(image)
           sigma_clip!(row, 3; workspace)
       end

julia> image
3×8 Matrix{Float64}:
 0.0  1.0  2.0  3.0  4.0  5.0  6.0  NaN
 0.0  1.0  2.0  3.0  4.0  5.0  6.0  NaN
 0.0  1.0  2.0  3.0  4.0  5.0  6.0  NaN
```

The main buffer must have the input's element type and at least as many
elements as the input. The auxiliary buffer is used only by `mad_std!`; pass
`nothing` when the spread estimator does not need it:

```jldoctest
julia> using Statistics

julia> data = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0];

julia> workspace = SigmaClipWorkspace(similar(data), nothing);

julia> sigma_clip_bounds(data, 3; workspace, spread = std)
(-1.4721359549995796, 7.47213595499958)
```

### Custom workspace types

Any type can serve as a workspace by implementing
[`SigmaClip.workspace_buffer`](@ref) and [`SigmaClip.workspace_auxbuffer`](@ref),
with the same requirements on the returned buffers:

```jldoctest
julia> struct MyWorkspace{B, A}
           buf::B
           aux::A
       end

julia> SigmaClip.workspace_buffer(ws::MyWorkspace) = ws.buf;

julia> SigmaClip.workspace_auxbuffer(ws::MyWorkspace) = ws.aux;

julia> workspace = MyWorkspace(Vector{Float64}(undef, 16), Vector{Float64}(undef, 16));

julia> sigma_clip_bounds([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], 3; workspace)
(-1.4478066555168052, 7.447806655516805)
```
