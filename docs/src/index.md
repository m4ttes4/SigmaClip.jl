```@meta
DocTestSetup = :(using SigmaClip)
```

# SigmaClip.jl

SigmaClip.jl finds outliers in numeric arrays with iterative sigma clipping.
It returns clipped copies, clips in place, builds validity masks, reports the
final clipping bounds, and computes statistics on the retained values. Scratch
buffers can be reused across calls, and the center and spread estimators can be
replaced with any function. The package has no runtime dependencies.

## Installation

```julia
using Pkg
Pkg.add("SigmaClip")
```

## How sigma clipping works

Each call runs the following loop on a private copy of the input:

1. Collect the finite values of `x`. `NaN`, `Inf`, and entries marked `true`
   in `exclude` do not take part in estimating the bounds.
2. Estimate a center ``c`` and a spread ``s`` of the collected values. The
   defaults are the median ([`fast_median!`](@ref)) and the median absolute
   deviation scaled to a normal standard deviation ([`mad_std!`](@ref)).
3. Compute the bounds ``[c - \sigma_\text{lower} s,\; c + \sigma_\text{upper} s]``
   and drop the values outside them.
4. Repeat from step 2 until no value is dropped, `maxiter` iterations have run
   (`maxiter = -1` means no limit), or fewer than two values remain.

After the loop, every element of `x` is compared with the final bounds:

- a finite value inside the bounds is **retained**;
- a finite value outside the bounds is an **outlier**;
- a non-finite value is invalid, but it is not an outlier.

The **validity mask** returned by [`sigma_clip_mask`](@ref) is `true` for
retained values and `false` for outliers and non-finite values.
[`sigma_clip`](@ref) and [`sigma_clip!`](@ref) replace every value where the
mask is `false` with `NaN`.

Because `exclude` only affects step 1, excluded values are still compared with
the final bounds and can be retained or classified as outliers.

## Quick start

```jldoctest quickstart
julia> data = [0, 1, 2, 3, 4, 5, 6, 50, NaN, Inf];

julia> sigma_clip(data)
10-element Vector{Float64}:
   0.0
   1.0
   2.0
   3.0
   4.0
   5.0
   6.0
 NaN
 NaN
 NaN

julia> findall(sigma_clip_mask(data))
7-element Vector{Int64}:
 1
 2
 3
 4
 5
 6
 7

julia> sigma_clip_bounds(data)
(-5.89561331103361, 11.89561331103361)

julia> sigma_clipped_stats(data)
(center = 3.0, spread = 2.9652044370112036)
```

These functions leave `data` unchanged. See the [Guide](@ref) for in-place
clipping, excluded values, custom statistics, and reusable workspaces.
