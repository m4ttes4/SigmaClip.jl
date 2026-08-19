using SigmaClip
using Statistics
using Test

@testset "statistics" begin
    @test fast_median!([3.0, 1.0, 2.0]) == 2.0
    @test fast_median!([4.0, 1.0, 3.0, 2.0]) == 2.5
    @test mad_std!([1.0, 1.0, 1.0, 10.0]) == 0.0
end

@testset "bounds" begin
    data = [0.0, 0.0, 0.0, NaN, Inf, -Inf, 50.0]
    @test sigma_clip_bounds(data) == (0.0, 0.0)
    @test sigma_clip_bounds(Float32[1, 1, 1, 9]) == (1.0f0, 1.0f0)
    @test sigma_clip_bounds(reshape([1.0, 1.0, 1.0, 99.0], 2, 2)) == (1.0, 1.0)

    excluded = Bool[true, false, false, false, true]
    @test sigma_clip_bounds([-100.0, 0.0, 0.0, 0.0, 50.0]; exclude = excluded) == (0.0, 0.0)

    plain_spread(x) = maximum(abs, x .- mean(x)) / 2
    @test sigma_clip_bounds(
        [-4.0, -1.0, 0.0, 1.0, 4.0];
        center = mean,
        spread = plain_spread,
        sigma_lower = 1,
        sigma_upper = 1,
        maxiter = 1,
    ) == (-2.0, 2.0)

    @test_throws ArgumentError sigma_clip_bounds(Float64[])
    @test_throws ArgumentError sigma_clip_bounds([1.0]; sigma_lower = 0)
    @test_throws ArgumentError sigma_clip_bounds([1.0]; maxiter = 0)
    @test_throws DimensionMismatch sigma_clip_bounds([1.0]; exclude = falses(2))
end

@testset "workspace" begin
    data = vcat(fill(1.0, 64), 99.0)
    target = falses(length(data))
    workspace = SigmaClipWorkspace(similar(data), similar(data))

    sigma_clip_mask!(data, target; workspace)
    @test (@allocated sigma_clip_mask!(data, target; workspace)) == 0
    @test target == vcat(trues(64), false)

    no_aux = SigmaClipWorkspace(similar(data), nothing)
    @test sigma_clip_bounds(data; workspace = no_aux, spread = std) isa Tuple
    @test_throws ArgumentError sigma_clip_bounds(data; workspace = no_aux)
end

@testset "public wrappers" begin
    data = [1.0, 1.0, 1.0, 9.0, NaN]
    expected = Bool[true, true, true, false, false]
    @test sigma_clip_mask(data) == expected

    target = trues(length(data))
    @test sigma_clip_mask!(data, target) === target
    @test target == expected

    inplace = copy(data)
    @test sigma_clip!(inplace) === inplace
    @test isequal(inplace, [1.0, 1.0, 1.0, NaN, NaN])
    @test_throws ArgumentError sigma_clip!([1, 1, 1, 9])

    @test isequal(sigma_clip(data), [1.0, 1.0, 1.0, NaN, NaN])
    @test isequal(sigma_clip([2, 2, 2, 8]), [2.0, 2.0, 2.0, NaN])
    @test isequal(data, [1.0, 1.0, 1.0, 9.0, NaN])
end
