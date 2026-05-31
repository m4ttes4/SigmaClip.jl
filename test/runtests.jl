using Test
using SigmaClip
using Statistics

struct PublicAPIWorkspace{T, A}
    buf::Vector{T}
    aux::Vector{A}
end

struct NoAuxWorkspace{T}
    buf::Vector{T}
end

PublicAPIWorkspace(::Type{T}, n::Integer) where {T} =
    PublicAPIWorkspace(Vector{T}(undef, n), Vector{float(T)}(undef, n))
PublicAPIWorkspace(x::AbstractArray{T}) where {T} =
    PublicAPIWorkspace(T, length(x))

SigmaClip.workspace_buffer(ws::PublicAPIWorkspace) = ws.buf
SigmaClip.workspace_auxbuffer(ws::PublicAPIWorkspace) = ws.aux
SigmaClip.workspace_buffer(ws::NoAuxWorkspace) = ws.buf
SigmaClip.workspace_auxbuffer(::NoAuxWorkspace) = nothing

struct WorkspaceCenter end
struct WorkspaceSpread end

SigmaClip.statistic(::WorkspaceCenter, ws::PublicAPIWorkspace, n::Int) =
    mean(@view SigmaClip.workspace_buffer(ws)[1:n])

SigmaClip.statistic(::WorkspaceSpread, ws::PublicAPIWorkspace, n::Int) = begin
    data = @view SigmaClip.workspace_buffer(ws)[1:n]
    aux = @view SigmaClip.workspace_auxbuffer(ws)[1:n]
    c = mean(data)
    @inbounds for i in eachindex(data)
        aux[i] = abs(data[i] - c)
    end
    return mean(aux)
end

plain_center(v) = mean(v)
plain_spread(v) = maximum(abs, v .- mean(v)) / 2

function same_nan_pattern(a, b)
    axes(a) == axes(b) || return false
    for i in eachindex(a, b)
        if isnan(a[i]) || isnan(b[i])
            isnan(a[i]) && isnan(b[i]) || return false
        elseif a[i] != b[i]
            return false
        end
    end
    return true
end

@testset "Workspace Traits" begin
    @testset "alloca aux solo per statistiche che lo richiedono" begin
        data = [1, 2, 3, 100]

        ws_mad = SigmaClip.prepare_ws(data, mad_std!, nothing)
        @test SigmaClip.workspace_buffer(ws_mad) isa Vector{Int}
        @test length(SigmaClip.workspace_buffer(ws_mad)) == length(data)
        @test SigmaClip.workspace_auxbuffer(ws_mad) isa Vector{Float64}
        @test length(SigmaClip.workspace_auxbuffer(ws_mad)) == length(data)

        ws_std = SigmaClip.prepare_ws(data, std, nothing)
        @test SigmaClip.workspace_buffer(ws_std) isa Vector{Int}
        @test SigmaClip.workspace_auxbuffer(ws_std) === nothing

        ws_custom = SigmaClip.prepare_ws(data, plain_spread, nothing)
        @test SigmaClip.workspace_buffer(ws_custom) isa Vector{Int}
        @test SigmaClip.workspace_auxbuffer(ws_custom) === nothing
    end

    @testset "workspace custom senza aux" begin
        data = Float64[1, 2, 3, 100]
        ws = NoAuxWorkspace(Vector{Float64}(undef, length(data)))

        @test SigmaClip.prepare_ws(data, std, ws) === ws
        @test_throws ArgumentError SigmaClip.prepare_ws(data, mad_std!, ws)
    end
end

@testset "sigma_clip_bounds" begin
    @testset "diversi tipi di input" begin
        @test SigmaClip.sigma_clip_bounds(Float64[0, 0, 0, 10]) == (0.0, 0.0)

        lb32, ub32 = SigmaClip.sigma_clip_bounds(Float32[1, 1, 1, 9])
        @test lb32 === 1.0f0
        @test ub32 === 1.0f0

        matrix = reshape(Float64[1, 1, 1, 1, 99, -99], 2, 3)
        lb, ub = SigmaClip.sigma_clip_bounds(matrix)
        @test lb == 1.0
        @test ub == 1.0
    end

    @testset "outliers chiari" begin
        low, high = SigmaClip.sigma_clip_bounds(vcat(fill(2.0, 10), [100.0, -100.0]))
        @test low == 2.0
        @test high == 2.0

        low, high = SigmaClip.sigma_clip_bounds(
            Float64[-100, 0, 0, 0, 0, 50];
            exclude = Bool[1, 0, 0, 0, 0, 1],
        )
        @test low == 0.0
        @test high == 0.0
    end

    @testset "NaN e Inf" begin
        low, high = SigmaClip.sigma_clip_bounds([0.0, 0.0, 0.0, NaN, Inf, -Inf, 50.0])
        @test low == 0.0
        @test high == 0.0
    end

    @testset "statistiche custom" begin
        data = [-4.0, -1.0, 0.0, 1.0, 4.0]
        low, high = SigmaClip.sigma_clip_bounds(
            data;
            center = plain_center,
            spread = plain_spread,
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )
        @test low == -2.0
        @test high == 2.0
    end

    @testset "workspace custom" begin
        data = Float64[0, 0, 0, 0, 100]
        ws = PublicAPIWorkspace(data)

        @test SigmaClip.sigma_clip_bounds(data; workspace = ws) == (0.0, 0.0)
        @test ws.buf[1:4] == zeros(4)
    end

    @testset "workspace-aware statistics" begin
        data = [-10.0, -1.0, 0.0, 1.0, 10.0]
        ws = PublicAPIWorkspace(data)
        low, high = SigmaClip.sigma_clip_bounds(
            data;
            workspace = ws,
            center = WorkspaceCenter(),
            spread = WorkspaceSpread(),
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )

        @test low == -4.4
        @test high == 4.4
    end
end

@testset "sigma_clip_mask" begin
    @testset "diversi tipi di input" begin
        @test sigma_clip_mask(Float64[1, 1, 1, 9]) == Bool[1, 1, 1, 0]
        @test sigma_clip_mask(Float32[2, 2, 2, -8]) == Bool[1, 1, 1, 0]
        @test sigma_clip_mask(reshape(Float64[0, 0, 0, 7], 2, 2)) == Bool[1 1; 1 0]
    end

    @testset "outliers chiari" begin
        data = vcat(fill(0.0, 8), [100.0, -100.0])
        mask = sigma_clip_mask(data)

        @test count(mask) == 8
        @test mask[1:8] == trues(8)
        @test mask[9:10] == falses(2)
    end

    @testset "NaN e Inf" begin
        mask = sigma_clip_mask([1.0, 1.0, 1.0, NaN, Inf, -Inf, 99.0])
        @test mask == Bool[1, 1, 1, 0, 0, 0, 0]
    end

    @testset "statistiche custom" begin
        mask = sigma_clip_mask(
            [-4.0, -1.0, 0.0, 1.0, 4.0];
            center = plain_center,
            spread = plain_spread,
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )
        @test mask == Bool[0, 1, 1, 1, 0]
    end

    @testset "workspace custom" begin
        data = Float64[0, 0, 0, 0, 100]
        ws = PublicAPIWorkspace(data)
        @test sigma_clip_mask(data; workspace = ws) == Bool[1, 1, 1, 1, 0]
    end

    @testset "workspace-aware statistics" begin
        data = [-10.0, -1.0, 0.0, 1.0, 10.0]
        ws = PublicAPIWorkspace(data)
        mask = sigma_clip_mask(
            data;
            workspace = ws,
            center = WorkspaceCenter(),
            spread = WorkspaceSpread(),
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )
        @test mask == Bool[0, 1, 1, 1, 0]
    end
end

@testset "sigma_clip_mask!" begin
    @testset "diversi tipi di input" begin
        target = trues(4)
        result = sigma_clip_mask!(Float64[1, 1, 1, 9], target)
        @test result === target
        @test target == Bool[1, 1, 1, 0]

        matrix = reshape(Float32[2, 2, 2, -8], 2, 2)
        target_matrix = falses(size(matrix))
        sigma_clip_mask!(matrix, target_matrix)
        @test target_matrix == Bool[1 1; 1 0]
    end

    @testset "outliers chiari" begin
        data = vcat(fill(3.0, 6), 30.0)
        target = falses(length(data))
        sigma_clip_mask!(data, target)

        @test target == Bool[1, 1, 1, 1, 1, 1, 0]
    end

    @testset "NaN e Inf" begin
        data = [5.0, 5.0, 5.0, NaN, Inf, -Inf, -50.0]
        target = trues(length(data))
        sigma_clip_mask!(data, target)

        @test target == Bool[1, 1, 1, 0, 0, 0, 0]
    end

    @testset "non alloca con workspace custom" begin
        data = vcat(fill(1.0, 64), 99.0)
        target = falses(length(data))
        ws = PublicAPIWorkspace(data)

        sigma_clip_mask!(data, target; workspace = ws)
        allocated = @allocated sigma_clip_mask!(data, target; workspace = ws)

        @test allocated == 0
        @test target[end] == false
        @test all(target[1:(end - 1)])
    end

    @testset "statistiche custom" begin
        data = [-4.0, -1.0, 0.0, 1.0, 4.0]
        target = falses(length(data))
        sigma_clip_mask!(
            data,
            target;
            center = plain_center,
            spread = plain_spread,
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )

        @test target == Bool[0, 1, 1, 1, 0]
    end

    @testset "workspace-aware statistics" begin
        data = [-10.0, -1.0, 0.0, 1.0, 10.0]
        target = falses(length(data))
        ws = PublicAPIWorkspace(data)
        sigma_clip_mask!(
            data,
            target;
            workspace = ws,
            center = WorkspaceCenter(),
            spread = WorkspaceSpread(),
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )

        @test target == Bool[0, 1, 1, 1, 0]
    end
end

@testset "sigma_clip!" begin
    @testset "diversi tipi di input" begin
        data = Float64[1, 1, 1, 9]
        @test sigma_clip!(data) === data
        @test same_nan_pattern(data, [1.0, 1.0, 1.0, NaN])

        data32 = Float32[2, 2, 2, -8]
        sigma_clip!(data32)
        @test eltype(data32) == Float32
        @test same_nan_pattern(data32, Float32[2, 2, 2, NaN])

        matrix = reshape(Float64[0, 0, 0, 7], 2, 2)
        sigma_clip!(matrix)
        @test same_nan_pattern(matrix, [0.0 0.0; 0.0 NaN])
    end

    @testset "outliers chiari" begin
        data = vcat(fill(0.0, 8), [100.0, -100.0])
        sigma_clip!(data)

        @test all(==(0.0), data[1:8])
        @test isnan(data[9])
        @test isnan(data[10])
    end

    @testset "NaN e Inf" begin
        data = [1.0, 1.0, 1.0, NaN, Inf, -Inf, 99.0]
        sigma_clip!(data)

        @test same_nan_pattern(data, [1.0, 1.0, 1.0, NaN, NaN, NaN, NaN])
    end

    @testset "non alloca con workspace custom" begin
        data = vcat(fill(1.0, 64), 99.0)
        ws = PublicAPIWorkspace(data)

        sigma_clip!(data; workspace = ws)
        data .= vcat(fill(1.0, 64), 99.0)
        allocated = @allocated sigma_clip!(data; workspace = ws)

        @test allocated == 0
        @test all(==(1.0), data[1:(end - 1)])
        @test isnan(data[end])
    end

    @testset "statistiche custom" begin
        data = [-4.0, -1.0, 0.0, 1.0, 4.0]
        sigma_clip!(
            data;
            center = plain_center,
            spread = plain_spread,
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )

        @test same_nan_pattern(data, [NaN, -1.0, 0.0, 1.0, NaN])
    end

    @testset "workspace-aware statistics" begin
        data = [-10.0, -1.0, 0.0, 1.0, 10.0]
        ws = PublicAPIWorkspace(data)
        sigma_clip!(
            data;
            workspace = ws,
            center = WorkspaceCenter(),
            spread = WorkspaceSpread(),
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )

        @test same_nan_pattern(data, [NaN, -1.0, 0.0, 1.0, NaN])
    end
end

@testset "sigma_clip" begin
    @testset "diversi tipi di input" begin
        data = Float64[1, 1, 1, 9]
        clipped = sigma_clip(data)
        @test data == [1.0, 1.0, 1.0, 9.0]
        @test same_nan_pattern(clipped, [1.0, 1.0, 1.0, NaN])

        int_clipped = sigma_clip([2, 2, 2, -8])
        @test int_clipped isa Vector{Float64}
        @test same_nan_pattern(int_clipped, [2.0, 2.0, 2.0, NaN])

        matrix = reshape(Float32[0, 0, 0, 7], 2, 2)
        clipped_matrix = sigma_clip(matrix)
        @test eltype(clipped_matrix) == Float32
        @test same_nan_pattern(clipped_matrix, Float32[0 0; 0 NaN])
    end

    @testset "outliers chiari" begin
        data = vcat(fill(4.0, 8), [400.0, -400.0])
        clipped = sigma_clip(data)

        @test all(==(4.0), clipped[1:8])
        @test isnan(clipped[9])
        @test isnan(clipped[10])
        @test data[9] == 400.0
        @test data[10] == -400.0
    end

    @testset "NaN e Inf" begin
        clipped = sigma_clip([3.0, 3.0, 3.0, NaN, Inf, -Inf, 30.0])
        @test same_nan_pattern(clipped, [3.0, 3.0, 3.0, NaN, NaN, NaN, NaN])
    end

    @testset "statistiche custom" begin
        clipped = sigma_clip(
            [-4.0, -1.0, 0.0, 1.0, 4.0];
            center = plain_center,
            spread = plain_spread,
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )

        @test same_nan_pattern(clipped, [NaN, -1.0, 0.0, 1.0, NaN])
    end

    @testset "workspace custom" begin
        data = Float64[0, 0, 0, 0, 100]
        ws = PublicAPIWorkspace(data)
        clipped = sigma_clip(data; workspace = ws)

        @test same_nan_pattern(clipped, [0.0, 0.0, 0.0, 0.0, NaN])
        @test data == [0.0, 0.0, 0.0, 0.0, 100.0]
    end

    @testset "workspace-aware statistics" begin
        data = [-10.0, -1.0, 0.0, 1.0, 10.0]
        ws = PublicAPIWorkspace(data)
        clipped = sigma_clip(
            data;
            workspace = ws,
            center = WorkspaceCenter(),
            spread = WorkspaceSpread(),
            sigma_lower = 1,
            sigma_upper = 1,
            maxiter = 1,
        )

        @test same_nan_pattern(clipped, [NaN, -1.0, 0.0, 1.0, NaN])
    end
end
