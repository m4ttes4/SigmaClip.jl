SUITE["sigma_clip"] = BenchmarkGroup()

function _sigma_clip_input(n::Int)
    rng = MersenneTwister(0x51a6c11 + n)
    data = randn(rng, n)

    outlier_count = max(2, min(n, n ÷ 16))
    for i in 1:outlier_count
        idx = 1 + mod((i - 1) * 37, n)
        data[idx] = isodd(i) ? 12.0 + 0.1i : -12.0 - 0.1i
    end

    if n >= 8
        data[cld(n, 3)] = NaN
        data[cld(2n, 3)] = Inf
    end

    return data
end

let
    cases = (
        "n=128" => 128,
        "n=1024" => 1_024,
        "n=10000" => 10_000,
    )

    SUITE["sigma_clip"]["out_of_place"] = BenchmarkGroup()
    SUITE["sigma_clip"]["in_place"] = BenchmarkGroup()
    SUITE["sigma_clip"]["bounds"] = BenchmarkGroup()
    SUITE["sigma_clip"]["out_of_place"]["sigma_clip"] = BenchmarkGroup()
    SUITE["sigma_clip"]["out_of_place"]["sigma_clip_mask"] = BenchmarkGroup()
    SUITE["sigma_clip"]["in_place"]["sigma_clip!"] = BenchmarkGroup()
    SUITE["sigma_clip"]["in_place"]["sigma_clip_mask!"] = BenchmarkGroup()
    SUITE["sigma_clip"]["bounds"]["sigma_clip_bounds"] = BenchmarkGroup()

    for (label, n) in cases
        data = _sigma_clip_input(n)
        workspace = SigmaClipWorkspace(Float64, n)

        SUITE["sigma_clip"]["out_of_place"]["sigma_clip"][label] =
            @benchmarkable sigma_clip($data; workspace = $workspace)
        SUITE["sigma_clip"]["out_of_place"]["sigma_clip_mask"][label] =
            @benchmarkable sigma_clip_mask($data; workspace = $workspace)
        SUITE["sigma_clip"]["bounds"]["sigma_clip_bounds"][label] =
            @benchmarkable SigmaClip.sigma_clip_bounds($data; workspace = $workspace)

        SUITE["sigma_clip"]["in_place"]["sigma_clip!"][label] =
            @benchmarkable sigma_clip!(x; workspace = ws) setup = begin
                x = copy($data)
                ws = SigmaClipWorkspace(Float64, $n)
            end
        SUITE["sigma_clip"]["in_place"]["sigma_clip_mask!"][label] =
            @benchmarkable sigma_clip_mask!(x, target; workspace = ws) setup = begin
                x = copy($data)
                target = falses($n)
                ws = SigmaClipWorkspace(Float64, $n)
            end
    end
end
