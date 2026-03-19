using Test
using SigmaClip
using Statistics
using Random

# Reference median (sort-based, used to validate fast_median!)
ref_median(v) = begin
    s = sort(v)
    n = length(s)
    iseven(n) ? 0.5 * (s[n÷2] + s[n÷2+1]) : s[(n+1)÷2]
end

# Reference MAD-std
ref_mad_std(v) = begin
    m = ref_median(v)
    dev = abs.(v .- m)
    ref_median(dev) * 1.4826022185056018
end

# @testset "SigmaClip.jl" begin

    # ─────────────────────────────────────────────────────────────────────────
    @testset "fast_median!" begin

        @testset "correctness vs sort-based median" begin
            for n in [1, 2, 3, 4, 5, 10, 11, 99, 100, 1001]
                v = randn(n)
                @test SigmaClip.fast_median!(copy(v)) ≈ ref_median(v)
            end
        end

        @testset "specific small cases" begin
            @test SigmaClip.fast_median!([1.0]) == 1.0
            @test SigmaClip.fast_median!([1.0, 2.0]) == 1.5
            @test SigmaClip.fast_median!([3.0, 1.0, 2.0]) == 2.0
            @test SigmaClip.fast_median!([4.0, 1.0, 3.0, 2.0]) == 2.5
        end

        @testset "already sorted / reverse sorted" begin
            asc = collect(1.0:100.0)
            desc = reverse(asc)
            @test SigmaClip.fast_median!(asc) ≈ ref_median(1.0:100.0)
            @test SigmaClip.fast_median!(desc) ≈ ref_median(1.0:100.0)
        end

        @testset "all identical values" begin
            v = fill(3.14, 50)
            @test SigmaClip.fast_median!(v) == 3.14
        end

        @testset "two identical values" begin
            @test SigmaClip.fast_median!([5.0, 5.0]) == 5.0
        end

        @testset "preserves all values (only reorders)" begin
            v = randn(200)
            sv = sort(v)
            SigmaClip.fast_median!(v)
            @test sort(v) == sv
        end

        @testset "empty array returns zero" begin
            @test SigmaClip.fast_median!(Float64[]) == 0.0
            @test SigmaClip.fast_median!(Float32[]) == 0.0f0
        end

        @testset "zero allocations" begin
            v = randn(500)
            @test (@allocated SigmaClip.fast_median!(v)) == 0
        end

    end # fast_median!


    # ─────────────────────────────────────────────────────────────────────────
    @testset "SigmaClipWorkspace" begin

        @testset "constructor (type, n)" begin
            ws = SigmaClipWorkspace(Float64, 100)
            @test ws isa SigmaClipWorkspace{Float64}
            @test length(ws.buf) == 100
            @test length(ws.aux) == 100
        end

        @testset "constructor from Float array" begin
            v = randn(Float32, 50)
            ws = SigmaClipWorkspace(v)
            @test ws isa SigmaClipWorkspace{Float32}
            @test length(ws.buf) == 50
        end

        @testset "constructor from Integer array promotes to Float64" begin
            v = rand(Int, 80)
            ws = SigmaClipWorkspace(v)
            @test ws isa SigmaClipWorkspace{Float64}
            @test length(ws.buf) == 80
        end

        @testset "workspace too short raises ArgumentError" begin
            ws = SigmaClipWorkspace(Float64, 10)
            v = randn(20)
            @test_throws ArgumentError sigma_clip!(v; workspace=ws)
        end

        @testset "workspace exact length is accepted" begin
            v = randn(50)
            ws = SigmaClipWorkspace(Float64, 50)
            @test_nowarn sigma_clip!(copy(v); workspace=ws)
        end

        @testset "workspace larger than array is accepted" begin
            v = randn(30)
            ws = SigmaClipWorkspace(Float64, 100)
            @test_nowarn sigma_clip!(copy(v); workspace=ws)
        end

        @testset "zero allocations in hot loop with workspace" begin
            ws = SigmaClipWorkspace(Float64, 1000)
            row = randn(1000)
            # warm up
            sigma_clip!(copy(row); workspace=ws)
            allocs = @allocated sigma_clip!(copy(row); workspace=ws)
            # copy() itself allocates; test that passing ws causes no extra allocs
            # inside _sigma_clip_bounds by re-using a pre-copied buffer
            buf = copy(row)
            sigma_clip!(buf; workspace=ws)          # warm up
            allocs = @allocated sigma_clip!(buf; workspace=ws)
            @test allocs == 0
        end

    end # SigmaClipWorkspace


    # ─────────────────────────────────────────────────────────────────────────
    @testset "sigma_clip_bounds" begin

        @testset "basic convergence" begin
            # zeros(98) has MAD=0 once outliers are removed, so bounds converge
            # to [0, 0]. The meaningful check is that outliers are outside bounds.
            data = vcat(zeros(98), [100.0, -100.0])
            lb, ub = SigmaClip.sigma_clip_bounds(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test 100.0 > ub
            @test -100.0 < lb
        end

        @testset "returns (0,0) on all-NaN input" begin
            lb, ub = SigmaClip.sigma_clip_bounds(fill(NaN, 10))
            @test lb == 0.0 && ub == 0.0
        end

        @testset "returns (0,0) on all-Inf input" begin
            lb, ub = SigmaClip.sigma_clip_bounds(fill(Inf, 10))
            @test lb == 0.0 && ub == 0.0
        end

        @testset "returns (0,0) on empty array" begin
            lb, ub = SigmaClip.sigma_clip_bounds(Float64[])
            @test lb == 0.0 && ub == 0.0
        end

        @testset "single element" begin
            lb, ub = SigmaClip.sigma_clip_bounds([5.0])
            @test isfinite(lb) && isfinite(ub)
        end

        @testset "maxiter=1 stops after one iteration" begin
            # With very tight sigma, multiple iterations would clip more
            data = Float64[1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 10, 50]
            lb1, ub1 = SigmaClip.sigma_clip_bounds(data; maxiter=1,
                cent_reducer=fast_median, std_reducer=mad_std)
            lb5, ub5 = SigmaClip.sigma_clip_bounds(data; maxiter=5,
                cent_reducer=fast_median, std_reducer=mad_std)
            # 5 iterations should converge to tighter or equal bounds
            @test ub5 <= ub1 + 1e-10
        end

        @testset "maxiter=-1 runs until convergence" begin
            data = vcat(ones(50), [1000.0])
            lb, ub = SigmaClip.sigma_clip_bounds(data; maxiter=-1,
                cent_reducer=fast_median, std_reducer=mad_std)
            @test ub < 100.0
        end

        @testset "mask excludes values from bound computation" begin
            data = [0.0, 0.0, 0.0, 0.0, 50.0]
            mask = falses(5)
            mask[end] = false   # mark 50.0 as already-bad

            # With mask: 50.0 excluded from stats, bounds should be tight
            mask_flagged = trues(5)
            mask_flagged[end] = true
            lb_m, ub_m = SigmaClip.sigma_clip_bounds(data; mask=mask_flagged,
                cent_reducer=fast_median, std_reducer=mad_std)
            lb_u, ub_u = SigmaClip.sigma_clip_bounds(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test ub_m <= ub_u
        end

        @testset "asymmetric sigma" begin
            # Uses std_reducer=std: zeros-dominated data has MAD=0, which would
            # collapse bounds to [0,0] regardless of sigma_lower/sigma_upper.
            # std is non-zero here and correctly exercises asymmetric thresholds.
            data = vcat(zeros(50), [10.0], [-10.0])
            lb, ub = SigmaClip.sigma_clip_bounds(data;
                sigma_lower=100.0, sigma_upper=2.0,
                cent_reducer=fast_median, std_reducer=std)
            # tight upper threshold: ub is well below 10.0
            @test ub < 10.0
            # very permissive lower threshold: lb is well below -10.0
            @test lb < -10.0
        end

    end # sigma_clip_bounds


    # ─────────────────────────────────────────────────────────────────────────
    @testset "sigma_clip! (in-place)" begin

        @testset "high outlier is clipped" begin
            data = vcat(zeros(Float64, 99), [1000.0])
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[end])
            @test all(==(0.0), data[1:99])
        end

        @testset "low outlier is clipped" begin
            data = vcat([-1000.0], zeros(Float64, 99))
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[1])
            @test all(==(0.0), data[2:end])
        end

        @testset "both extremes clipped independently" begin
            data = vcat(zeros(Float64, 98), [500.0, -500.0])
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[99])
            @test isnan(data[100])
            @test count(isnan, data) == 2
        end

        @testset "NaN input stays NaN" begin
            data = [1.0, 2.0, NaN, 3.0]
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[3])
        end

        @testset "Inf is always clipped" begin
            data = vcat(ones(Float64, 10), [Inf])
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[end])
        end

        @testset "-Inf is always clipped" begin
            data = vcat(ones(Float64, 10), [-Inf])
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[end])
        end

        @testset "outlier at first element" begin
            data = vcat([1000.0], zeros(Float64, 50))
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[1])
        end

        @testset "outlier at last element" begin
            data = vcat(zeros(Float64, 50), [1000.0])
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[end])
        end

        @testset "constant array — nothing clipped" begin
            data = fill(3.0, 50)
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test all(==(3.0), data)
        end

        @testset "very large sigma — nothing clipped" begin
            data = randn(100)
            original = copy(data)
            sigma_clip!(data; sigma_lower=1000.0, sigma_upper=1000.0,
                cent_reducer=fast_median, std_reducer=mad_std)
            @test all(!isnan, data)
        end

        @testset "returns the modified array" begin
            data = randn(20)
            result = sigma_clip!(data)
            @test result === data
        end

        @testset "mask kwarg shields value from bound computation" begin
            data = Float64[0, 0, 0, 0, 50]
            mask = trues(5)
            mask[5] = true   # treat index 5 as already bad
            original = copy(data)
            sigma_clip!(data; mask=mask,
                cent_reducer=fast_median, std_reducer=mad_std)
            # 50.0 was excluded from stats: bounds are tight around 0
            # so 50.0 should also be clipped in the output
            @test isnan(data[5])
            @test all(==(0.0), data[1:4])
        end

        @testset "asymmetric sigma_upper clips high only" begin
            # std_reducer=std: zeros-based data has MAD=0, which defeats asymmetry.
            # std correctly produces non-zero dispersion here.
            data = vcat(zeros(Float64, 50), [100.0], [-100.0])
            sigma_clip!(data; sigma_lower=1000.0, sigma_upper=2.0,
                cent_reducer=fast_median, std_reducer=std)
            @test isnan(data[51])      # high outlier clipped
            @test !isnan(data[52])     # low outlier kept
        end

        @testset "asymmetric sigma_lower clips low only" begin
            data = vcat(zeros(Float64, 50), [100.0], [-100.0])
            sigma_clip!(data; sigma_lower=2.0, sigma_upper=1000.0,
                cent_reducer=fast_median, std_reducer=std)
            @test !isnan(data[51])     # high outlier kept
            @test isnan(data[52])      # low outlier clipped
        end

        @testset "integer input raises MethodError" begin
            @test_throws MethodError sigma_clip!([1, 2, 3, 100])
        end

        @testset "Float32 input preserved as Float32" begin
            data = Float32[1, 1, 1, 1, 100]
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test eltype(data) == Float32
            @test isnan(data[end])
        end

    end # sigma_clip!


    # ─────────────────────────────────────────────────────────────────────────
    @testset "sigma_clip (out-of-place)" begin

        @testset "original array is never modified" begin
            original = vcat(zeros(Float64, 99), [1000.0])
            backup = copy(original)
            sigma_clip(original; cent_reducer=fast_median, std_reducer=mad_std)
            @test original == backup
        end

        @testset "outlier replaced in returned copy" begin
            data = vcat(zeros(Float64, 99), [1000.0])
            cleaned = sigma_clip(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(cleaned[end])
        end

        @testset "integer input promoted to Float64" begin
            data = [1, 1, 1, 1, 100]
            result = sigma_clip(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test result isa Vector{Float64}
            @test isnan(result[end])
        end

        @testset "result matches sigma_clip!" begin
            data = randn(200)
            data[42] = 999.0
            r1 = sigma_clip(data; cent_reducer=fast_median, std_reducer=mad_std)
            r2 = sigma_clip!(copy(data); cent_reducer=fast_median, std_reducer=mad_std)
            for i in eachindex(r1)
                if isnan(r1[i])
                    @test isnan(r2[i])
                else
                    @test r1[i] == r2[i]
                end
            end
        end

    end # sigma_clip


    # ─────────────────────────────────────────────────────────────────────────
    @testset "sigma_clip_mask / sigma_clip_mask!" begin

        @testset "known outliers are marked true" begin
            data = vcat(zeros(Float64, 98), [500.0, -500.0])
            mask = sigma_clip_mask(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test mask[99] == true
            @test mask[100] == true
        end

        @testset "clean elements are marked false" begin
            data = randn(200)
            data[1] = 1e6
            mask = sigma_clip_mask(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test mask[1] == true
            # the vast majority of normally distributed values should survive
            @test count(mask) < 10
        end

        @testset "NaN input element is marked true" begin
            data = [1.0, 2.0, NaN, 3.0]
            mask = sigma_clip_mask(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test mask[3] == true
        end

        @testset "Inf input element is marked true" begin
            data = vcat(ones(Float64, 20), [Inf])
            mask = sigma_clip_mask(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test mask[end] == true
        end

        @testset "result size matches input" begin
            data = randn(77)
            @test size(sigma_clip_mask(data)) == size(data)
        end

        @testset "sigma_clip_mask! writes into pre-allocated target" begin
            data = vcat(zeros(Float64, 50), [999.0])
            target = falses(51)
            sigma_clip_mask!(data, target;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test target[end] == true
            @test count(target) == 1
        end

        @testset "sigma_clip_mask! returns target" begin
            data = randn(20)
            target = falses(20)
            result = sigma_clip_mask!(data, target)
            @test result === target
        end

        @testset "consistency: mask NaN iff sigma_clip! returns NaN" begin
            data = randn(300)
            data[77] = 1e5
            data[200] = NaN
            mask = sigma_clip_mask(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            clipped = sigma_clip(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            for i in eachindex(data)
                @test mask[i] == isnan(clipped[i])
            end
        end

    end # sigma_clip_mask


    # ─────────────────────────────────────────────────────────────────────────
    @testset "reducer dispatch paths" begin

        BASE = vcat(zeros(Float64, 98), [500.0, -500.0])

        @testset "(fast_median, mad_std) — specialised, shared median" begin
            data = copy(BASE)
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(data[99]) && isnan(data[100])
            @test count(isnan, data) == 2
        end

        @testset "(fast_median, std) — specialised centre, generic std" begin
            data = copy(BASE)
            sigma_clip!(data; cent_reducer=fast_median, std_reducer=std)
            @test isnan(data[99]) && isnan(data[100])
        end

        @testset "(fast_median!, std) — function as cent_reducer, generic fallback" begin
            # Passing the mutating function, not the sentinel — still works
            data = copy(BASE)
            sigma_clip!(data; cent_reducer=fast_median!, std_reducer=std)
            @test isnan(data[99]) && isnan(data[100])
        end

        @testset "(mean, std) — fully generic fallback" begin
            using_mean(v) = sum(v) / length(v)
            data = copy(BASE)
            sigma_clip!(data; cent_reducer=using_mean, std_reducer=std)
            # mean is less robust but should still catch a 5-sigma outlier
            @test isnan(data[99]) || isnan(data[100])
        end

        @testset "custom cent and std with known bounds" begin
            # Fixed centre at 0, fixed spread of 1 → bounds are [-50, 50]
            data = copy(BASE)
            sigma_clip!(data;
                cent_reducer=_ -> 0.0,
                std_reducer=_ -> 1.0,
                sigma_lower=50.0,
                sigma_upper=50.0)
            @test isnan(data[99]) && isnan(data[100])
            @test all(==(0.0), data[1:98])
        end

        @testset "mad_std result matches manual calculation" begin
            v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            ws = SigmaClipWorkspace(Float64, length(v))
            buf = copy(v)
            _, s = SigmaClip._compute_stats(
                SigmaClip.fast_median, SigmaClip.mad_std,
                buf, length(buf), ws)
            @test s ≈ ref_mad_std(v)
        end

    end # reducer dispatch


    # ─────────────────────────────────────────────────────────────────────────
    @testset "outlier detection correctness" begin

        # Fixed seed for reproducibility
        Random.seed!(42)

        @testset "single obvious outlier always detected" begin
            for _ in 1:20
                data = randn(500)
                idx = rand(1:500)
                data[idx] = 1000.0
                mask = sigma_clip_mask(data;
                    cent_reducer=fast_median, std_reducer=mad_std)
                @test mask[idx] == true
            end
        end

        @testset "false positive rate on clean normal data is low" begin
            data = randn(10_000)
            mask = sigma_clip_mask(data; sigma_lower=3.0, sigma_upper=3.0,
                cent_reducer=fast_median, std_reducer=mad_std)
            # For σ=3 and normal data, theoretical false positive rate ≈ 0.27%
            # We allow up to 1% to be safe
            @test count(mask) / length(data) < 0.01
        end

        @testset "multiple outliers at different magnitudes" begin
            data = vcat(zeros(Float64, 97), [10.0, 100.0, 1000.0])
            mask = sigma_clip_mask(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test mask[98] && mask[99] && mask[100]
        end

        @testset "outliers at both ends of distribution" begin
            data = vcat([-200.0], zeros(Float64, 98), [200.0])
            mask = sigma_clip_mask(data;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test mask[1] && mask[end]
        end

        @testset "convergence: iterative clipping catches cascading outliers" begin
            # 50 is a mild outlier; 1000 inflates std so 50 might survive iter 1
            data = vcat(zeros(Float64, 97), [50.0, 100.0, 1000.0])
            mask_1 = sigma_clip_mask(data; maxiter=1,
                cent_reducer=fast_median, std_reducer=mad_std)
            mask_5 = sigma_clip_mask(data; maxiter=5,
                cent_reducer=fast_median, std_reducer=mad_std)
            # More iterations ⟹ at least as many outliers found
            @test count(mask_5) >= count(mask_1)
        end

        @testset "2-D array: each element treated independently" begin
            img = zeros(Float64, 10, 10)
            img[3, 7] = 999.0
            mask = sigma_clip_mask(img;
                cent_reducer=fast_median, std_reducer=mad_std)
            @test mask[3, 7] == true
            @test count(mask) == 1
        end

    end # outlier detection correctness


    # ─────────────────────────────────────────────────────────────────────────
    @testset "edge cases" begin

        @testset "length-0 array" begin
            @test_nowarn sigma_clip(Float64[])
            @test_nowarn sigma_clip_mask(Float64[])
        end

        @testset "length-1 array — no clipping" begin
            data = [42.0]
            result = sigma_clip(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test !isnan(result[1])
        end

        @testset "length-2 array" begin
            data = [1.0, 1000.0]
            result = sigma_clip(data; cent_reducer=fast_median, std_reducer=mad_std)
            # With only 2 points MAD is 0, std is non-zero — just verify no crash
            @test length(result) == 2
        end

        @testset "all-NaN array" begin
            data = fill(NaN, 20)
            @test_nowarn sigma_clip!(data)
            @test all(isnan, data)
        end

        @testset "all-Inf array" begin
            data = fill(Inf, 20)
            result = sigma_clip(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test all(isnan, result)
        end

        @testset "mixed NaN and Inf" begin
            data = [1.0, NaN, Inf, -Inf, 2.0, 3.0]
            result = sigma_clip(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test isnan(result[2])
            @test isnan(result[3])
            @test isnan(result[4])
        end

        @testset "array with a single finite element (rest NaN)" begin
            data = fill(NaN, 10)
            data[5] = 1.0
            @test_nowarn sigma_clip!(data)
        end

        @testset "array with two finite elements (rest NaN)" begin
            data = fill(NaN, 10)
            data[3] = 1.0
            data[7] = 2.0
            @test_nowarn sigma_clip!(data)
        end

        @testset "input already clean — no NaN introduced" begin
            data = collect(1.0:10.0)
            result = sigma_clip(data; cent_reducer=fast_median, std_reducer=mad_std)
            @test !any(isnan, result)
        end

    end # edge cases

# end # SigmaClip.jl