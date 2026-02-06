using Test
using Statistics


using SigmaClip

@testset "SigmaClip Module Tests" begin

    @testset "Internal Helpers: fast_median!" begin


        for n in [1, 5, 10, 100, 101]
            data = randn(n)
            # Testiamo contro l'implementazione standard di Julia
            @test SigmaClip.fast_median!(copy(data)) ≈ median(data)
        end

        # Test caso pari vs dispari specifici
        @test SigmaClip.fast_median!([1.0, 3.0, 2.0]) == median([1.0, 3.0, 2.0])
        rd = rand(100)
        @test SigmaClip.fast_median!(rd) ≈ median(rd)
        @test SigmaClip.fast_median!(Float64[]) == 0.0

        allocs = @allocated SigmaClip.fast_median!(rd)
        @test allocs == 0

    end

    @testset "sigma_clip! (In-place Values)" begin

        data = zeros(100)
        data[50] = 1000.0

        # Eseguiamo il clip
        sigma_clip!(data, sigma_lower=3, sigma_upper=3)


        @test isnan(data[50])

        @test count(x -> x == 0.0, data) == 99
        @test count(isnan, data) == 1

        int_data = [1, 2, 100]
        @test_throws MethodError sigma_clip!(int_data)
    end

    @testset "sigma_clip (Out-of-place Values)" begin
        
        original_data = zeros(Float64, 100)
        data = copy(original_data)
        data[50] = 1000.0

        # Clip out-of-place
        cleaned = sigma_clip(data, sigma_lower=2, sigma_upper=2)

        # Verifica che l'originale non sia cambiato

        # Verifica risultato
        @test count(x -> x == 0.0, cleaned) == 99
        @test isnan(cleaned[50])
        @test original_data[50] == 0
    end

    @testset "sigma_clip_mask! (BitArray logic)" begin
        data = randn(100)
        data[1] = 100.0 # Outlier 
        data[2] = -100.0 # Outlier 

        mask = falses(length(data))

        sigma_clip_mask!(data, mask, sigma_lower=3, sigma_upper=3, maxiter=5)

        @test mask[1] == true
        @test mask[2] == true
        @test count(mask) >= 2 


        # inv_mask = trues(length(data)) # Iniziamo con tutto buono
        # sigma_clip_mask!(data, inv_mask, sigma_lower=3, sigma_upper=3, bad=false)

        # @test inv_mask[1] == false 
        # @test inv_mask[2] == false
    end

    @testset "Algorithmic Behavior & Params" begin

        data = zeros(50)
        data[1] = 100.0  # Outlier 
        data[2] = -100.0 # Outlier 


        res_high = sigma_clip(data, sigma_upper=2, sigma_lower=1000)
        @test isnan(res_high[1])
        @test res_high[2] == -100.0 # Non deve essere tagliato

        # Taglia solo sotto
        res_low = sigma_clip(data, sigma_upper=1000, sigma_lower=2)
        @test res_low[1] == 100.0
        @test isnan(res_low[2])

        # Test maxiter
        # slow_data = Float64[1, 1, 1, 1, 10, 20, 100]

        # # Con 1 iterazione, potrebbe non beccare il 10 o 20 perché il 100 gonfia la STD
        # mask_iter1 = falses(length(slow_data))
        # sigma_clip_mask!(slow_data, mask_iter1, maxiter=1, sigma_upper=2)

        # # Con 5 iterazioni, dovrebbe convergere e pulire meglio
        # mask_iter5 = falses(length(slow_data))
        # sigma_clip_mask!(slow_data, mask_iter5, maxiter=5, sigma_upper=2)

        # # Ci aspettiamo che iter5 sia più aggressivo o uguale a iter1
        # @test sum(mask_iter5) >= sum(mask_iter1)
    end

    @testset "Edge Cases: NaN and Inf" begin
        # Gestione NaN in input
        data = rand(100)
        data[3] = NaN 
        data[4] = 100.0

        res = sigma_clip(data, sigma_upper=2)


        @test isnan(res[3]) #
        @test isnan(res[4]) # Clip

        # Inf in input
        data[5] = Inf
        # Inf dovrebbe essere sempre escluso dai calcoli bounds e marcato come outlier
        res_inf = sigma_clip(data)
        @test isnan(res_inf[5])

    end

    @testset "Custom Reducers" begin
        data = [1.0, 2.0, 3.0, 100.0]

        # Usiamo la media invece della mediana (meno robusto, l'outlier alza la media)
        # Media ~ 26. 100 è < 26 + 3*std?
        # Dipende dalla std.
        # Proviamo a forzare un clip usando una funzione custom

        # Custom center reducer che ritorna sempre 0
        res = sigma_clip(data, cent_reducer=x -> 0.0, std_reducer=x -> 1.0, sigma_upper=50)
        # Bounds: [-50, 50]. 100 viene tagliato.
        @test isnan(res[4])
        @test res[1] == 1.0
    end

    # @testset "Wrapper sigma_clip_mask logic" begin

    #     data = randn(10)
    #     mask = sigma_clip_mask(data)

    #     @test all(mask)
    # end
end