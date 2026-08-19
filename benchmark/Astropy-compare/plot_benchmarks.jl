#!/usr/bin/env julia

using CSV
using DataFrames
using CairoMakie

set_theme!(theme_minimal())

dir = @__DIR__
output = joinpath(dir, "benchmark_plot.png")

julia = CSV.read(joinpath(dir, "julia_results.csv"), DataFrame)
astropy = CSV.read(joinpath(dir, "astropy_results.csv"), DataFrame)
results = vcat(julia, astropy)
results.time_ms = results.median_ns ./ 1.0e6

vector = results[results.family .== "vector", :]
matrix = results[results.family .== "matrix", :]

vector_sigma_out = sort(vector[(vector.algorithm .== "SigmaClip.jl") .& (vector.mode .== "out_of_place"), :], :elements)
vector_sigma_in = sort(vector[(vector.algorithm .== "SigmaClip.jl") .& (vector.mode .== "in_place"), :], :elements)
vector_astropy = sort(vector[vector.algorithm .== "astropy.stats.sigma_clip", :], :elements)

matrix_sigma_out = sort(matrix[(matrix.algorithm .== "SigmaClip.jl") .& (matrix.mode .== "out_of_place"), :], :elements)
matrix_sigma_in = sort(matrix[(matrix.algorithm .== "SigmaClip.jl") .& (matrix.mode .== "in_place"), :], :elements)
matrix_astropy = sort(matrix[matrix.algorithm .== "astropy.stats.sigma_clip", :], :elements)

vector_ticks = sort(unique(vector.elements))
vector_labels = string.(vector_ticks)
matrix_tick_rows = unique(matrix[:, [:elements, :shape]])
sort!(matrix_tick_rows, :elements)
matrix_ticks = matrix_tick_rows.elements
matrix_labels = replace.(string.(matrix_tick_rows.shape), "x" => "×")

blue = Makie.to_color("#2563EB")
orange = Makie.to_color("#F59E0B")
red = Makie.to_color("#E11D48")

figure = Figure(size = (1000, 600), backgroundcolor = :white)
Label(figure[0, 1:2], "Sigma clipping performance", fontsize = 28, font = :bold, padding = (0, 0, 10, 0))

vector_axis = Axis(
    figure[1, 1],
    title = "Vectors",
    xlabel = "Number of elements",
    ylabel = "Median time (ms)",
    xscale = log10,
    yscale = log10,
    xticks = (vector_ticks, vector_labels),
    xticklabelrotation = pi / 6,
    xminorticksvisible = true,
    yminorticksvisible = true,
    xgridvisible = true,
    ygridvisible = true,
    xminorgridvisible = true,
    yminorgridvisible = true,
    xgridstyle = :dash,
    ygridstyle = :dash,
    xminorgridstyle = :dot,
    yminorgridstyle = :dot,
    xgridcolor = (:gray, 0.18),
    ygridcolor = (:gray, 0.18),
    xminorgridcolor = (:gray, 0.08),
    yminorgridcolor = (:gray, 0.08),
    spinewidth = 1.5,
    xticksize = 14,
    yticksize = 14,
    xticklabelsize = 18,
    yticklabelsize = 18,
    titlesize = 20,
    xlabelsize = 20,
    ylabelsize = 20,
)

lines!(vector_axis, vector_sigma_out.elements, vector_sigma_out.time_ms, color = blue, linewidth = 4, label = "SigmaClip.jl · out-of-place")
scatter!(vector_axis, vector_sigma_out.elements, vector_sigma_out.time_ms, color = blue, markersize = 14, marker = :circle)
lines!(vector_axis, vector_sigma_in.elements, vector_sigma_in.time_ms, color = orange, linewidth = 4, label = "SigmaClip.jl · in-place")
scatter!(vector_axis, vector_sigma_in.elements, vector_sigma_in.time_ms, color = orange, markersize = 14, marker = :diamond)
lines!(vector_axis, vector_astropy.elements, vector_astropy.time_ms, color = red, linewidth = 4, label = "Astropy · out-of-place")
scatter!(vector_axis, vector_astropy.elements, vector_astropy.time_ms, color = red, markersize = 14, marker = :utriangle)

matrix_axis = Axis(
    figure[1, 2],
    title = "Matrices",
    xlabel = "Shape",
    xscale = log10,
    yscale = log10,
    xticks = (matrix_ticks, matrix_labels),
    xticklabelrotation = pi / 6,
    xminorticksvisible = true,
    yminorticksvisible = true,
    xgridvisible = true,
    ygridvisible = true,
    xminorgridvisible = true,
    yminorgridvisible = true,
    xgridstyle = :dash,
    ygridstyle = :dash,
    xminorgridstyle = :dot,
    yminorgridstyle = :dot,
    xgridcolor = (:gray, 0.18),
    ygridcolor = (:gray, 0.18),
    xminorgridcolor = (:gray, 0.08),
    yminorgridcolor = (:gray, 0.08),
    spinewidth = 1.5,
    xticksize = 14,
    yticksize = 14,
    xticklabelsize = 18,
    yticklabelsize = 18,
    titlesize = 20,
    xlabelsize = 20,
    ylabelsize = 20,
)

lines!(matrix_axis, matrix_sigma_out.elements, matrix_sigma_out.time_ms, color = blue, linewidth = 4)
scatter!(matrix_axis, matrix_sigma_out.elements, matrix_sigma_out.time_ms, color = blue, markersize = 14, marker = :circle)
lines!(matrix_axis, matrix_sigma_in.elements, matrix_sigma_in.time_ms, color = orange, linewidth = 4)
scatter!(matrix_axis, matrix_sigma_in.elements, matrix_sigma_in.time_ms, color = orange, markersize = 14, marker = :diamond)
lines!(matrix_axis, matrix_astropy.elements, matrix_astropy.time_ms, color = red, linewidth = 4)
scatter!(matrix_axis, matrix_astropy.elements, matrix_astropy.time_ms, color = red, markersize = 14, marker = :utriangle)

linkyaxes!(vector_axis, matrix_axis)
Legend(figure[2, 1:2], vector_axis, orientation = :horizontal, framevisible = false, labelsize = 14)


display(figure)
save(output, figure, px_per_unit = 2)
