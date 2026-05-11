using VIDA
using Krang
import CairoMakie as CM
using LaTeXStrings
using Comrade
using VLBIImagePriors
using Enzyme
using BasicInterpolators
using Images
using Accessors
using ImageIO
using FileIO
using Optimization
using OptimizationOptimisers
using Enzyme
using StableRNGs
using StatsBase
using Plots: Plots
using LaTeXStrings
using Distributions, DistributionsAD
using Pyehtim
using Accessors
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Optim
using Accessors
using LaTeXStrings
using DataFrames

include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF.jl"))

function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa))#, Shift(θ2.x, θ2.y))
	return RenormalizedFlux(m, θ2.f)
end


lr = 0.05
seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.01
bulkx = 1.0
bulky = 1.0
bulkpix = 240#20
raster_size = 120.0# 90 # in microarcseconds    
snrcut = 3.0
uv_min = 0.1e9

year = 2017#"bhex"

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 => "L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(dirname(@__DIR__)), "data", data[year])
files = "/Users/dominicchang/Desktop/CenterComparison/data/snapshots"

function normalize!(img)
	img ./= flux(img)
	return img
end

best_fits = filter(x->occursin("ma+", x) && occursin("best_fits", x) && occursin("1_nall", x), readdir(joinpath((@__DIR__))))[1:100]
jukebox_fits = filter(x->occursin("jukebox", x), best_fits)
phibi_fits = filter(x->!occursin("jukebox", x), best_fits)

dfjk = DataFrame(
	m_d = Float64[],
	spin = Float64[],
	θo = Float64[],
	θs = Float64[],
	rpeak = Float64[],
	p1 = Float64[],
	p2 = Float64[],
	χ = Float64[],
	ι = Float64[],
	βv = Float64[],
	spec = Float64[],
	η = Float64[],
	frac = Float64[],
	pa = Float64[],
	f = Float64[],
)
dfjk = let dfjk = dfjk
	for fname in jukebox_fits[begin:length(phibi_fits)]
		fpath = open(joinpath((@__DIR__), fname), "r")
		best_fit = eval(Meta.parse(readline(fpath)))
		close(fpath)
		dfjk = vcat(
			dfjk,
			DataFrame(;
				m_d = rad2μas(best_fit.sky.m_d),
				spin = best_fit.sky.spin,
				θo = best_fit.sky.θo,
				θs = best_fit.sky.θs,
				rpeak = best_fit.sky.rpeak,
				p1 = best_fit.sky.p1,
				p2 = best_fit.sky.p2,
				χ = best_fit.sky.χ,
				ι = best_fit.sky.ι,
				βv = best_fit.sky.βv,
				spec = best_fit.sky.spec,
				η = best_fit.sky.η,
				frac = best_fit.sky.frac,
				pa = best_fit.sky.pa,
				f = best_fit.sky.f,
			),
		)
	end
	dfjk
end

df = DataFrame(
	m_d = Float64[],
	spin = Float64[],
	θo = Float64[],
	θs = Float64[],
	rpeak = Float64[],
	p1 = Float64[],
	p2 = Float64[],
	χ = Float64[],
	ι = Float64[],
	βv = Float64[],
	spec = Float64[],
	η = Float64[],
	frac = Float64[],
	pa = Float64[],
	f = Float64[],
	σimg = Float64[],
)
df = let df = df
	for fname in phibi_fits[begin:end]
		fpath = open(joinpath((@__DIR__), fname), "r")
		best_fit = eval(Meta.parse(readline(fpath)))
		close(fpath)
		df = vcat(
			df,
			DataFrame(;
				m_d = rad2μas(best_fit.sky.m_d),
				spin = best_fit.sky.spin,
				θo = best_fit.sky.θo,
				θs = best_fit.sky.θs,
				rpeak = best_fit.sky.rpeak,
				p1 = best_fit.sky.p1,
				p2 = best_fit.sky.p2,
				χ = best_fit.sky.χ,
				ι = best_fit.sky.ι,
				βv = best_fit.sky.βv,
				spec = best_fit.sky.spec,
				η = best_fit.sky.η,
				frac = best_fit.sky.frac,
				pa = best_fit.sky.pa,
				f = best_fit.sky.f,
				σimg = best_fit.sky.σimg,
			),
		)
	end
	df
end

best_fits = filter(x->occursin("sa+", x) && occursin("best_fits", x), readdir(joinpath((@__DIR__))))[3:102]
jukebox_fits = filter(x->occursin("jukebox", x), best_fits)
phibi_fits = filter(x->!occursin("jukebox", x), best_fits)

dfjk_s = DataFrame(
	m_d = Float64[],
	spin = Float64[],
	θo = Float64[],
	θs = Float64[],
	rpeak = Float64[],
	p1 = Float64[],
	p2 = Float64[],
	χ = Float64[],
	ι = Float64[],
	βv = Float64[],
	spec = Float64[],
	η = Float64[],
	frac = Float64[],
	pa = Float64[],
	f = Float64[],
)
dfjk_s = let dfjk = dfjk_s
	for fname in jukebox_fits[begin:length(phibi_fits)]
		fpath = open(joinpath((@__DIR__), fname), "r")
		best_fit = eval(Meta.parse(readline(fpath)))
		close(fpath)
		dfjk = vcat(
			dfjk,
			DataFrame(;
				m_d = rad2μas(best_fit.sky.m_d),
				spin = best_fit.sky.spin,
				θo = best_fit.sky.θo,
				θs = best_fit.sky.θs,
				rpeak = best_fit.sky.rpeak,
				p1 = best_fit.sky.p1,
				p2 = best_fit.sky.p2,
				χ = best_fit.sky.χ,
				ι = best_fit.sky.ι,
				βv = best_fit.sky.βv,
				spec = best_fit.sky.spec,
				η = best_fit.sky.η,
				frac = best_fit.sky.frac,
				pa = best_fit.sky.pa,
				f = best_fit.sky.f,
			),
		)
	end
	dfjk
end


df_s = DataFrame(
	m_d = Float64[],
	spin = Float64[],
	θo = Float64[],
	θs = Float64[],
	rpeak = Float64[],
	p1 = Float64[],
	p2 = Float64[],
	χ = Float64[],
	ι = Float64[],
	βv = Float64[],
	spec = Float64[],
	η = Float64[],
	frac = Float64[],
	pa = Float64[],
	f = Float64[],
	σimg = Float64[],
)
df_s = let df = df_s
	for fname in phibi_fits[begin:end]
		fpath = open(joinpath((@__DIR__), fname), "r")
		best_fit = eval(Meta.parse(readline(fpath)))
		close(fpath)
		df = vcat(
			df,
			DataFrame(;
				m_d = rad2μas(best_fit.sky.m_d),
				spin = best_fit.sky.spin,
				θo = best_fit.sky.θo,
				θs = best_fit.sky.θs,
				rpeak = best_fit.sky.rpeak,
				p1 = best_fit.sky.p1,
				p2 = best_fit.sky.p2,
				χ = best_fit.sky.χ,
				ι = best_fit.sky.ι,
				βv = best_fit.sky.βv,
				spec = best_fit.sky.spec,
				η = best_fit.sky.η,
				frac = best_fit.sky.frac,
				pa = best_fit.sky.pa,
				f = best_fit.sky.f,
				σimg = best_fit.sky.σimg,
			),
		)
	end
	df
end

clrs = CM.Makie.wong_colors()
curr_theme = CM.Theme(
	Axis = (
		ygridvisible = false,
		xgridvisible = false,
		xticklabelcolor = :black,
		yticklabelcolor = :black,
		titlecolor = :black,
		xlabelcolor = :black,
		ylabelcolor = :black,
		xticksmirrored = true,
		yticksmirrored = true,
		xtickalign = 1,
		ytickalign = 1,
		aspect = 1.8,
		width = 340.0,
		titlesize = 40.0,
		ylabelsize = 35.0,
		xticklabelsize = 25.0,
		yticklabelsize = 25.0,),
	Scatter = (markersize = 10.0, alpha = 0.90, rasterize = true),
	Text = (fontsize = 40.0, color = :black),
	Legend = (framevisibile = false, margin = (0.0, 0.0, 0.0, 0.0), labelsize = 35, titlesize = 40.0),
)
CM.set_theme!(CM.merge(CM.theme_latexfonts(), curr_theme))
diagonal_square_marker(color1, color2) = Any[
	CM.PolyElement(
		color = color1,
		strokecolor = :transparent,
		points = CM.Point2f[(0, 0), (1, 0), (1, 1)],
	),
	CM.PolyElement(
		color = color2,
		strokecolor = :transparent,
		points = CM.Point2f[(0, 0), (0, 1), (1, 1)],
	),
]

fig = begin
	fig = CM.Figure(size = (900, 1500))

	xvals = LinRange(1000, 1000 + 5*50, size(dfjk.m_d)[1])
	xvalsband = [1000-5, 1000 + 5*50 + 5]

	ax = CM.Axis(fig[1, 1], ylabel = latexstring("m_d\\ (\\mu as)"), title = latexstring("\\text{MAD R}_{\\text{high}}=1 "))
	sc = CM.scatter!(xvals, dfjk.m_d)
	m, s = mean_and_std(dfjk.m_d)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df.m_d)
	m, s = mean_and_std(df.m_d)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.hlines!([3.83], color = :black, linestyle = :dash, linewidth = 3.0)
	CM.xlims!(xvalsband...)
	add_kde_estimate_text!(ax, dfjk.m_d, df.m_d; x = 0.03, y = 0.35)
	#add_kde_estimate_text!(ax, dfjk.m_d, df.m_d; x =0.03, y=0.3)
	#CM.ylims!(3.5, 5.5)

	ax = CM.Axis(fig[2, 1], ylabel = latexstring("a"))
	sc = CM.scatter!(xvals, dfjk.spin)
	m, s = mean_and_std(dfjk.spin)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df.spin)
	m, s = mean_and_std(df.spin)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.hlines!([0.5], color = :black, linestyle = :dash, linewidth = 3.0)
	CM.xlims!(xvalsband...)
	CM.ylims!(0.0, 1.0)
	add_kde_estimate_text!(ax, dfjk.spin, df.spin; x = 0.03, y = 1.0)

	ax = CM.Axis(fig[3, 1], ylabel = latexstring("\\theta_o\\ ({}^\\circ)"))
	sc = CM.scatter!(xvals, dfjk.θo)
	m, s = mean_and_std(dfjk.θo)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df.θo)
	m, s = mean_and_std(df.θo)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.hlines!([163.0], color = :black, linestyle = :dash, linewidth = 3.0)
	CM.xlims!(xvalsband...)
	CM.ylims!(100.0, 180.0)
	add_kde_estimate_text!(ax, dfjk.θo, df.θo; x = 0.03, y = 0.35)

	#ax = CM.Axis(fig[4,1], xlabel=latexstring("\\text{Time }(GM/c^3)"), ylabel=latexstring("\\theta_s \\ ({}^\\circ)"))
	ax = CM.Axis(fig[4, 1], ylabel = latexstring("\\theta_s \\ ({}^\\circ)"))
	sc = CM.scatter!(xvals, dfjk.θs)
	m, s = mean_and_std(dfjk.θs)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df.θs)
	m, s = mean_and_std(df.θs)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.xlims!(xvalsband...)
	CM.ylims!(35.0, 95.0)
	add_kde_estimate_text!(ax, dfjk.θs, df.θs; x = 0.03, y = 0.35)

	ax = CM.Axis(fig[5, 1], ylabel = latexstring("\\chi \\ ({}^\\circ)"))
	sc = CM.scatter!(xvals, dfjk.χ .* 180/π)
	m, s = mean_and_std(dfjk.χ .* 180/π)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df.χ .* 180/π)
	m, s = mean_and_std(df.χ .* 180/π)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.xlims!(xvalsband...)
	CM.ylims!(-180.0, 180.0)
	add_kde_estimate_text!(ax, dfjk.χ .* 180/π, df.χ .* 180/π; x = 0.03, y = 0.35)

	xvals = LinRange(1002, 1002+3*50, size(dfjk_s.m_d)[1])
	xvalsband = [1002-5, 1002+3*50+5]

	ax = CM.Axis(fig[1, 2], title = latexstring("\\text{SANE R}_{\\text{high}}=160 "))
	sc = CM.scatter!(xvals, dfjk_s.m_d)
	m, s = mean_and_std(dfjk_s.m_d)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df_s.m_d)
	m, s = mean_and_std(df_s.m_d)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.hlines!([3.83], color = :black, linestyle = :dash, linewidth = 3.0)
	CM.xlims!(xvalsband...)
	add_kde_estimate_text!(ax, dfjk_s.m_d, df_s.m_d)
	CM.ylims!(3.4, 5.8)

	ax = CM.Axis(fig[2, 2])#, ylabel=latexstring("\\text{Spin}"))
	sc = CM.scatter!(xvals, dfjk_s.spin)
	m, s = mean_and_std(dfjk_s.spin)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df_s.spin)
	m, s = mean_and_std(df_s.spin)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.hlines!([0.94], color = :black, linestyle = :dash, linewidth = 3.0)
	CM.xlims!(xvalsband...)
	CM.ylims!(0.0, 1.0)
	add_kde_estimate_text!(ax, dfjk_s.spin, df_s.spin; x = 0.03, y = 0.45)

	ax = CM.Axis(fig[3, 2])#, ylabel=latexstring("\\text{Observer Inclination } ({}^\\circ)"))
	sc = CM.scatter!(xvals, dfjk_s.θo)
	m, s = mean_and_std(dfjk_s.θo)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df_s.θo)
	m, s = mean_and_std(df_s.θo)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.hlines!([163.0], color = :black, linestyle = :dash, linewidth = 3.0)
	CM.xlims!(xvalsband...)
	CM.ylims!(100.0, 180.0)
	add_kde_estimate_text!(ax, dfjk_s.θo, df_s.θo; x = 0.03, y = 0.35)

	ax = CM.Axis(fig[4, 2])#, xlabel=latexstring("\\text{Time }(GM/c^3)"), ylabel=latexstring("\\chi \\ ({}^\\circ)"))
	sc = CM.scatter!(xvals, dfjk_s.θs)
	m, s = mean_and_std(dfjk_s.θs)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df_s.θs)
	m, s = mean_and_std(df_s.θs)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.xlims!(xvalsband...)
	CM.ylims!(35.0, 95.0)
	add_kde_estimate_text!(ax, dfjk_s.θs, df_s.θs)

	ax = CM.Axis(fig[5, 2])#, xlabel=latexstring("\\text{Time }(GM/c^3)"), ylabel=latexstring("\\chi \\ ({}^\\circ)"))
	sc = CM.scatter!(xvals, dfjk_s.χ .* 180/π)
	m, s = mean_and_std(dfjk_s.χ .* 180/π)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.scatter!(xvals, df_s.χ .* 180/π)
	m, s = mean_and_std(df_s.χ .* 180/π)
	b = CM.band!(xvalsband, Float64[m-s for i in xvalsband], Float64[m+s for i in xvalsband]; alpha = 0.2)#, color=clrs[1], alpha=0.5)
	CM.translate!(b, (0.0, 0.0, -5.0))
	CM.xlims!(xvalsband...)
	CM.ylims!(-180.0, 180.0)
	add_kde_estimate_text!(ax, dfjk_s.χ .* 180/π, df_s.χ .* 180/π; x = 0.03, y = 0.35)

	temp_clrs1 = clrs[1]
	@reset temp_clrs1.alpha = 0.3
	temp_clrs2 = clrs[2]
	@reset temp_clrs2.alpha = 0.3
	model_markers = [
		push!(Any[CM.MarkerElement(marker = :circle, color = color, strokecolor = :transparent) for color in clrs[1:2]], CM.LineElement(color = :black, linestyle = :dash, linewidth=3.0)),
		[diagonal_square_marker(temp_clrs1, temp_clrs2)],#CM.PolyElement(color=RGBA(0,0,0,0.2))]
	]
	model_labels = [[L"\text{Dual-Cone}", L"\text{PHIBI}", "Truth"], ["Uncertainty from intrinsic-variability"]]
	legend = CM.Legend(fig[6, 1:2], model_markers, model_labels, ["", ""], valign = :center, halign = :center, framevisible = false)
	legend.nbanks = 4
	CM.rowgap!(fig.layout, 1, 10.0)
	CM.rowgap!(fig.layout, 2, 10.0)
	CM.rowgap!(fig.layout, 3, 10.0)
	CM.rowgap!(fig.layout, 4, 10.0)
	CM.rowgap!(fig.layout, 5, 10.0)
	CM.colgap!(fig.layout, 1, 4.0)
	#display(fig)

	save(joinpath((@__DIR__), "phibi_jukebox_comparison.pdf"), fig)
	fig
end
