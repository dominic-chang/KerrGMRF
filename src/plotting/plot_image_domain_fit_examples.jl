using VIDA
using Krang
import CairoMakie as CM
using LaTeXStrings
using Comrade
using VLBIImagePriors
using Enzyme
using BasicInterpolators
using Accessors

include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF.jl"))

const SNAPSHOT_DIR = "/Users/dominicchang/Desktop/CenterComparison/data/snapshots"
const TAVG_DIR = "/Users/dominicchang/Desktop/CenterComparison/data/tavgs_fits_480"
const IMAGE_FOV = μas2rad(120.0)
const IMAGE_PIXELS = 240
const ROTATION_ANGLE = 108.0 / 180 * π
const PA_ROTATION_ANGLE = π * (180.0 - 72.0) / 180.0

curr_theme = CM.Theme(
	Image = (rasterize = true,),
	Text = (fontsize = 30,),
)
CM.set_theme!(merge(CM.theme_latexfonts(), curr_theme))

bulkgrid = imagepixels(1.0, 1.0, 240, 240; executor = Serial())
transform1, _ = matern(size(bulkgrid))
transform2, _ = matern(size(bulkgrid))
metadata = (; bulkgrid, transform1, transform2, raster_size = 120.0, offset = 0.0)

function ModifiedKerrGMRF(θ, metadata)
	model = Comrade.modify(KerrGMRF(θ, metadata), Stretch(θ.m_d, θ.m_d), Rotate(θ.pa))
	return RenormalizedFlux(model, θ.f)
end

function normalize_flux!(img)
	img ./= flux(img)
	return img
end

function load_snapshot(fname; normalize = true)
	img = VIDA.rotated(VIDA.load_image(joinpath(SNAPSHOT_DIR, fname)), ROTATION_ANGLE)
	img = regrid(img, imagepixels(IMAGE_FOV, IMAGE_FOV, IMAGE_PIXELS, IMAGE_PIXELS))
	img .= max.(img, 0.0)
	normalize && normalize_flux!(img)
	return img
end

function load_best_fit(fname)
	return open(joinpath(@__DIR__, fname), "r") do io
		eval(Meta.parse(readline(io)))
	end
end

function model_images(best_fit, grid; normalize = true)
	mean_best_fit = deepcopy(best_fit)
	@reset mean_best_fit.sky.σimg = 1e-5

	model_intmap = intensitymap(ModifiedKerrGMRF(best_fit.sky, metadata), grid)
	model_mean_component_intmap = intensitymap(ModifiedKerrGMRF(mean_best_fit.sky, metadata), grid)

	if normalize
		normalize_flux!(model_intmap)
		normalize_flux!(model_mean_component_intmap)
	end

	return model_intmap, model_mean_component_intmap
end

function time_averaged_grmhd(fname, grid; normalize = true)
	img = Comrade.rotated(regrid(Comrade.load_fits(joinpath(TAVG_DIR, fname), IntensityMap), grid), PA_ROTATION_ANGLE)
	normalize && normalize_flux!(img)
	return img
end

function add_fit_text!(ax, best_fit; x = 55.0)
	CM.text!(ax, μas2rad(x), μas2rad(46.5); text = latexstring("a_*= $(round(best_fit.sky.spin, digits=2))"), color = :white, fontsize = 25.0)
	CM.text!(ax, μas2rad(x), μas2rad(34.5); text = latexstring("\\theta_o= $(Int(round(best_fit.sky.θo, digits=0)))^\\circ"), color = :white, fontsize = 25.0)
	return CM.text!(ax, μas2rad(x), μas2rad(23.5); text = latexstring("m_d= $(round(rad2μas(best_fit.sky.m_d), digits=2))\\ \\mu as"), color = :white, fontsize = 25.0)
end

function add_truth_text!(ax; spin, rhigh, x = 57.0)
	CM.text!(ax, μas2rad(x), μas2rad(46.5); text = latexstring("a_*= $(spin)"), color = :white, fontsize = 25.0)
	CM.text!(ax, μas2rad(x), μas2rad(34.5); text = L"\theta_o=163^\circ", color = :white, fontsize = 25.0)
	CM.text!(ax, μas2rad(x), μas2rad(23.5); text = latexstring("m_d= 3.83\\ \\mu as"), color = :white, fontsize = 25.0)
	return CM.text!(ax, μas2rad(-5.0), μas2rad(46.5); text = latexstring("R_{\\text{high}}=$(rhigh)"), color = :white, fontsize = 25.0)
end

function plot_image_domain_fit_example(;
	snapshot_fname,
	best_fit_fname,
	tavg_fname,
	output_fname,
	system_label,
	truth_spin,
	truth_rhigh,
	fit_text_x = 55.0,
	truth_text_x = 57.0,
	normalize_images = true,
)
	inimg = load_snapshot(snapshot_fname; normalize = normalize_images)
	grid = imagepixels(fieldofview(inimg).X, fieldofview(inimg).Y, length(inimg.X), length(inimg.Y); executor = ThreadsEx())

	best_fit = load_best_fit(best_fit_fname)
	model_intmap, model_mean_component_intmap = model_images(best_fit, grid; normalize = normalize_images)
	ave_grmhd_intmap = time_averaged_grmhd(tavg_fname, grid; normalize = normalize_images)
	nx = NxCorr(max.(0.0, inimg))

	fig = CM.Figure(size = (600, 600))

	ax = CM.Axis(fig[1, 1], aspect = 1)
	CM.hidedecorations!(ax)
	_imgviz!(fig, ax, model_mean_component_intmap; show_colorbar = false)
	CM.text!(ax, μas2rad(55.0), μas2rad(30.5); text = "PHIBI\nMean Component", color = :white)

	ax = CM.Axis(fig[1, 2], aspect = 1)
	CM.hidedecorations!(ax)
	_imgviz!(fig, ax, ave_grmhd_intmap; show_colorbar = false)
	CM.text!(ax, μas2rad(55.0), μas2rad(31.0); text = "Time Averaged\nGRMHD", color = :white)

	ax = CM.Axis(fig[2, 1], aspect = 1)
	CM.hidedecorations!(ax)
	_imgviz!(fig, ax, model_intmap; show_colorbar = false)
	CM.text!(ax, μas2rad(5.0), μas2rad(-45.5); text = "PHIBI", color = :white)
	CM.text!(ax, μas2rad(5.0), μas2rad(-55.5); text = "NxCORR: $(round(1 - divergence(nx, model_intmap), digits = 3))", color = :white, fontsize = 20.0)
	add_fit_text!(ax, best_fit; x = fit_text_x)

	ax = CM.Axis(fig[2, 2], aspect = 1)
	CM.hidedecorations!(ax)
	_imgviz!(fig, ax, inimg; show_colorbar = false)
	CM.text!(ax, μas2rad(-15.0), μas2rad(-45.5); text = system_label, color = :white)
	CM.text!(ax, μas2rad(-15.0), μas2rad(-55.5); text = "Snapshot", color = :white, fontsize = 20.0)
	add_truth_text!(ax; spin = truth_spin, rhigh = truth_rhigh, x = truth_text_x)

	CM.colgap!(fig.layout, 1, 5.0)
	CM.rowgap!(fig.layout, 1, 5.0)
	CM.save(joinpath(@__DIR__, output_fname), fig)
	return fig
end

plot_image_domain_fit_example(
	snapshot_fname = "image_ma+0.5_1275_163_1_nall.h5",
	best_fit_fname = "image_ma+0.5_1275_163_1_nall.h5_best_fits.txt",
	tavg_fname = "ma+0.5_r1_nall_tavg.fits",
	output_fname = "mad_mean_model_comp.png",
	system_label = "MAD",
	truth_spin = "0.5",
	truth_rhigh = "1",
	fit_text_x = 55.0,
	truth_text_x = 55.0,
	normalize_images = true,
) |> display

plot_image_domain_fit_example(
	snapshot_fname = "image_sa+0.94_981_163_160_nall.h5",
	best_fit_fname = "image_sa+0.94_981_163_160_nall.h5_best_fits.txt",
	tavg_fname = "sa+0.94_r160_nall_tavg.fits",
	output_fname = "sane_mean_model_comp.png",
	system_label = "SANE",
	truth_spin = "0.94",
	truth_rhigh = "160",
	fit_text_x = 57.0,
	truth_text_x = 57.0,
	normalize_images = false,
) |> display
