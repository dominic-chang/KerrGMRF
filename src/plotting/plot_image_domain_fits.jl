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
using Plots: Plots
using LaTeXStrings
using Distributions, DistributionsAD
using Pyehtim
using Accessors
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Optim

curr_theme = CM.Theme(
	Image=(rasterize=true,),
	Text=(fontsize=30,)
)
CM.set_theme!(merge(CM.theme_latexfonts(), curr_theme))

include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF.jl"))

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
files= "/Users/dominicchang/Desktop/CenterComparison/data/snapshots"
fname = "image_ma+0.5_1275_163_1_nall.h5"
img_path = joinpath(files, fname)

inimg = rotated(VIDA.load_image(img_path),108.0/180*π)
npix = inimg.X |> length 
fovx, fovy = fieldofview(inimg)
inimg = regrid(inimg, imagepixels(μas2rad(120.0), μas2rad(120.0), 240,240))
npix = inimg.X |> length 
fovx, fovy = fieldofview(inimg)
inimg .= max.(inimg, 0.0)

#joinpath(dirname(@__DIR__), "smiley.png")
#smiley = map(
#    x->Float64(Gray(x).val),
#    FileIO.load(joinpath(dirname(@__DIR__), "smiley.png"))[Int.(floor.(1:2.5:1200)), Int.(floor.(1:2.5:1200))]
#)
#inimg .= smiley
imageviz(inimg, colorscale = log10, colorrange = (1e-6, 1e-3), colormap = :inferno)
inimg .= max.(inimg, 0.0)
inimg ./= flux(inimg)
bh = Bhattacharyya(inimg)
nx = NxCorr(inimg)

bulkgrid            = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor = Serial())
transform1, cprior1 = matern(size(bulkgrid))
transform2, cprior2 = matern(size(bulkgrid))
prior               = (
m_d = Uniform(μas2rad(1.0), μas2rad(8.0)),
spin = Uniform(0.01, 0.99),#-0.99, -0.01),
θo = Uniform(120, 179.0),
θs = Uniform(40, 90.0),#,VLBIImagePriors.DeltaDist(90.0),
rpeak = Uniform(1.0, 8.0),# Uniform(1.0, 8.0),
p1 = Uniform(0.1, 5.0),#Uniform(0.1, 5.0),
p2 = Uniform(1.0, 5.0),
χ =VLBIImagePriors.DiagonalVonMises(0, inv(π^2)), 
ι = Uniform(0, π/2),
βv = Uniform(0.01, 0.9),
spec = Uniform(-1.0, 5.0),
η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
frac = VLBIImagePriors.DeltaDist(1.0),#Uniform(0.0, 1.0),
pa = VLBIImagePriors.DeltaDist(π*(180.0-72.0)),# Uniform(-π, 0),
f = VLBIImagePriors.DeltaDist(1.0),#Uniform(-π, π),
σimg = truncated(Normal(0.0, 1.0); lower = 0.0),
ρpr = truncated(InverseGamma(1.0, -log(0.1)*10); lower = 1.0, upper = 2*max(size(bulkgrid)...)),
νpr = Uniform(1.0, 5.0),
c1 = cprior1,
c2 = cprior2,
x = Uniform(μas2rad.((-1,1))...),
y = Uniform(μas2rad.((-1,1))...),
)
offset = 0.0
function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa), Shift(θ2.x, θ2.y))
	return RenormalizedFlux(m, θ2.f)
end
obsin = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min = 0.1e9).add_fractional_noise(fractional_noise)
ehtimg = ehtim.image.load_image(img_path).rotate(π-72*π/180)
ehtimg.ivec.shape
ehtimg.rf = obsin.rf
ehtimg.ra = obsin.ra
ehtimg.dec = obsin.dec
ehtimg.ivec
ehtimg.mjd = 57848
ehtimg.display()
obs = ehtimg.observe_same(obsin, ampcal = ampcal, phasecal = phasecal, add_th_noise = add_th_noise, seed = seed, ttype = "fast")
obs = scan_average(obs.flag_uvdist(uv_min = 0.1e9))
obs = obs.add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))
skym = SkyModel(
	ModifiedKerrGMRF,
	prior,
	imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx());
	metadata = (; bulkgrid, transform1, transform2, raster_size, offset),
)
post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode = set_runtime_activity(Enzyme.Reverse))
fpost = Comrade.asflat(post)
using Accessors
using LaTeXStrings

function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa))#, Shift(θ2.x, θ2.y))
	return RenormalizedFlux(m, θ2.f)
end

topgrid = imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx())
grid = imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx())


fpath = open("/Users/dominicchang/Desktop/KerrGMRF/src/plotting/image_ma+0.5_1275_163_1_nall.h5_best_fits.txt", "r")
best_fit = eval(Meta.parse(readline(fpath)))
mean_best_fit = deepcopy(best_fit)
@reset mean_best_fit.sky.σimg = 1e-5
close(fpath)
model_intmap = intensitymap(ModifiedKerrGMRF(best_fit.sky, skym.metadata), topgrid)
model_mean_component_intmap = intensitymap(ModifiedKerrGMRF(mean_best_fit.sky, skym.metadata), grid)
ave_grmhd_intmap = Comrade.rotated(regrid(Comrade.load_fits("/Users/dominicchang/Desktop/CenterComparison/data/tavgs_fits_480/ma+0.5_r1_nall_tavg.fits", IntensityMap), grid), π*(180.0-72.0)/180.0)
imageviz(ave_grmhd_intmap)
nx = NxCorr(inimg)
1-divergence(nx, model_intmap)
begin
	fig = CM.Figure(size=(600,600))
	ax = CM.Axis(fig[1,1], aspect=1);CM.hidedecorations!(ax)
	_imgviz!(fig, ax, model_mean_component_intmap; show_colorbar=false)
	CM.text!(ax, μas2rad(55.), μas2rad(30.5);text="PHIBI\nMean Component", color=:white)
	ax = CM.Axis(fig[1,2], aspect=1);CM.hidedecorations!(ax)
	_imgviz!(fig, ax, ave_grmhd_intmap; show_colorbar=false)
	CM.text!(ax, μas2rad(55.), μas2rad(31.0);text="Time Averaged\nGRMHD", color=:white)
	ax = CM.Axis(fig[2,1], aspect=1);CM.hidedecorations!(ax)
	_imgviz!(fig, ax, model_intmap; show_colorbar=false)
	CM.text!(ax, μas2rad(5.), μas2rad(-45.5);text="PHIBI", color=:white)
	CM.text!(ax, μas2rad(5.), μas2rad(-55.5);text="NxCORR: $(round(1-divergence(nx, model_intmap), digits=3))", color=:white, fontsize=20.0)
	CM.text!(ax, μas2rad(55.), μas2rad(46.5);text=latexstring("a_*= $(round(best_fit.sky.spin, digits=2))"), color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(55.), μas2rad(34.5);text=latexstring("\\theta_o= $(Int(round(best_fit.sky.θo, digits=0)))^\\circ"), color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(55.), μas2rad(23.5);text=latexstring("m_d= $(round(rad2μas(best_fit.sky.m_d), digits=2))\\ \\mu as"), color=:white, fontsize=25.0)
	ax = CM.Axis(fig[2,2], aspect=1);CM.hidedecorations!(ax)
	_imgviz!(fig, ax, inimg; show_colorbar=false)
	CM.text!(ax, μas2rad(-15.), μas2rad(-45.5);text="MAD", color=:white)
	CM.text!(ax, μas2rad(-15.), μas2rad(-55.5);text="Snapshot", color=:white, fontsize=20.0)
	CM.text!(ax, μas2rad(55.), μas2rad(46.5);text=L"a_*=0.5", color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(55.), μas2rad(34.5);text=L"\theta_o=163^\circ", color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(55.), μas2rad(23.5);text=latexstring("m_d= 3.83\\ \\mu as"), color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(-5.), μas2rad(46.5);text=L"R_{\text{high}}=160", color=:white, fontsize=25.0)
	CM.colgap!(fig.layout,1, 5.0)
	CM.rowgap!(fig.layout,1, 5.0)
	CM.save(joinpath((@__DIR__),"mad_mean_model_comp.png"), fig)
	display(fig)
end

fname = "image_sa+0.94_981_163_160_nall.h5"
img_path = joinpath(files, fname)

inimg = rotated(VIDA.load_image(img_path),108.0/180*π)
npix = inimg.X |> length 
fovx, fovy = fieldofview(inimg)
inimg = regrid(inimg, imagepixels(μas2rad(120.0), μas2rad(120.0), 240,240))
npix = inimg.X |> length 
fovx, fovy = fieldofview(inimg)
nx = NxCorr(max.(0.0, inimg))

fpath = open("/Users/dominicchang/Desktop/KerrGMRF/src/plotting/image_sa+0.94_981_163_160_nall.h5_best_fits.txt", "r")
best_fit = eval(Meta.parse(readline(fpath)))
mean_best_fit = deepcopy(best_fit)
@reset mean_best_fit.sky.σimg = 1e-5
close(fpath)
model_intmap = intensitymap(ModifiedKerrGMRF(best_fit.sky, skym.metadata), topgrid)
model_mean_component_intmap = intensitymap(ModifiedKerrGMRF(mean_best_fit.sky, skym.metadata), grid)
mean_best_fit = deepcopy(best_fit)
@reset mean_best_fit.sky.σimg = 1e-5
close(fpath)
ave_grmhd_intmap = Comrade.rotated(regrid(Comrade.load_fits("/Users/dominicchang/Desktop/CenterComparison/data/tavgs_fits_480/sa+0.94_r160_nall_tavg.fits", IntensityMap), grid), π*(180.0-72.0)/180.0)
imageviz(ave_grmhd_intmap)
curr_theme = CM.Theme(
	Image=(rasterize=true,),
	Text=(fontsize=30,)
)
CM.set_theme!(merge(CM.theme_latexfonts(), curr_theme))
begin
	fig = CM.Figure(size=(600,600))
	ax = CM.Axis(fig[1,1], aspect=1);CM.hidedecorations!(ax)
	_imgviz!(fig, ax, model_mean_component_intmap; show_colorbar=false)
	CM.text!(ax, μas2rad(55.), μas2rad(30.5);text="PHIBI\nMean Component", color=:white)
	ax = CM.Axis(fig[1,2], aspect=1);CM.hidedecorations!(ax)
	_imgviz!(fig, ax, ave_grmhd_intmap; show_colorbar=false)
	CM.text!(ax, μas2rad(55.), μas2rad(31.0);text="Time Averaged\nGRMHD", color=:white)
	ax = CM.Axis(fig[2,1], aspect=1);CM.hidedecorations!(ax)
	_imgviz!(fig, ax, model_intmap; show_colorbar=false)
	CM.text!(ax, μas2rad(5.), μas2rad(-45.5);text="PHIBI", color=:white)
	CM.text!(ax, μas2rad(5.), μas2rad(-55.5);text="NxCORR: $(round(1-divergence(nx, model_intmap), digits=3))", color=:white, fontsize=20.0)
	CM.text!(ax, μas2rad(57.), μas2rad(46.5);text=latexstring("a_*= $(round(best_fit.sky.spin, digits=2))"), color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(57.), μas2rad(34.5);text=latexstring("\\theta_o= $(Int(round(best_fit.sky.θo, digits=0)))^\\circ"), color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(57.), μas2rad(23.5);text=latexstring("m_d= $(round(rad2μas(best_fit.sky.m_d), digits=2))\\ \\mu as"), color=:white, fontsize=25.0)
	ax = CM.Axis(fig[2,2], aspect=1);CM.hidedecorations!(ax)
	_imgviz!(fig, ax, inimg; show_colorbar=false)
	CM.text!(ax, μas2rad(-15.), μas2rad(-45.5);text="SANE", color=:white)
	CM.text!(ax, μas2rad(-15.), μas2rad(-55.5);text="Snapshot", color=:white, fontsize=20.0)
	CM.text!(ax, μas2rad(57.), μas2rad(46.5);text=L"a_*=0.94", color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(57.), μas2rad(34.5);text=L"\theta_o=163^\circ", color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(57.), μas2rad(23.5);text=latexstring("m_d= 3.83\\ \\mu as"), color=:white, fontsize=25.0)
	CM.text!(ax, μas2rad(-5.), μas2rad(46.5);text=L"R_{\text{high}}=160", color=:white, fontsize=25.0)
	CM.colgap!(fig.layout,1, 5.0)
	CM.rowgap!(fig.layout,1, 5.0)
	CM.save(joinpath((@__DIR__),"sane_mean_model_comp.png"), fig)
	display(fig)
end

