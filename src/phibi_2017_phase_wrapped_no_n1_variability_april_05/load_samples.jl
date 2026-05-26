using Pkg;
Pkg.activate(dirname(dirname(@__DIR__)));
using Comrade
using Pyehtim
using Krang
using StableRNGs
using Accessors
import CairoMakie as CM
using BasicInterpolators
using LinearAlgebra
using FINUFFT
using VLBIImagePriors
using Distributions, DistributionsAD
using Enzyme
using Optimization, OptimizationOptimisers
using StableRNGs
using LaTeXStrings
using ColorSchemes: ColorSchemes
using MCMCChains
using AdvancedHMC


LinearAlgebra.BLAS.set_num_threads(1) # to avoid threading conflicts with FINUFFT
rng = StableRNG(1234)


include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF_no_n1_variability.jl"))

function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa*π/180), Shift(μas2rad(6.0), μas2rad(0.0)))
	return RenormalizedFlux(m, θ2.f)
end

lr = 0.01
seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.01
bulkx = 1.0
bulky = 1.0
snrcut = 3.0
uv_min = 0.1e9
bulkpix = 90
raster_size = 180.0 # in microarcseconds    
fovx = μas2rad(180.0)
fovy = μas2rad(120.0)
npixx = 90
npixy = 60
year = 2017

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 => "L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(dirname(@__DIR__)), "data", data[year])

# Get observer information
obs = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min = 0.1e9).add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))

bulkgrid            = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor = ThreadsEx())
transform1, cprior1 = matern(size(bulkgrid))
transform2, cprior2 = matern(size(bulkgrid))
prior               = (
m_d = Uniform(μas2rad(1.0), μas2rad(8.0)),
spin = Uniform(0.01, 0.99),
θo = Uniform(120.0, 179.0),
θs = Uniform(40.0, 90.0),
rpeak = Uniform(1.0, 10.0),
p1 = Uniform(0.1, 5.0),
p2 = Uniform(1.0, 5.0),
χ = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)), #pa =Uniform(0.0, 180.0),
ι = Uniform(0, π/2),
βv = Uniform(0.01, 0.99),
spec = Uniform(-1.0, 5.0),
η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
frac = VLBIImagePriors.DeltaDist(1.0),
pa = VLBIImagePriors.DeltaDist(π-72*π/180),# Uniform(-π, 0),
f = VLBIImagePriors.DeltaDist(0.6),
σimg = truncated(Normal(0.0, 1.0); lower = 0.0),
ρpr = truncated(InverseGamma(1.0, -log(0.1)*10); lower = 1.0, upper = 2*max(size(bulkgrid)...)),
νpr = Uniform(1.0, 5.0),
c1 = cprior1,
c2 = cprior2
)
offset              = 0.0
skym                = SkyModel(
ModifiedKerrGMRF,
prior,
imagepixels(fovx, fovy, npixx, npixy; executor = ThreadsEx());
metadata = (; bulkgrid, transform1, transform2, raster_size, offset),
algorithm = FINUFFTAlg(; threads = 1)
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode = set_runtime_activity(Enzyme.Reverse))
fpost = asflat(post)

# Import data from disk
using Serialization
run_name = joinpath((@__DIR__), "Results_non_diagonal_metric_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npixx)_$(npixy)_res_$(Int(floor(rad2μas(fovx))))_$(Int(floor(rad2μas(fovy))))_fov_year_$(year)_phase_wrapped")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")

prechain = load_samples(joinpath(dirname(@__DIR__), run_name))
chain = prechain[6000:end]
checkpoints = deserialize(joinpath(run_name, "checkpoint.jls"))
M_inv = checkpoints.state[3].state[2].metric.M⁻¹
#@show extrema(M_inv)
#@show isposdef(diagm(M_inv))

post = checkpoints.pt.c.rf.xform.model.logdensity.lpost
fpost = asflat(post)
fchain = Comrade.inverse.(Ref(fpost), chain)
avechain = map(x->begin
		t = @reset x.sky.σimg = 1e-7
	end, chain)

CM.lines([i[1] for i in fchain])
msamples = skymodel.(Ref(post), chain)[700:end]
avemsamples = skymodel.(Ref(post), avechain)[700:end]
stats = samplerstats(chain)

CM.lines((chain.sky.spin))
m_d, rpeak = begin
	m_d = :m_d in keys(chain[1].sky) ? rad2μas.(chain.sky.m_d) : rad2μas.(sqrt.(chain.sky.m_d_x_rpeak .* chain.sky.m_d_d_rpeak))
	rpeak = :rpeak in keys(chain[1].sky) ? (chain.sky.rpeak) : sqrt.(chain.sky.m_d_x_rpeak ./ chain.sky.m_d_d_rpeak)
	(m_d, rpeak)
end
#m_d = rad2μas.(chain.sky.m_d)
#rpeak = sqrt.(chain.sky.m_d_x_rpeak ./ chain.sky.m_d_d_rpeak)
#rpeak = chain.sky.rpeak
CM.lines(m_d)

xopt = chain[end]
using Accessors
using StatsBase
using MCMCChains
temp = rand(chain)


rh = 1 + √(1-temp.sky.spin^2)
mat1 = []
bulkmodel = bulk(transform1(temp.sky.c1, temp.sky.ρpr, temp.sky.νpr), temp.sky.σimg, bulkgrid)
for x in LinRange(-50.0,50.0,100)
	for y in LinRange(-50.0,50.0,100)
		r = sqrt(x^2 + y^2)
		ϕ = atan(y, x)
		rat = (r / temp.sky.rpeak)
		rs_grid = (r - rh) * rad2μas(temp.sky.m_d) / (raster_size - rh * rad2μas(temp.sky.m_d)) # convert to microarcseconds
		dim = (X = rs_grid * cos(ϕ) + offset, Y = rs_grid * sin(ϕ) + offset)
		cp = exp(ComradeBase.intensity_point(bulkmodel, dim) / (bulkpix^2))
		ans = rat^temp.sky.p1 / (1 + rat^(temp.sky.p1 + temp.sky.p2)) * cp# max(redshift, eps())^(3 + temp.sky.spec) * cp
		push!(mat1, ans)
	end
end
reverse(reverse(reshape(mat1, (100,100))', dims=2), dims=1) |> heatmap

mat2 = []
bulkmodel = bulk(transform1(temp.sky.c2, temp.sky.ρpr, temp.sky.νpr), temp.sky.σimg, bulkgrid)
for x in LinRange(-50.0,50.0,100)
	for y in LinRange(-50.0,50.0,100)
		r = sqrt(x^2 + y^2)
		ϕ = atan(y, x)
		rat = (r / temp.sky.rpeak)
		rs_grid = (r - rh) * rad2μas(temp.sky.m_d) / (raster_size - rh * rad2μas(temp.sky.m_d)) # convert to microarcseconds
		dim = (X = rs_grid * cos(ϕ) + offset, Y = rs_grid * sin(ϕ) + offset)
		cp = exp(ComradeBase.intensity_point(bulkmodel, dim) / (bulkpix^2))
		ans = rat^temp.sky.p1 / (1 + rat^(temp.sky.p1 + temp.sky.p2)) * cp# max(redshift, eps())^(3 + temp.sky.spec) * cp
		push!(mat2, ans)
	end
end
reverse(reverse(reshape(mat2, (100,100))', dims=2), dims=1) |> heatmap
reverse(reverse(reshape(mat1 .+ mat2, (100,100))', dims=2), dims=1) |> heatmap
intensitymap(post.skymodel.f(temp.sky, post.skymodel.metadata), imagepixels(fovx, fovy, npixx, npixy; executor = ThreadsEx())) |> imageviz 

CM.set_theme!(CM.theme_dark())
CM.lines(stats.log_density)

hpd(Chains(rad2μas.(chain.sky.θo)))
hpd(Chains((chain.sky.θo)))
CM.lines(m_d)
CM.lines(moving_average(m_d, 1_00))
CM.lines((chain.sky.spin))
CM.lines(rpeak)
CM.lines((chain.sky.θo))
CM.lines((chain.sky.θs))
CM.lines((chain.sky.χ))
CM.lines((chain.sky.νpr))
CM.lines((chain.sky.ρpr))
CM.lines((chain.sky.σimg))
CM.lines((chain.sky.p1))
CM.lines((chain.sky.p2))
CM.lines((chain.sky.βv))

CM.hist(m_d)
CM.hist(chain.sky.θo)
CM.hist((chain.sky.spin))
CM.hist((chain.sky.χ))
CM.hist(rpeak)

#for j in 1:length(chain.sky.c1[1])
#    CM.lines([chain.sky.c1[i][j] for i in 5000:length(chain.sky.c1)]) |> display
#end

begin
	fig = CM.Figure();
	ax = CM.Axis(fig[1, 1], yscale = log10)
	CM.lines!(ax, stats.step_size)
	display(fig)
end
begin
	fig = CM.Figure();
	ax = CM.Axis(fig[1, 1], yscale = log10)
	CM.lines!(ax, moving_average(stats.step_size, 10))
	display(fig)
end
CM.scatter(stats.numerical_error, alpha = 0.1)
CM.scatter(stats.tree_depth, alpha = 0.2)


using MCMCChains
ess((reduce(hcat, fchain))[1, begin:end])

using StatsBase
using LinearAlgebra

newgrid = imagepixels(fovx, fovy, npixx, npixy; executor = ThreadsEx())
imgs = intensitymap.(msamples, Ref(newgrid))
aveimgs = intensitymap.(avemsamples, Ref(newgrid))
imgs_blur = smooth.(imgs, μas2rad(7/(2.355)))
aveimgs_blur = smooth.(aveimgs, μas2rad(7/(2.355)))
fig = CM.Figure(size = (300.0, 100.0));
CM.image!(CM.Axis(fig[1, 1], xreversed = true, aspect = 1), intensitymap(smoothed(msamples[end], μas2rad(10/(2.355))), newgrid), colormap = :afmhot)
CM.image!(CM.Axis(fig[1, 1], xreversed = false, aspect = 1), intensitymap(msamples[1], newgrid), colormap = :afmhot)

currtheme = CM.Theme(
	margin = (0.0, 0.0, 0.0, 0.0),
	padding = (0.0, 0.0, 0.0, 0.0),
	Text = (fontsize = 30.0,),
	Colorbar = (ticklabelsize = 25.0, labelsize = 25.0),
	Axis = (margin = (0.0, 0.0, 0.0, 0.0),),
)
CM.set_theme!(merge(currtheme, CM.theme_latexfonts()))
inimg = Comrade.regrid(Comrade.load_fits("/n/home06/dochang/KerrGMRF/src/phibi_2017_phase_wrapped_no_n1_variability_april_05/3598_blur_avg.fits", IntensityMap), imagepixels(fovx, fovy, 3npixx, 3npixy; executor = ThreadsEx()))
mimg = mean(imgs)
maveimg = mean(aveimgs)
mimg_blur = mean(imgs_blur)
mimg_blur ./= maximum(mimg_blur)
simg = std(imgs)

#fig = CM.Figure(; resolution = (1250, 610*fovy/fovx));
#CM.record(fig, "data_samples.gif", 1:50; framerate=2) do h
##begin
#
#	axs = [CM.Axis(fig[i, j], xreversed = true, aspect = fovx/fovy) for i in 1:2, j in 1:5]
#	a, b, c = rand(1:length(imgs), 3)
#	_imgviz!(fig, axs[1, 1], inimg, colormap = :inferno, show_colorbar = false);
#	CM.text!(axs[1, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "EHT consensus", color = :white, fontsize = 30.0)
#	#_imgviz!(fig, axs[1, 2], inimg_blurnew, colormap = :inferno, show_colorbar=false);
#	#CM.text!(axs[1, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text=L"\text{Truth }(10\,\mu as \text{ blur})", color=:white, fontsize=30.0)
#	_imgviz!(fig, axs[2, 1], mimg, colormap = :inferno, show_colorbar = false);
#	CM.text!(axs[2, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Posterior average", color = :white, fontsize = 30.0)
#	#CM.text!(axs[1, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"(7\mu as\; \text{blur})", color = :white, fontsize = 30.0)
#	_imgviz!(fig, axs[1, 2], imgs[a], colormap = :inferno, show_colorbar = false);
#	CM.text!(axs[1, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Sample 1", color = :white, fontsize = 30.0)
#	_imgviz!(fig, axs[2, 2], imgs[b], colormap = :inferno, show_colorbar = false);
#	CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Sample 2", color = :white, fontsize = 30.0)
#	#_imgviz!(fig, axs[2, 2], imgs[c], colormap = :inferno, show_colorbar = false);
#	#_imgviz!(fig, axs[2, 2], mean(aveimgs), colormap = :inferno, show_colorbar=false);
#	#CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Average", color = :white, fontsize = 30.0)
#	#CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = "mean component", color = :white, fontsize = 30.0)
#
#	_imgviz!(fig, axs[2, 4], std(imgs_blur), colormap = :inferno, show_colorbar = false);
#	#CM.text!(axs[1, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Average mean\n component", color = :white, fontsize = 30.0)
#	CM.text!(axs[2, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Absolute", color = :white, fontsize = 30.0)
#	CM.text!(axs[2, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = "standard dev.", color = :white, fontsize = 30.0)
#	#CM.text!(axs[1, 4], fovx/2-μas2rad(7.5), fovy/2 - μas2rad(20.0), text = "Absolute", color = :white, fontsize = 30.0)
#	#CM.text!(axs[1, 4], fovx/2-μas2rad(7.5), fovy/2 - μas2rad(40.0), text = "standard dev.", color = :white, fontsize = 30.0)
#	_imgviz!(fig, axs[1, 3], aveimgs[a], colormap = :inferno, show_colorbar = false);
#	CM.text!(axs[1, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = L"\text{Mean component}", color = :white, fontsize = 30.0)
#	CM.text!(axs[1, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"\text{of sample 1}", color = :white, fontsize = 30.0)
#	fracstd = _imgviz!(fig, axs[1, 4], simg ./ (max.(mimg, eps())), colorrange = (0, 1.5), colormap = :viridis, show_colorbar = false, highclip = ColorSchemes.colorschemes[:viridis][end]);
#	CM.text!(axs[1, 4], fovx/2-μas2rad(5.0), fovy/2 - μas2rad(20.0), text = "Fractional", color = :white, fontsize = 30.0, strokewidth = 20.0)#, font=:bold)
#	CM.text!(axs[1, 4], fovx/2-μas2rad(5.0), fovy/2 - μas2rad(40.0), text = "standard dev.", color = :white, fontsize = 30.0, strokewidth = 20.0)#, font=:bold)
#	CM.text!(axs[1, 4], fovx/2-μas2rad(5.5), fovy/2 - μas2rad(20.0), text = "Fractional", color = :white, fontsize = 30.0, strokewidth = 20.0)#, font=:bold)
#	CM.text!(axs[1, 4], fovx/2-μas2rad(5.5), fovy/2 - μas2rad(40.0), text = "standard dev.", color = :white, fontsize = 30.0, strokewidth = 20.0)#, font=:bold)
#
#	fracsnp = _imgviz!(fig, axs[2, 3], aveimgs[b] ./ maximum(aveimgs[b]), colormap = :inferno, show_colorbar = false);
#	CM.text!(axs[2, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = L"\text{Mean component}", color = :white, fontsize = 30.0)
#	CM.text!(axs[2, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"\text{of sample 2}", color = :white, fontsize = 30.0)
#	#ax = CM.Axis(fig[1,6])
#	CM.Colorbar(fig[1, 5], fracstd.plot)
#	CM.Colorbar(fig[2, 5], fracsnp.plot, label = CM.rich("Rel. Intensity\n", "(mJy/", CM.rich("μas", font = :italic), ")"), labelsize = 25.0)
#
#
#	CM.hidedecorations!.(axs)
#
#	for i in 1:1
#		CM.rowgap!(fig.layout, i, 20.0)
#	end
#	for j in 1:2
#		CM.colgap!(fig.layout, j, 1.0)
#	end
#	CM.colgap!(fig.layout, 3, 20.0)
#	CM.colgap!(fig.layout, 4, 1.0)
#
#	fig
#end



rand(imgs) |> imageviz
mimg = mean(imgs)
maveimg = mean(aveimgs)
mimg_blur = mean(imgs_blur)
mimg_blur ./= maximum(mimg_blur)


#CM.record(fig, "GRMHD_samples.gif", 1:10; framerate=1) do h
begin

    fig = CM.Figure(; resolution = (1550, 610*fovy/fovx));
		simg = std(imgs)
	axs = [CM.Axis(fig[i, j], xreversed = true, aspect = fovx/fovy) for i in 1:2, j in 1:5]
	a, b, c = rand(1:length(imgs), 3)
	_imgviz!(fig, axs[1, 1], inimg, colormap = :inferno, show_colorbar = false);
	CM.text!(axs[1, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "EHT consensus", color = :white, fontsize = 30.0)
	#_imgviz!(fig, axs[1, 2], inimg_blurnew, colormap = :inferno, show_colorbar=false);
	#CM.text!(axs[1, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text=L"\text{Truth }(10\,\mu as \text{ blur})", color=:white, fontsize=30.0)
	_imgviz!(fig, axs[2, 1], mimg, colormap = :inferno, show_colorbar = false);
	CM.text!(axs[2, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Posterior average", color = :white, fontsize = 30.0)
	#CM.text!(axs[1, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"(7\mu as\; \text{blur})", color = :white, fontsize = 30.0)
	_imgviz!(fig, axs[1, 3], imgs[a], colormap = :inferno, show_colorbar = false);
	CM.text!(axs[1, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Sample 1", color = :white, fontsize = 30.0)
	_imgviz!(fig, axs[2, 3], imgs[b], colormap = :inferno, show_colorbar = false);
	CM.text!(axs[2, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Sample 2", color = :white, fontsize = 30.0)
	#_imgviz!(fig, axs[2, 2], imgs[c], colormap = :inferno, show_colorbar = false);
	_imgviz!(fig, axs[2, 2], mean(aveimgs), colormap = :inferno, show_colorbar=false);
	CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Average", color = :white, fontsize = 30.0)
	CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = "mean component", color = :white, fontsize = 30.0)

	_imgviz!(fig, axs[2, 5], std(imgs_blur), colormap = :inferno, show_colorbar = false);
	#CM.text!(axs[1, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Average mean\n component", color = :white, fontsize = 30.0)
	CM.text!(axs[2, 5], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Absolute", color = :white, fontsize = 30.0)
	CM.text!(axs[2, 5], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = "standard dev.", color = :white, fontsize = 30.0)
	#CM.text!(axs[1, 4], fovx/2-μas2rad(7.5), fovy/2 - μas2rad(20.0), text = "Absolute", color = :white, fontsize = 30.0)
	#CM.text!(axs[1, 4], fovx/2-μas2rad(7.5), fovy/2 - μas2rad(40.0), text = "standard dev.", color = :white, fontsize = 30.0)
	_imgviz!(fig, axs[1, 4], aveimgs[a], colormap = :inferno, show_colorbar = false);
	CM.text!(axs[1, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = L"\text{Mean component}", color = :white, fontsize = 30.0)
	CM.text!(axs[1, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"\text{of sample 1}", color = :white, fontsize = 30.0)
	fracstd = _imgviz!(fig, axs[1, 5], simg ./ (max.(mimg, eps())), colorrange = (0, 1.5), colormap = :viridis, show_colorbar = false, highclip = ColorSchemes.colorschemes[:viridis][end]);
	CM.text!(axs[1, 5], fovx/2-μas2rad(5.0), fovy/2 - μas2rad(20.0), text = "Fractional", color = :white, fontsize = 30.0, strokewidth = 20.0)#, font=:bold)
	CM.text!(axs[1, 5], fovx/2-μas2rad(5.0), fovy/2 - μas2rad(40.0), text = "standard dev.", color = :white, fontsize = 30.0, strokewidth = 20.0)#, font=:bold)
	CM.text!(axs[1, 5], fovx/2-μas2rad(5.5), fovy/2 - μas2rad(20.0), text = "Fractional", color = :white, fontsize = 30.0, strokewidth = 20.0)#, font=:bold)
	CM.text!(axs[1, 5], fovx/2-μas2rad(5.5), fovy/2 - μas2rad(40.0), text = "standard dev.", color = :white, fontsize = 30.0, strokewidth = 20.0)#, font=:bold)

	fracsnp = _imgviz!(fig, axs[2, 4], aveimgs[b] ./ maximum(aveimgs[b]), colormap = :inferno, show_colorbar = false);
	CM.text!(axs[2, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = L"\text{Mean component}", color = :white, fontsize = 30.0)
	CM.text!(axs[2, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"\text{of sample 2}", color = :white, fontsize = 30.0)
	#ax = CM.Axis(fig[1,6])
	CM.Colorbar(fig[1, 6], fracstd.plot)
	CM.Colorbar(fig[2, 6], fracsnp.plot, label = CM.rich("Rel. Intensity\n", "(mJy/", CM.rich("μas", font = :italic), ")"), labelsize = 25.0)


	CM.hidedecorations!.(axs)

	for i in 1:1
		CM.rowgap!(fig.layout, i, 20.0)
	end
	for j in 1:2
		CM.colgap!(fig.layout, j, 1.0)
	end
	CM.colgap!(fig.layout, 3, 20.0)
	CM.colgap!(fig.layout, 4, 1.0)

	CM.save("data_visibility_posterior_draws.png", fig)

	fig
end

begin
    fig = CM.Figure(; resolution = (1100, 610*fovy/fovx));
	rand(imgs) |> imageviz
	mimg = mean(imgs)
	maveimg = mean(aveimgs)
	mimg_blur = mean(imgs_blur)
	mimg_blur ./= maximum(mimg_blur)

	axs = [CM.Axis(fig[i, j], xreversed = true, aspect = fovx/fovy) for i in 1:1, j in 1:2]
	CM.hidedecorations!.(axs)
	a, b, c = rand(1:length(imgs), 3)
	_imgviz!(fig, axs[1, 1], inimg, colormap = :afmhot, show_colorbar = false, show_scalebar=false);
	#CM.text!(axs[1, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "EHT consensus", color = :white, fontsize = 50.0)
	#_imgviz!(fig, axs[1, 2], inimg_blurnew, colormap = :inferno, show_colorbar=false);
	#CM.text!(axs[1, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text=L"\text{Truth }(10\,\mu as \text{ blur})", color=:white, fontsize=30.0)
	_imgviz!(fig, axs[1, 2], imgs, colormap = :afmhot, show_colorbar = false, show_scalebar=false);
	#CM.text!(axs[1, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "In prep.", color = :white, fontsize = 50.0)
	CM.colgap!(fig.layout, 1, 0.0)
	fig
end
	

imgs_blur = smooth.(imgs, μas2rad(10/(2.355)))
aveimgs_blur = smooth.(aveimgs, μas2rad(10/(2.355)))

begin

    fig = CM.Figure(; resolution = (600, 1210*fovy/fovx));

	axs = [CM.Axis(fig[i, j], xreversed = true, aspect = fovx/fovy) for i in 1:4, j in 1:2]
	a, b, c, d = rand(1:length(imgs), 4)
	#_imgviz!(fig, axs[1, 1], inimg, colormap = :inferno, show_colorbar = false);
	#CM.text!(axs[1, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "EHT consensus", color = :white, fontsize = 30.0)
	_imgviz!(fig, axs[1, 1], imgs_blur[a], colormap = :inferno, show_colorbar = false);
	CM.text!(axs[1, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Sample 1", color = :white, fontsize = 30.0)
	CM.text!(axs[1, 1], μas2rad(2.0), -fovy/2 + μas2rad(5.0), text = L"(10\mu as\; \text{blur})", color = :white, fontsize = 25.0)
	_imgviz!(fig, axs[1, 2], aveimgs_blur[a], colormap = :inferno, show_colorbar = false);
	CM.text!(axs[1, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = L"\text{Mean component}", color = :white, fontsize = 30.0)
	CM.text!(axs[1, 2], μas2rad(2.0), -fovy/2 + μas2rad(5.0), text = L"(10\mu as\; \text{blur})", color = :white, fontsize = 25.0)
	_imgviz!(fig, axs[2, 1], imgs_blur[b], colormap = :inferno, show_colorbar = false);
	CM.text!(axs[2, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Sample 2", color = :white, fontsize = 30.0)
	#_imgviz!(fig, axs[2, 2], imgs[c], colormap = :inferno, show_colorbar = false);
	_imgviz!(fig, axs[2, 2], aveimgs_blur[b], colormap = :inferno, show_colorbar=false);
	CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = L"\text{Mean component}", color = :white, fontsize = 30.0)
	CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"\text{of sample 2}", color = :white, fontsize = 30.0)
	_imgviz!(fig, axs[3, 1], imgs_blur[c], colormap = :inferno, show_colorbar = false);
	CM.text!(axs[3, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Sample 3", color = :white, fontsize = 30.0)
	#_imgviz!(fig, axs[2, 2], imgs[c], colormap = :inferno, show_colorbar = false);
	_imgviz!(fig, axs[3, 2], aveimgs_blur[c], colormap = :inferno, show_colorbar=false);
	CM.text!(axs[3, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = L"\text{Mean component}", color = :white, fontsize = 30.0)
	CM.text!(axs[3, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"\text{of sample 3}", color = :white, fontsize = 30.0)
	_imgviz!(fig, axs[4, 1], imgs_blur[d], colormap = :inferno, show_colorbar = false);
	CM.text!(axs[4, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = "Sample 4", color = :white, fontsize = 30.0)
	#_imgviz!(fig, axs[2, 2], imgs[c], colormap = :inferno, show_colorbar = false);
	_imgviz!(fig, axs[4, 2], aveimgs_blur[d], colormap = :inferno, show_colorbar=false);
	CM.text!(axs[4, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(20.0), text = L"\text{Mean component}", color = :white, fontsize = 30.0)
	CM.text!(axs[4, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(40.0), text = L"\text{of sample 4}", color = :white, fontsize = 30.0)
	CM.hidedecorations!.(axs)
	for i in 1:3
		CM.rowgap!(fig.layout, i, 5.0)
	end
	CM.colgap!(fig.layout, 1, 5.0)

	CM.save("data_visibility_posterior_draws_blured.png", fig)
	fig

end