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
using Enzyme
using Optimization, OptimizationOptimisers
using StableRNGs
using LaTeXStrings
using VLBIImagePriors
using Distributions, DistributionsAD
using Pathfinder
using AdvancedHMC
using Serialization
using MCMCChains
using StatsBase
using ColorSchemes

LinearAlgebra.BLAS.set_num_threads(24) # to avoid threading conflicts with FINUFFT
rng = StableRNG(1234)

include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF.jl"))

function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa))#, Shift(μas2rad(12.0), μas2rad(12.0)))
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
bulkpix = 40
raster_size = 100.0# in microarcseconds    
snrcut = 3.0
uv_min = 0.1e9
fovx = μas2rad(120.0)
fovy = μas2rad(120.0)
npix = 40
year = 2017

bulkpix = 70#20
raster_size = 140.0# 90 # in microarcseconds    
fovx = μas2rad(140.0)
fovy = μas2rad(140.0)
npix = 70
year = 2017

#bulkpix = 70#20
#raster_size = 140.0# 90 # in microarcseconds    
#fovx = μas2rad(140.0)
#fovy = μas2rad(140.0)
#npix = 70
#year = 2017

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 => "L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "30nights_subchanneled_mergedobs_244GHz.uvfits")#/n/home06/dochang/KerrGMRF/data/sim140_frame0026_230.5_GHz_synthdata_ngEHTsim.uvfits")#"frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(dirname(@__DIR__)), "data", data[year])
img_path = joinpath(dirname(dirname(@__DIR__)), "data", "image_ma+0.5_1275_163_1_nall.h5")

# Get observer information
obsin = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min = 0.1e9).add_fractional_noise(fractional_noise)
inimg = ehtim.image.load_image(img_path)
inimg.rf, inimg.ra, inimg.dec = obsin.rf, obsin.ra, obsin.dec
inimg = inimg.rotate(π - 72*pi/180)
inimg.imvec *= 0.65/inimg.total_flux()
inimg.display(scale = "log")
inimg_blur = inimg.blur_circ(μas2rad(10.0))
inimg_blur.display()
inimg.mjd = 57848
obsin.mjd = 57848
obs = inimg.observe_same(obsin, ampcal = ampcal, phasecal = phasecal, add_th_noise = add_th_noise, seed = seed, ttype = "fast")
obs = scan_average(obs.flag_uvdist(uv_min = 0.1e9)).add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))
baselineplot(dvisamp, :uvdist, :measwnoise, error = true, label = "Visibility Amplitudes")

bulkgrid            = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor = ThreadsEx())
transform1, cprior1 = matern(size(bulkgrid))
transform2, cprior2 = matern(size(bulkgrid))
prior               = (
m_d = Uniform(μas2rad(1.0), μas2rad(8.0)),
spin = Uniform(0.01, 0.99),
θo = Uniform(120.0, 179.0),
θs = Uniform(40.5, 90.0),
rpeak = Uniform(1.0, 8.0),
p1 = Uniform(0.1, 5.0),
p2 = Uniform(1.0, 5.0),
χ = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
ι = Uniform(0, π/2),
βv = Uniform(0.01, 0.99),
spec = Uniform(-1.0, 5.0),
η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
frac = VLBIImagePriors.DeltaDist(1.0),
pa = VLBIImagePriors.DeltaDist(π-72*π/180),
f = VLBIImagePriors.DeltaDist(0.65),
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
imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx());
metadata = (; bulkgrid, transform1, transform2, raster_size, offset)
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode = set_runtime_activity(Enzyme.Reverse))
fpost = asflat(post)
run_name=joinpath((@__DIR__), "Results_non_diagonal_metric_grmhd_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
out=joinpath((@__DIR__), run_name)
prechain = load_samples(joinpath(dirname(@__DIR__), run_name))#, 500:100:120_000)
chain = prechain[1500:end]
checkpoints = deserialize(joinpath(out, "checkpoint.jls"))
checkpoints.state[3].state[2].metric.M⁻¹ |> findmax
checkpoints.state[3].state[2].metric.M⁻¹ |> findmin

post = checkpoints.pt.c.rf.xform.model.logdensity.lpost
fpost = asflat(post)
fchain = Comrade.inverse.(Ref(fpost), chain)

CM.lines([i[1] for i in fchain])
avechain = map(x->begin
	t = @reset x.sky.σimg = 1e-7	
end, chain)
msamples = skymodel.(Ref(post), chain)
avemsamples = skymodel.(Ref(post), avechain)
stats = samplerstats(chain)

rand_dir = rand(1:length(cprior1))
CM.lines([i[rand_dir] for i in fchain], linewidth = 0.5)
CM.lines((chain.sky.spin))
m_d, rpeak = begin
	m_d = :m_d in keys(chain[1].sky) ? rad2μas.(chain.sky.m_d) : rad2μas.(sqrt.(chain.sky.m_d_x_rpeak .* chain.sky.m_d_d_rpeak))
	rpeak = :rpeak in keys(chain[1].sky) ? (chain.sky.rpeak) : sqrt.(chain.sky.m_d_x_rpeak ./ chain.sky.m_d_d_rpeak)
	(m_d, rpeak)
end
CM.lines(m_d)

xopt = chain[end]
using Accessors
using StatsBase
temp = rand(chain)#
intensitymap(skymodel(post, temp), skym.grid) |> imageviz

CM.lines(stats.log_density)

Chains(chain.sky.m_d)
hpd(Chains(rad2μas.(chain.sky.θo[1500:end])))
hpd(Chains((chain.sky.θo[1500:end])))
CM.lines(m_d)
CM.lines(moving_average(m_d, 10))
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

h = CM.hist(m_d)

CM.hist((m_d[2000:end]))
CM.hist((chain.sky.spin[1500:end]))
CM.hist((chain.sky.χ[1500:end]))
CM.hist(rpeak)

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
CM.lines([(mean([i[55] for i in chain.sky.c1][(120+i):(140+i)])) for i in 1:(length(m_d)-140)])
begin
	fig = CM.Figure();
	a, b, c, d = rand(3:dimension(post), 4)
	CM.lines!(CM.Axis(fig[1, 1]), [i[a] for i in chain.sky.c1])
	CM.lines!(CM.Axis(fig[1, 2]), [i[b] for i in chain.sky.c1])
	CM.lines!(CM.Axis(fig[2, 1]), [i[c] for i in chain.sky.c1])
	CM.lines!(CM.Axis(fig[2, 2]), [i[d] for i in chain.sky.c1])
	display(fig)
end
CM.scatter(m_d, ([i[rand_dir] for i in chain.sky.c1]), alpha = 0.5)
CM.scatter(([i[1000] for i in chain.sky.c1]), ([i[1200] for i in chain.sky.c1]), alpha = 0.5)
begin
	mvs = CM.scatter(m_d, (chain.sky.spin), alpha = 0.2)
	CM.xlims!(mvs.axis, (2.0, 6.0))
	CM.ylims!(mvs.axis, (0.0, 1.0))
	mvs.axis.xlabel = "Mass (μas)"
	mvs.axis.ylabel = "Spin"
	mvs
end
begin
	mvs = CM.scatter(m_d, rpeak, alpha = 0.5)
	mvs.axis.xlabel = "Mass (μas)"
	mvs.axis.ylabel = "rpeak (GM/c²)"
	mvs
end

cor(m_d, rpeak)

CM.scatter(m_d, (chain.sky.χ), alpha = 0.5)
#CM.scatter(rad2μas.((chain.sky.m_d[120:end])), (chain.sky.pa[120:end]), alpha=0.5)
#CM.scatter((chain.sky.spin[120:end]), (chain.sky.pa[120:end]), alpha=0.5)
#CM.scatter((chain.sky.rpeak[120:end]), (chain.sky.pa[120:end]), alpha=0.5)

fig = CM.Figure(resolution = (700, 700));
ax = CM.Axis(fig[1, 1], aspect = 1)

CM.plot(post.(chain))

psample = Comrade.inverse(fpost, rand(chain))
chi2(post, Comrade.transform(fpost, psample)) ./ (length(dlcamp), length(dcphase))
post(Comrade.transform(fpost, psample))

using MCMCChains
ess((reduce(hcat, fchain))[1, begin:end])

newgrid = imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx())
imgs_blur = intensitymap.(smoothed.(msamples, μas2rad(7/(2.355))), Ref(newgrid))
imgs = intensitymap.(msamples, Ref(newgrid))
aveimgs = intensitymap.(avemsamples, Ref(newgrid))
fig = CM.Figure(size=(300.0,100.0));
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

begin
	inimgnew = begin
		t = inimg.regrid_image(fovx, 3npix)
		IntensityMap(reverse(reverse(reshape(pyconvert(Vector{Float64}, t.ivec), (3npix, 3npix)), dims=1), dims=2), newgrid)
	end
	time_ave_inimg = smooth(Comrade.rotated(regrid(Comrade.load_fits("/n/home06/dochang/KerrGMRF/data/ma+0.5_r1_nall_tavg.fits", IntensityMap), newgrid), π*(108.0)/180.0) |> normalize!, μas2rad(10/(2*√(2*log(2)))))

	inimg_blurnew = begin
		t = inimg_blur.regrid_image(fovx, npix)
		temp = IntensityMap(reverse(reverse(reshape(pyconvert(Vector{Float64}, t.ivec), (npix, npix)), dims=1), dims=2), newgrid)
		temp ./= maximum(temp)
		temp
	end

	rand(imgs) |> imageviz
	mimg = mean(imgs)
	maveimg = mean(aveimgs)
	mimg_blur = mean(imgs_blur)
	mimg_blur ./= maximum(mimg_blur)

	simg = std(imgs)
	fig = CM.Figure(; resolution = (1600, 610));
	axs = [CM.Axis(fig[i, j], xreversed = true, aspect = 1) for i in 1:2, j in 1:5]
	a,b,c = rand(1:length(imgs), 3)
	_imgviz!(fig, axs[2, 1], time_ave_inimg, colormap = :inferno, show_colorbar=false);
	CM.text!(axs[2, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text="Time ave. of Truth", color=:white, fontsize=30.0)
	CM.text!(axs[2, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(30.0), text=L"(10\,\mu as\text{ blur})", color=:white, fontsize=30.0)
	_imgviz!(fig, axs[1, 1], inimg_blurnew, colormap = :inferno, show_colorbar=false);
	CM.text!(axs[1, 1], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text=L"\text{Truth }(10\,\mu as \text{ blur})", color=:white, fontsize=30.0)
	_imgviz!(fig, axs[1, 2], mimg, colormap = :inferno, show_colorbar=false);
	CM.text!(axs[1, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text="Posterior average", color=:white, fontsize=30.0)
	#CM.text!(axs[1, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(32.0), text=L"(7\mu as\; \text{blur})", color=:white, fontsize=30.0)
	_imgviz!(fig, axs[1, 3], imgs[a], colormap = :inferno, show_colorbar=false);
	CM.text!(axs[1, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text="Sample 1", color=:white, fontsize=30.0)
	_imgviz!(fig, axs[2, 3], imgs[b], colormap = :inferno, show_colorbar=false);
	CM.text!(axs[2, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text="Sample 2", color=:white, fontsize=30.0)
	#_imgviz!(fig, axs[2, 3], imgs[c], colormap = :inferno, show_colorbar=false);
	#CM.text!(axs[2, 3], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text="Sample 3", color=:white, fontsize=30.0)


	_imgviz!(fig, axs[2, 5], std(imgs_blur), colormap = :inferno, show_colorbar=false);
	CM.text!(axs[2, 5], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text="Absolule", color=:white, fontsize=30.0)
	CM.text!(axs[2, 5], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(32.0), text="standard dev.", color=:white, fontsize=30.0)
	_imgviz!(fig, axs[2, 2], mean(aveimgs), colormap = :inferno, show_colorbar=false);
	CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text="Average", color=:white, fontsize=30.0)
	CM.text!(axs[2, 2], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(32.0), text="mean component", color=:white, fontsize=30.0)
	#CM.text!(axs[1, 4], fovx/2-μas2rad(5.0), fovy/2 - μas2rad(40.0), text="Average mean\ncomponent", color=:white, fontsize=30.0, font=:bold)
	_imgviz!(fig, axs[1, 4], aveimgs[a], colormap = :inferno, show_colorbar=false);
	CM.text!(axs[1, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text=L"\text{Mean component}", color=:white, fontsize=30.0)
	CM.text!(axs[1, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(32.0), text=L"\text{of sample 1}", color=:white, fontsize=30.0)
	fracstd = _imgviz!(fig, axs[1, 5], simg ./ (max.(mimg, eps())), colorrange=(0, 1.2), colormap = :viridis, show_colorbar=false, highclip=ColorSchemes.colorschemes[:viridis][end])#, lowclip=ColorSchemes.colorschemes[:viridis][begin]);
	CM.text!(axs[1, 5], fovx/2-μas2rad(5.0), fovy/2 - μas2rad(40.0), text="Fractional\nstandard dev.", color=:white, fontsize=30.0, strokewidth=20.0, font=:bold)
	fracsnp = _imgviz!(fig, axs[2, 4], aveimgs[b] ./ maximum(aveimgs[b]), colormap = :inferno, show_colorbar=false);
	CM.text!(axs[2, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(17.0), text=L"\text{Mean component}", color=:white, fontsize=30.0)
	CM.text!(axs[2, 4], fovx/2-μas2rad(7.0), fovy/2 - μas2rad(32.0), text=L"\text{of sample 2}", color=:white, fontsize=30.0)
	#ax = CM.Axis(fig[1,6])
	CM.Colorbar(fig[1,6], fracstd.plot)
	CM.Colorbar(fig[2,6], fracsnp.plot, label=L"\text{Rel. intensity mJy/}\mu as")


	CM.hidedecorations!.(axs)

	for i in 1:1
		CM.rowgap!(fig.layout, i, 20.0)
	end
	for j in 1:2
		CM.colgap!(fig.layout, j, 0.0)
	end
	CM.colgap!(fig.layout, 3, 30.0)
	CM.colgap!(fig.layout, 4, 0.0)

	CM.save("GRMHD_visibility_posterior_draws.png", fig)

	fig
end

using Plots
p = Plots.plot(layout = (1, 2));
for s in sample(chain[1000:end], 10)
	residual!(post, s)
end
p