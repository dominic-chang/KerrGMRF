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
inimg.blur_circ(μas2rad(7.0)).display()
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
chain = load_samples(joinpath(dirname(@__DIR__), run_name))
checkpoints = deserialize(joinpath(out, "checkpoint.jls"))
checkpoints.state[3].state[2].metric.M⁻¹ |> findmax
checkpoints.state[3].state[2].metric.M⁻¹ |> findmin

post = checkpoints.pt.c.rf.xform.model.logdensity.lpost
fpost = asflat(post)
fchain = Comrade.inverse.(Ref(fpost), chain)

CM.lines([i[1] for i in fchain])
msamples = skymodel.(Ref(post), chain)
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
temp = chain[end]
intensitymap(skymodel(post, temp), skym.grid) |> imageviz

CM.lines(stats.log_density)

Chains(chain.sky.m_d)
hpd(Chains(rad2μas.(chain.sky.θo[5_000:end])))
hpd(Chains((chain.sky.θo[5_000:end])))
CM.lines(m_d)
CM.lines(moving_average(m_d, 10))
CM.lines((chain.sky.spin[5_000:end]))
CM.lines(rpeak)
CM.lines((chain.sky.frac))
CM.lines((chain.sky.pa .* 180/π))
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
CM.hist((chain.sky.spin))
CM.hist((chain.sky.χ))
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
	mvs = CM.scatter(m_d[15_000:end], (chain[15_000:end].sky.spin), alpha = 0.2)
	CM.lines!(mvs.axis, [3.83 for i in 1:10], range(-1, 1, length = 10), color = :red, linestyle = :dash)
	CM.lines!(mvs.axis, range(2.0, 6.0, length = 10), [0.5 for i in 1:10], color = :red, linestyle = :dash)
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

newgrid = imagepixels(fovx, fovy, 2*npix, 2*npix; executor = ThreadsEx())

imgs = intensitymap.(msamples, Ref(newgrid))
fig = CM.Figure();
CM.image!(CM.Axis(fig[1, 1], xreversed = true, aspect = 1), intensitymap(smoothed(msamples[end], μas2rad(10/(2.355))), newgrid), colormap = :afmhot)
CM.image!(CM.Axis(fig[1, 1], xreversed = false, aspect = 1), intensitymap(msamples[1], newgrid), colormap = :afmhot)
fig
rand(imgs) |> imageviz
mimg = mean(imgs)
simg = std(imgs)
fig = CM.Figure(; resolution = (700, 700));
axs = [CM.Axis(fig[i, j], xreversed = true, aspect = 1) for i in 1:2, j in 1:2]
CM.image!(axs[1, 1], mimg, colormap = :afmhot);
axs[1, 1].title="Mean"
CM.image!(axs[1, 2], simg ./ (max.(mimg, 1e-2)), colorrange = (0.0, 2.0), colormap = :afmhot);
axs[1, 2].title = "Std"
CM.image!(axs[2, 1], rand(imgs), colormap = :afmhot);
CM.image!(axs[2, 2], rand(imgs), colormap = :afmhot);
CM.hidedecorations!.(axs)
fig

using Plots
p = Plots.plot(layout = (1, 2));
for s in sample(chain[1000:end], 10)
	residual!(post, s)
end
p