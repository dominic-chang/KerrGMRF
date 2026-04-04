using Pkg;
Pkg.activate(dirname(dirname(@__DIR__)));
using Comrade
using Pyehtim
using Krang
using Enzyme
using StableRNGs
using Accessors
import CairoMakie as CM
using BasicInterpolators
using LinearAlgebra
using FINUFFT
LinearAlgebra.BLAS.set_num_threads(24) # to avoid threading conflicts with FINUFFT
rng = StableRNG(1234)

include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF2.jl"))

lr = 0.1
seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.01
bulkx = 1.0
bulky = 1.0
bulkpix = 40#20
raster_size = 100.0# 90 # in microarcseconds    
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
inimg.display()#scale = "log")
inimg.rf = obsin.rf
inimg.ra = obsin.ra
inimg.dec = obsin.dec
inimg = inimg.rotate(π - 72*pi/180)
inimg.display()
inimg.total_flux()
inimg = ehtim.image.Image.regrid_image(inimg, fovx, 480)
inimg.imvec *= 0.65/inimg.total_flux()
inimg.display(scale = "log")
inimg.blur_circ(μas2rad(7.0)).display()
inimg.mjd = 57848
obsin.mjd = 57848
obs = inimg.observe_same(obsin, ampcal = ampcal, phasecal = phasecal, add_th_noise = add_th_noise, seed = seed, ttype = "fast")
obs = scan_average(obs.flag_uvdist(uv_min = 0.1e9))
obs = obs.add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))
baselineplot(dvisamp, :uvdist, :measwnoise, error = true, label = "Visibility Amplitudes")

using VLBIImagePriors
using Distributions, DistributionsAD

function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa))#, Shift(μas2rad(12.0), μas2rad(12.0)))
	return RenormalizedFlux(m, θ2.f)
end

using Optimization
using OptimizationOptimisers
using Enzyme
using StableRNGs
using Plots: Plots
using LaTeXStrings

bulkgrid            = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor = ThreadsEx())
transform1, cprior1 = matern(size(bulkgrid))
transform2, cprior2 = matern(size(bulkgrid))
prior               = (
m_d = Uniform(μas2rad(1.0), μas2rad(8.0)), #Uniform(μas2rad(2.0), μas2rad(6.0)),  #m_d_d_rpeak = Uniform(μas2rad(2.0/8.0), μas2rad(6.0)),  #m_d_x_rpeak = Uniform(μas2rad(2.0), μas2rad(6.0*8.0)),
spin = Uniform(0.01, 0.99),#Uniform(-0.99, -0.01),#-0.99, -0.01),
θo = Uniform(120.0, 179.0), #θs = VLBIImagePriors.DeltaDist(90.0),
θs = Uniform(40.5, 90.0),
rpeak = Uniform(1.0, 8.0),
p1 = Uniform(0.1, 5.0),
p2 = Uniform(1.0, 5.0),
χ = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
ι = Uniform(0, π/2),
βv = Uniform(0.01, 0.99),
spec = Uniform(-1.0, 5.0),
η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
frac = VLBIImagePriors.DeltaDist(1.0),#Uniform(0.0, 1.0),
pa = VLBIImagePriors.DeltaDist(π-72*π/180),# Uniform(-π, 0),
f = VLBIImagePriors.DeltaDist(0.65),#Uniform(-π, π),
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
psample = prior_sample(post)
Comrade.transform(fpost, rand(dimension(fpost)))

vals = rand(dimension(fpost))
dvals = zeros(size(vals))
using LaTeXStrings
mutable struct Callback
	counter::Int
	stride::Int
	const f::Function
end
Callback(stride, f) = Callback(0, stride, f)

loss_arr = []
fig = CM.Figure(; size = (1200, 600));
num_data_prods = length(post.data)
function (c::Callback)(state, loss, others...)
	global loss_arr
	global fpost
	global st
	c.counter += 1
	curr = transform(fpost, state.u)
	fig = CM.Figure(; size = (1200, 600));

	#CM.empty!(fig)
	if c.counter % c.stride == 0
		append!(loss_arr, loss)
		curr = transform(fpost, state.u)
		resids = Comrade.residuals(post, curr)
		chi2_vals = Comrade.chi2(post, curr) ./ (length(resids[i].measurement) for i in 1:length(resids))
		for (i, resid) in enumerate(resids)
			ax = CM.Axis(
				fig[i, 1],
				xlabel = CM.L"\sqrt{\text{(Convex Quadrangle Area)}} (G$\lambda$)",
				ylabel = LaTeXStrings.latexstring("\\text{Residual $(typeof(resid).parameters[1].name.name)}"),
				title = LaTeXStrings.latexstring("\\langle\\chi^2\\rangle=$(round(chi2_vals[i], digits=2))"),
				titlesize = 24,
				xlabelsize = 24,
				ylabelsize = 14,
			)
			baselineplot!(ax, resid, :uvdist, :measwnoise)
		end
		imgax = CM.Axis(fig[1:length(resids), 2], aspect = 1, xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
		_imgviz!(fig, imgax, intensitymap(skymodel(post, curr), skym.grid))
		CM.text!(imgax, μas2rad(40.0), μas2rad(40.0); text = "m/d :$(round(rad2μas(curr.sky.m_d),digits=2))", fontsize = 32, color = :white)
		CM.text!(imgax, μas2rad(40.0), μas2rad(30.0); text = "a :$(round(curr.sky.spin,digits=2))", fontsize = 32, color = :white)
		CM.text!(imgax, μas2rad(40.0), μas2rad(20.0); text = "θo :$(round(curr.sky.θo,digits=2))", fontsize = 32, color = :white)
		display(fig)
		m_d = begin
			if :m_d in keys(curr.sky)
				rad2μas.(curr.sky.m_d)
			else
				rad2μas.(sqrt.(curr.sky.m_d_x_rpeak .* curr.sky.m_d_d_rpeak))
			end
		end
		@info "On step $(c.counter) mass=$(m_d) inc=$(curr.sky.θo), spin=$(curr.sky.spin), flux_rat=$(curr.sky.frac), σimg=$(curr.sky.σimg), pa=$(unsafe_trunc(Int, curr.sky.pa*180/π)), θs=$(curr.sky.θs), p2=$(curr.sky.p2)"
		return false
	else
		return false
	end
end


xopt = ((rng, post) -> begin
	temp = prior_sample(rng, post)
	truth = (m_d = μas2rad(4.0), spin = 0.5, θo = 163.0, θs = 80.0, σimg = 1e-1, p1 = 2.0, p2 = 2.0, rpeak = 3.5, χ = π/1.5, ι = π/4, βv = 0.6, spec = 1.0, η = π/2, pa = π-72*π/180, ν = 1.0, ρpr = 2.0)#,frac = 0.1)
	for i in keys(truth)
		if i in keys(temp.sky)
			@reset temp.sky[i] = truth[i]
		end
	end
	#@reset temp.sky.σimg = 0.10
	return temp
end)(rng, post)


prior_sample(rng, post)
xopt=transform(fpost, Comrade.inverse(fpost, xopt))#chain[end]))

m = skymodel(post, xopt)
newgrid = imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx())
ComradeBase.intensitymap(m, newgrid)
img = imageviz(ComradeBase.intensitymap(m, newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))
display(img)
imageviz(ComradeBase.intensitymap(m, newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))

lr = 0.05
xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(lr); initial_params = xopt, maxiters = 2_000, g_tol = 1e-1, callback = Callback(100, ()->nothing))

using Pathfinder
result = pathfinder(fpost; init = Comrade.inverse(fpost, xopt))#, ndraws_elbo = 6, ntries = 2_000)
inv_metric = result.fit_distribution_transformed.Σ
init_params = result.draws[:, 1]
transform(fpost, init_params)

using AdvancedHMC
metric = DiagEuclideanMetric(diag(inv_metric))
integrator = Leapfrog(0.01)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator);
	init_buffer = 100, term_buffer = 200)
smplr = HMCSampler(kernel, metric, adaptor)
out=joinpath(dirname(@__DIR__), "Results_non_diagonal_metric_grmhd_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
println("starting")
#chain = sample(rng, post, NUTS(0.9), 60_000; n_adapts = 10_000, progress = true, saveto = DiskStore(mkpath(out), 10), initial_params = transform(fpost, init_params))#, restart = true)
chain = sample(rng, post, smplr, 120_000; n_adapts = 500, progress = true, saveto = DiskStore(mkpath(out), 10), initial_params = transform(fpost, init_params), restart = true)

using Serialization
run_name=joinpath(dirname(@__DIR__), "Results_non_diagonal_metric_grmhd_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
out=joinpath(dirname(@__DIR__), run_name)
chain = begin
	temp = load_samples(joinpath(dirname(@__DIR__), run_name))#[10_000:end]
	temp
	#temp[Int(length(temp)/2):end]
end
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

CM.set_theme!(CM.theme_dark())

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

inimg.total_flux()

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

using StatsBase
using LinearAlgebra
fchain_snippet = fchain#[end-200:end]
#fchain_snippet = fchain[begin:100]
covmat = StatsBase.cov(fchain_snippet)
cormat = cov2cor(covmat)
datamat = reduce(hcat, fchain_snippet)'
svdfchain = svd(datamat)
svdfchain.U*Diagonal(svdfchain.S)*svdfchain.Vt
#
begin
	fig = CM.Figure();
	CM.plot!(CM.Axis(fig[1, 1], title = "Singular Values of chain"), svdfchain.S)
	fig
end
#
datamat_test = reduce(hcat, (randn(dimension(fpost)) for i in 1:60))'
svdtest = svd(datamat_test)
pctest = svdtest.U * Diagonal(svdtest.S)
begin
	fig = CM.Figure();
	CM.plot!(CM.Axis(fig[1, 1], title = "Singular Values of chain"), svdtest.S)
	fig
end

begin
	fig = CM.Figure(size = (700, 700));
	CM.plot!(CM.Axis(fig[1, 1], title = "1"), svdfchain.V[:, 1])
	CM.plot!(CM.Axis(fig[1, 2], title = "2"), svdfchain.V[:, 2])
	CM.plot!(CM.Axis(fig[1, 3], title = "3"), svdfchain.V[:, 3])
	CM.plot!(CM.Axis(fig[2, 1], title = "4"), svdfchain.V[:, 4])
	CM.plot!(CM.Axis(fig[2, 2], title = "5"), svdfchain.V[:, 5])
	CM.plot!(CM.Axis(fig[2, 3], title = "6"), svdfchain.V[:, 6])
	CM.plot!(CM.Axis(fig[3, 1], title = "7"), svdfchain.V[:, 7])
	CM.plot!(CM.Axis(fig[3, 2], title = "8"), svdfchain.V[:, 8])
	CM.plot!(CM.Axis(fig[3, 3], title = "9"), svdfchain.V[:, 9])
	fig
end
pc1 = svdfchain.V[:, 1]
svdfchain.V[:, 1] |> norm
svdfchain.V[:, 1]' * svdfchain.V[:, 2]
pc1 = normalize(svdfchain.U*Diagonal(svdfchain.S)*svdfchain.Vt[:, 1])
pc2 = normalize(svdfchain.U*Diagonal(svdfchain.S)*svdfchain.Vt[:, 2])
pc1' * pc2

#cmatindx = findmax(abs.(cormat- diagm(ones(108))))
#cormat[begin:100, begin:100] |> CM.heatmap
#CM.scatter(([i[cmatindx[2][1]] for i in fchain_snippet]),([i[cmatindx[2][2]] for i in fchain_snippet]), alpha=0.5)
#CM.scatter(([i[1] for i in fchain_snippet]),([i[cmatindx[2][1]] for i in fchain_snippet]), alpha=0.5)
#CM.scatter(([i[2] for i in fchain_snippet]),([i[cmatindx[2][1]] for i in fchain_snippet]), alpha=0.5)
#CM.scatter(([i[3] for i in fchain_snippet]),([i[cmatindx[2][1]] for i in fchain_snippet]), alpha=0.5)
#CM.scatter(([i[4] for i in fchain_snippet]),([i[cmatindx[2][1]] for i in fchain_snippet]), alpha=0.5)
#CM.scatter(([i[5] for i in fchain_snippet]),([i[cmatindx[2][1]] for i in fchain_snippet]), alpha=0.5)
#CM.scatter(([i[6] for i in fchain_snippet]),([i[cmatindx[2][1]] for i in fchain_snippet]), alpha=0.5)
#CM.scatter(([i[7] for i in fchain_snippet]),([i[cmatindx[2][1]] for i in fchain_snippet]), alpha=0.5)
#CM.scatter(([i[8] for i in fchain_snippet]),([i[cmatindx[2][1]] for i in fchain_snippet]), alpha=0.5)
#CM.lines([i[1] for i in fchain_snippet])
#CM.lines([i[cmatindx[2][1]] for i in fchain_snippet])
#CM.lines([i[cmatindx[2][2]] for i in fchain_snippet])
#CM.lines(chain[end-100:end].sky.spin)


fpaindx = 1
fpa = [i[fpaindx] for i in fchain]
pmx = findmax((i == fpaindx ? 0.0 : abs(cor(fpa, [j[i] for j in fchain])) for i in 1:dimension(post)))

CM.scatter([i[pmx[2]] for i in fchain], [i[fpaindx] for i in fchain], alpha = 0.5)

newgrid = imagepixels(fovx, fovy, 2*npix, 2*npix; executor = ThreadsEx())
#imgs = intensitymap.(msamples, Ref(skym.grid))

imgs = intensitymap.(msamples, Ref(newgrid))#skym.grid))
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

intensitymap(skymodel(post, chain[1000]), newgrid) |> imageviz
intensitymap(skymodel(post, chain[end]), newgrid) |> imageviz

newgrid = imagepixels(fovx, fovy, 2npix, 2npix; executor = ThreadsEx())
fig = CM.Figure()
ax1 = CM.Axis(fig[1, 1], title = "log density")
ax2 = CM.Axis(fig[2, 1], title = "step size", yscale = log10)
ax3 = CM.Axis(fig[1:2, 2], xreversed = true, yticklabelsvisible = false, xticklabelsvisible = false, aspect = 1)

for i in 250:1:310
	CM.empty!(ax1)
	CM.empty!(ax2)
	CM.empty!(ax3)
	CM.lines!(ax1, stats.log_density[begin:i])
	CM.lines!(ax2, stats.step_size[begin:i])
	CM.image!(ax3, intensitymap(skymodel(post, chain[i]), newgrid), colormap = :afmhot)
	sleep(0.1)
	display(fig)
end
temp = chain[end-100]
@reset temp.sky.σimg = 0.01
intensitymap(skymodel(post, temp), newgrid) |> imageviz
intensitymap(smoothed(skymodel(post, temp), μas2rad(20.0)/(2√2*log(2))), newgrid) |> imageviz


c = chain.sky[(end-500):end]
mt1 = mean(skym.metadata.transform1.(c.c1, c.ρpr, c.νpr))
t1 = CM.image(mt1)
CM.Colorbar(t1.figure[1, 2], t1.plot)
t1

st1 = std(skym.metadata.transform1.(c.c1, c.ρpr, c.νpr))
t2 = CM.image(st1)
CM.Colorbar(t2.figure[1, 2], t2.plot)
t2


for i in 5_000:100:length(chain)
	#img = imageviz(intensitymap(smoothed(skymodel(post, chain[i]), μas2rad(7/(2*√(2*log(2))))), skym.grid), size=(500, 400))
	img = imageviz(intensitymap(skymodel(post, chain[i]), newgrid), size = (500, 400))
	println(i)
	sleep(0.1)
	display(img)
end

using PairPlots
using LaTeXStrings

postsamps = Comrade.postsamples(chain[5_000:end])
ks = [:m_d, :spin, :θo, :rpeak, :χ, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]
mat = reduce(hcat, getproperty.(Ref(postsamps.sky), ks))#[2000:end,:]
table = NamedTuple{Tuple(ks)}([mat[:, i] for i in 1:length(ks)])

pairplot(
	#reduce(hcat,getproperty.(Ref(postsamps.sky), [:m_d, :spin, :θo,:rpeak, :χ, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]))=>
	table =>
		(PairPlots.Scatter(), PairPlots.MarginHist()),
	#labels=[:m_d, :spin, :θo,:rpeak, :χ, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]
	labels = Dict(:m_d => L"\theta_g", :spin => L"a", :θo => L"\theta_o", :pa => L"p.a.", :rpeak => L"R", :θs => L"\theta_s"),
)


model = skymodel(post, chain[end])
model.model.model.scene[1].material

b = sqrt(27)

q = [2*b^2, 0.0]
p = [-b^2, 0.0]

pow(x, y) = x^y

function c_m(x, y)
	return [x[1] * y[1] - x[2] * y[2], x[1] * y[2] + x[2] * y[1]]
end

function c_d(x, y)
	return c_m(x, [y[1], -y[2]]) / (pow(y[1], 2.0) + pow(y[2], 2.0));
end
function c_pow(x, y)
	mag = pow(sqrt(x[1] * x[1] + x[2] * x[2]), y)
	angle = y * (atan(x[2], x[1]))
	return [mag * cos(angle), mag * sin(angle)]
end
C = c_pow([q[1]*q[1], 0.0] / 4.0 + [p[1]*p[1]*p[1], 0.0] / 27.0, 0.5);
C1 = c_pow(-q / 2.0 + C, 1.0 / 3.0)
C2 = c_m(C1, [-1/2.0, sqrt(3)/2.0])#c_pow(-q / 2. + c_m(C, [-(1. / 2.), sqrt(3)]/ 2.), 1. / 3.);
C3 = c_m(C1, [-1/2.0, -sqrt(3)/2.0])#c_pow(-q / 2. + c_m(C, [-(1. / 2.), -sqrt(3)]/ 2.), 1. / 3.);

v1 = C2 - c_d(p, 3.0 * C2)
v3 = C3 - c_d(p, 3.0 * C3)
v4 = C1 - c_d(p, 3.0 * C1)

u1 = c_pow(-q / 2.0 + c_pow(c_pow(q, 2.0) / 4.0 + c_pow(p, 3.0) / 27.0, 0.5), 1.0 / 3.0);
u2 = c_pow(-q / 2.0 - c_pow(c_pow(q, 2.0) / 4.0 + c_pow(p, 3.0) / 27.0, 0.5), 1.0 / 3.0);

e1 = [-(1.0), sqrt(3)]
e2 = [-(1.0), -sqrt(3)]
c_m(e1, u1)/2.0 + c_m(e2, u2)/2.0
c_m(e2, u1)/2.0 + c_m(e1, u2)/2.0
u1 + u2


