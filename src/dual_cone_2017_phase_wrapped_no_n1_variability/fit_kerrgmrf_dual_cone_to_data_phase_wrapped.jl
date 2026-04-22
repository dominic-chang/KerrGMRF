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
using Pathfinder
using Plots: Plots
using AdvancedHMC
LinearAlgebra.BLAS.set_num_threads(24) # to avoid threading conflicts with FINUFFT
rng = StableRNG(1234)

include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF_no_n1_variability.jl"))

function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa*π/180), Shift(μas2rad(10.0), μas2rad(0.0)))
	return RenormalizedFlux(m, θ2.f)
end

function (c::Callback)(state, loss, others...)
	fpost = c.fpost
	post = fpost.lpost
	loss_arr = c.loss_arr
	grid = fpost.lpost.skymodel.grid
	curr = transform(fpost, state.u)
	c.counter += 1
	fig = CM.Figure(; size = (1200, 600));
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
		_imgviz!(fig, imgax, intensitymap(skymodel(post, curr), grid))
		CM.text!(imgax, μas2rad(50.0), μas2rad(50.0); text = "M/D :$(round(rad2μas(curr.sky.m_d),digits=2))", fontsize = 30, color = :white)
		CM.text!(imgax, μas2rad(50.0), μas2rad(40.0); text = "θo :$(Int(floor(curr.sky.θo)))", fontsize = 30, color = :white)
		CM.text!(imgax, μas2rad(50.0), μas2rad(30.0); text = "spin :$(round(curr.sky.spin,digits=2))", fontsize = 30, color = :white)
		CM.text!(imgax, μas2rad(50.0), μas2rad(20.0); text = "pa :$(Int(floor(curr.sky.pa)))", fontsize = 30, color = :white)

		display(fig)
		m_d = begin
			if :m_d in keys(curr.sky)
				rad2μas.(curr.sky.m_d)
			else
				rad2μas.(sqrt.(curr.sky.m_d_x_rpeak .* curr.sky.m_d_d_rpeak))
			end
		end
		@info "On step $(c.counter) mass=$(m_d) inc=$(curr.sky.θo), spin=$(curr.sky.spin), flux_rat=$(curr.sky.frac), σimg=$(curr.sky.σimg), pa=$(unsafe_trunc(Int, curr.sky.pa)), θs=$(curr.sky.θs), p2=$(curr.sky.p2)"
		return false
	else
		return false
	end
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
bulkpix = 60
raster_size = 150.0 # in microarcseconds    
snrcut = 3.0
uv_min = 0.1e9
fovx = μas2rad(130.0)
fovy = μas2rad(130.0)
npix = 30
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
χ = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
pa =VLBIImagePriors.DeltaDist(108.0),
ι = Uniform(0, π/2),
βv = Uniform(0.01, 0.99),
spec = Uniform(-1.0, 5.0),
η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
frac = VLBIImagePriors.DeltaDist(1.0), #pa =VLBIImagePriors.DeltaDist(π-72*π/180),# Uniform(-π, 0),
f = VLBIImagePriors.DeltaDist(1.0),
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
metadata = (; bulkgrid, transform1, transform2, raster_size, offset)#algorithm = FINUFFTAlg(; threads = 1)
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode = set_runtime_activity(Enzyme.Reverse))
fpost = asflat(post)

xopt =
	curr =
		tsol = (
			sky = (
				m_d = μas2rad(3.83),
				spin = 0.34261943710501536,
				θo = 159.66325910346768,
				θs = 72.55221410100953,
				rpeak = 1.0320609725554761,
				p1 = 4.997485810130196,
				p2 = 3.4271109236512496,
				χ = 0.784485331711496,
				ι = 1.4672026597904312,
				βv = 0.3511992071422593,
				spec = 2.745807873295397,
				η = -0.2979777845654403,
				#pa = 108.0,
			),
		)
xvals = ((post) -> begin
	temp = prior_sample(post)#transform(fpost, prior_sample(fpost))
	newvals = []
	for key in keys(temp.sky)
		if key in keys(curr.sky)
			push!(newvals, getproperty(curr.sky, key))
		else
			push!(newvals, getproperty(temp.sky, key))
		end
	end
	outvals = (sky = NamedTuple{keys(temp.sky)}(newvals),)
	@reset outvals.sky.σimg = 1e-1
	return outvals
end)(post)
m = skymodel(post, xvals)
imageviz(ComradeBase.intensitymap(m, skym.grid))

let curr = curr, xopt=xopt,xvals=xvals, lr=lr, fpost=fpost, post=post
	curr = (sky = NamedTuple{keys(curr.sky)}(xopt.sky[keys(curr.sky)]),)
	xvals = ((post) -> begin
		temp = prior_sample(post)
		newvals = []
		for key in keys(temp.sky)
			if key in keys(curr.sky)
				push!(newvals, getproperty(curr.sky, key))
			else
				push!(newvals, getproperty(temp.sky, key))
			end
		end
		outvals = (sky = NamedTuple{keys(temp.sky)}(newvals),)
		@reset outvals.sky.σimg = 1e0
		return outvals
	end)(post)
	xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(lr); initial_params = xvals, maxiters = 100, g_tol = 1e-1, callback = Callback(20, fpost, ()->nothing))
end

curr = (sky = NamedTuple{keys(curr.sky)}(xopt.sky[keys(curr.sky)]),)
xvals = ((post) -> begin
	temp = prior_sample(post)
	newvals = []
	for key in keys(temp.sky)
		if key in keys(curr.sky)
			push!(newvals, getproperty(curr.sky, key))
		else
			push!(newvals, getproperty(temp.sky, key))
		end
	end
	outvals = (sky = NamedTuple{keys(temp.sky)}(newvals),)
	@reset outvals.sky.σimg = 5e-1
	return outvals
end)(post)
xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(lr); initial_params = xvals, maxiters = 200, g_tol = 1e-1, callback = Callback(20, fpost, ()->nothing))

Comrade.residual(post, xopt)
imageviz(ComradeBase.intensitymap(skymodel(post, xopt), skym.grid))#,colorscale=log, colorrange=(1e-10, 1e-2))

# initialize with pathfinder
result = pathfinder(fpost; init = Comrade.inverse(fpost, xopt), ndraws_elbo = 50, ntries = 5_000)
inv_metric = result.fit_distribution_transformed.Σ
init_params = result.draws[:, 1]
transform(fpost, init_params)
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost, init_params)), skym.grid))
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost, init_params)), skym.grid), colorscale = log, colorrange = (1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, transform(fpost, init_params)), μas2rad(20/(2.355))), skym.grid), colormap = :afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, transform(fpost, init_params)), μas2rad(5/(2.355))), skym.grid), colormap = :afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))


# Run HMC
#metric = DenseEuclideanMetric(Matrix(inv_metric))
metric = DiagEuclideanMetric(diag(inv_metric))
integrator = Leapfrog(0.01)#find_good_stepsize(Hamiltonian(metric, fpost), init_params))
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator);
	init_buffer = 100, term_buffer = 200)
smplr = HMCSampler(kernel, metric, adaptor)
out=joinpath((@__DIR__), "Results_non_diagonal_metric_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$(year)_phase_wrapped")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
println("starting")

#sample(rng, post, smplr, 120_000; n_adapts=15000, initial_params=transform(fpost, init_params), progress=true)
chain = sample(rng, post, smplr, 120_000; n_adapts = 15_000, saveto = DiskStore(mkpath(out), 10), initial_params = transform(fpost, init_params), restart = true)
