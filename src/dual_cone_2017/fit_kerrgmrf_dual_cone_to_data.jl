using Pkg;Pkg.activate(dirname(dirname(@__DIR__)));
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
using Pathfinder
using AdvancedHMC
using Optimization, OptimizationOptimisers
using Enzyme
using StableRNGs
import Plots 
using LaTeXStrings

LinearAlgebra.BLAS.set_num_threads(48) # to avoid threading conflicts with FINUFFT
rng = StableRNG(1234)

include(joinpath(dirname(@__DIR__),"utils.jl"))
include(joinpath(dirname(@__DIR__),"plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__),"modifiers.jl"))
include(joinpath(dirname(@__DIR__),"models","KerrGMRF_shared_variability.jl"))

function ModifiedKerrGMRF(θ2, meta) 
    m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa), Shift(μas2rad(12.0), μas2rad(12.0)))
    return m
end

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
raster_size = 180.0# 90 # in microarcseconds    
snrcut = 3.0
uv_min = 0.1e9
fovx = μas2rad(150.0)
fovy = μas2rad(150.0)
npix = 40
year = 2017

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 =>"L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(dirname(@__DIR__)), "data", data[year])

# Get observer information
obs = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min=0.1e9).add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))

bulkgrid = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor=ThreadsEx())
transform1, cprior1  = matern(size(bulkgrid))
transform2, cprior2  = matern(size(bulkgrid))
prior = (
    m_d = Uniform(μas2rad(1.0), μas2rad(8.0)), 
    spin = Uniform(0.01, 0.99),
    θo = Uniform(120.0, 179.0),
    θs = Uniform(40.0, 90.0),
    rpeak = Uniform(1.0, 8.0),
    p1 = Uniform(0.1, 5.0),
    p2 = Uniform(1.0, 5.0),
    χ = Uniform(-π, π),
    ι = Uniform(0, π/2),
    βv = Uniform(0.01, 0.99),
    spec = Uniform(-1.0, 5.0),
    η = Uniform(-π, π),
    frac = VLBIImagePriors.DeltaDist(1.0),
    pa =VLBIImagePriors.DeltaDist(π-72*π/180),
    f = VLBIImagePriors.DeltaDist(0.6),
    σimg = truncated(Normal(0.0, 1.0); lower=0.0),
    ρpr = truncated(InverseGamma(1.0, -log(0.1)*10); lower = 1.0, upper=2*max(size(bulkgrid)...)),
    νpr = Uniform(1.0, 5.0),
    c1 = cprior1,
    c2 = cprior2,
)
offset=0.0
skym = SkyModel(
    ModifiedKerrGMRF, 
    prior, 
    imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx()); 
    metadata=(;bulkgrid, transform1, transform2, raster_size, offset)
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode=set_runtime_activity(Enzyme.Reverse))
fpost = asflat(post)

psample = prior_sample(post)
Comrade.transform(fpost, rand(dimension(fpost)))

vals = rand(dimension(fpost))
dvals = zeros(size(vals))

xopt = ((rng, post) -> begin 
    temp = prior_sample(rng, post)
    truth = temp 
    truth = (m_d = μas2rad(4.0), spin = 0.5, θo = 163.0, σimg = 1e-1, rpeak= 3.0, p1 =2.0, p2=2.0, χ = π/1.5, ι = π/4, βv = 0.6, spec = 1.0, η = π/2, pa=π-72*π/180, ν = 2.0, ρpr = 10.0)#,frac = 0.1)

    for i in keys(truth)
        if i in keys(temp.sky)
            @reset temp.sky[i] = truth[i]
        end
    end
    return temp
end)(rng, post)

m = skymodel(post, xopt)
newgrid = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
ComradeBase.intensitymap(m, newgrid)
img = imageviz(ComradeBase.intensitymap(m, newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))
display(img)
imageviz(ComradeBase.intensitymap(m, newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))

xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(lr); initial_params=xopt, maxiters=500, g_tol=1e-1, callback=Callback(10, fpost,()->nothing))

result = pathfinder(fpost; init=Comrade.inverse(fpost,xopt), ndraws_elbo=100, ntries=1_00)
inv_metric = result.fit_distribution_transformed.Σ
init_params = result.draws[:, 1]
transform(fpost,init_params)
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost,init_params)), newgrid))
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost,init_params)), newgrid),colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, transform(fpost, init_params)),μas2rad(20/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, transform(fpost, init_params)),μas2rad(5/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))

integrator = Leapfrog(0.01)
metric = DenseEuclideanMetric(Matrix(inv_metric))
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator); 
                         init_buffer=100, term_buffer=200)
smplr = HMCSampler(kernel, metric, adaptor)
out=joinpath(dirname(@__DIR__),"Results_non_diagonal_metric_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
println("starting")
chain = sample(rng, post, smplr, 120_000; n_adapts=500, saveto=DiskStore(mkpath(out), 10), initial_params=transform(fpost, init_params), restart=true)
