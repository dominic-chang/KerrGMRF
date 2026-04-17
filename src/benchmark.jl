using Pkg;Pkg.activate(dirname(@__DIR__));
using Comrade
using Krang
using Enzyme
using StableRNGs
using Accessors
import CairoMakie as CM
using BasicInterpolators
using FINUFFT
rng = StableRNG(1234)

include(joinpath((@__DIR__),"utils.jl"))
include(joinpath((@__DIR__),"modifiers.jl"))
include(joinpath((@__DIR__),"models","KerrGMRF_equatorial.jl"))
include(joinpath((@__DIR__),"models","JuKeBOX.jl"))

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

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 =>"L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "data/frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(@__DIR__), "data", data[year])

using VLBIImagePriors
using Distributions, DistributionsAD

function ModifiedKerrGMRF(θ2, meta) 
    m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa), Shift(μas2rad(12.0), μas2rad(12.0)))
    return RenormalizedFlux(m, θ2.f) 
    
end
ModifiedJuKeBOX(θ, meta) = RenormalizedFlux(Comrade.modify(JuKeBOX(θ), Stretch((θ.m_d), (θ.m_d)), Rotate(θ.pa), Shift(μas2rad(12.0), μas2rad(12.0))), 0.6)

using Optimization
using OptimizationOptimisers
using Enzyme
using StableRNGs
import Plots 
using LaTeXStrings
using AdvancedHMC
using Serialization
run_name = "Results_non_diagonal_metric_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(rad2μas(fovx)))_fov_year_$year"
#run_name = "Results_non_diagonal_metric_50_rast_low_res"
out=joinpath(dirname(@__DIR__),run_name)
chain = begin
    load_samples(joinpath(dirname(@__DIR__), run_name))[15_000:end]
end
xopt = chain[end]
checkpoints = deserialize(joinpath(out, "checkpoint.jls"))
gppost = checkpoints.pt.c.rf.xform.model.logdensity.lpost
imageviz(ComradeBase.intensitymap(skymodel(gppost, xopt), imagepixels(fovx, fovy, npix*3, npix*3; executor=ThreadsEx())), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))

dlcamp, dcphase = Comrade.simulate_observation(gppost, xopt; add_thermal_noise=false)

prior = (
    m_d = Uniform(μas2rad(2.0), μas2rad(6.0)), 
    spin = Uniform(-0.99, -0.01),#-0.99, -0.01),
    θo = Uniform(1, 40.0),
    θs = VLBIImagePriors.DeltaDist(90.0),
    rpeak = Uniform(1.0, 8.0),
    p1 = Uniform(0.1, 5.0),
    p2 = Uniform(1.0, 5.0),
    χ = Uniform(-π, π),
    #χ = VLBIImagePriors.DeltaDist(-π/2),
    #pa =Uniform(-π, 0),
    ι = Uniform(0, π/2),
    βv = Uniform(0.01, 0.99),
    spec = Uniform(-1.0, 5.0),
    η = Uniform(-π, π),
    frac = VLBIImagePriors.DeltaDist(1.0),#Uniform(0.0, 1.0),
    pa =VLBIImagePriors.DeltaDist(-72*π/180),# Uniform(-π, 0),
    f = VLBIImagePriors.DeltaDist(0.6),#Uniform(-π, π),
)

skym = SkyModel(
    ModifiedJuKeBOX, 
    prior, 
    imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx()),;
    metadata = ()
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode=set_runtime_activity(Enzyme.Reverse))
fpost = asflat(post)
psample = prior_sample(post)
Comrade.transform(fpost, rand(dimension(fpost)))
imageviz(ComradeBase.intensitymap(skymodel(post, chain[end]), imagepixels(fovx, fovy, npix*3, npix*3; executor=ThreadsEx())), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))

slice(nt, inds) = nt[keys(nt)[inds]]
xopt = (sky=slice(xopt.sky, firstindex(xopt.sky):(lastindex(xopt.sky)-4)), )
newgrid = imagepixels(fovx, fovy, npix*3, npix*3; executor=ThreadsEx())
imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, xopt),μas2rad(20/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, xopt),μas2rad(5/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))

using Suppressor
using BenchmarkTools
Krang.isAxisymmetric(material::Krang.ElectronSynchrotronPowerLawIntensity) = true
fpasstime = (@capture_out(@btime Comrade.logdensityof(post, xopt)))[begin:end-1]
rpasstime = (@capture_out(@btime Comrade.logdensityof(gppost, chain[end])))[begin:end-1]

fname = joinpath((@__DIR__), "benchmark_results.csv")
if isfile(fname)
    open(fname, "a") do f
        write(f, "$fpasstime,$rpasstime\n")
    end
else
    open(fname, "a") do f
        write(f, "$fpasstime,$rpasstime\n")
    end
end
close(fname)

#fpasstime = Threads.nthreads()
#rpasstime = Threads.nthreads()
#
#fname = joinpath((@__DIR__), "benchmark_results.csv")
#if isfile(fname)
#    open(fname, "a") do f
#        write(f, "$fpasstime,$rpasstime\n")
#    end
#else
#    open(fname, "a") do f
#        write(f, "$fpasstime,$rpasstime\n")
#    end
#end
#close(fname)