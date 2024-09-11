using Pkg;Pkg.activate(dirname(@__DIR__));
using Comrade
using Pyehtim
using Krang
include(joinpath((@__DIR__),"modifiers.jl"))
include(joinpath((@__DIR__),"models","KerrGMRF.jl"))
include(joinpath((@__DIR__),"models","JuKeBOX.jl"))

seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.01
bulkx = 40.0
bulky = 40.0
npix = 16
bulkpix = 60
fovx = μas2rad(100.0)
fovy = μas2rad(100.0)

inobs = ehtim.obsdata.load_uvfits(joinpath(dirname(@__DIR__), "data", "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits"))
grid = imagepixels(fovx, fovy, npix, npix; executor=Serial(), header=ComradeBase.MinimalHeader(string(inobs.source), pyconvert.(Ref(Float64), [inobs.ra, inobs.dec, inobs.mjd, inobs.rf])...))

truth = (m_d = μas2rad(5.01), spin = -0.94, θo = 30.0, θs = 70.0, rpeak = 4.7, p1 = 0.1, p2 = 3.7, χ = 2.5, ι = 1.6, βv = 0.2, spec = 0.09, η = -2.95)
ModifiedJuKeBOX(θ) = RenormalizedFlux(modify(JuKeBOX(θ), Stretch( θ.m_d, θ.m_d)), 0.6)
mdl = ModifiedJuKeBOX(truth)
true_intmap = intensitymap(mdl, imagepixels(fovx, fovy, 400, 400; executor=ThreadsEx()))
Comrade.save_fits(joinpath(dirname(@__DIR__), "data", "true_intmap_jukebox.fits"), true_intmap)
inimg = ehtim.image.load_fits(joinpath(dirname(@__DIR__), "data", "true_intmap_jukebox.fits"))
inimg.rf = inobs.rf
inimg.dec = inobs.dec
inimg.ra = inobs.ra

obs = inimg.observe_same(inobs, ampcal=ampcal, phasecal=phasecal, add_th_noise=add_th_noise, seed=seed, ttype="fast")
obs = scan_average(obs.flag_uvdist(uv_min=0.1e9))
obs = obs.add_fractional_noise(fractional_noise)

dvis = extract_table(obs, Visibilities())
dvisamp = extract_table(obs, VisibilityAmplitudes())
dcphase = extract_table(obs, ClosurePhases())
dlcamp = extract_table(obs, LogClosureAmplitudes())

import CairoMakie as CM
using VLBIImagePriors
using Distributions, DistributionsAD

true_intmap |> imageviz

# # Define model to fit
function ModifiedKerrGMRF(θin, meta) 
#function ModifiedKerrGMRF(θ, meta) 
    (;θ) = meta
    params = (;m_d = θin.m_d, spin = θ.spin, θo = θ.θo, θs = θ.θs, rpeak = θin.rpeak, p1 = θin.p1, p2 = θin.p2, χ = θ.χ, ι = θ.ι, βv = θ.βv, spec = θ.spec, η = θ.η, c = θin.c, σimg = θin.σimg)
    #RenormalizedFlux(modify(KerrGMRF(θ, meta), Stretch(θ.m_d, θ.m_d)),0.6)
    RenormalizedFlux(modify(KerrGMRF(params, meta), Stretch(θin.m_d, θin.m_d)),0.6)
end
bulkgrid = imagepixels(bulkx, bulky, bulkpix, bulkpix, executor=ThreadsEx())
grid = imagepixels(fovx, fovy, npix, npix; executor=Serial())#ThreadsEx(:Enzyme))

bulkimg = IntensityMap(zeros(Float64, bulkpix, bulkpix), bulkgrid)
skymeta = (;ftot = 1.0, bulkimg = bulkimg, screengrid=grid, npix=npix, θ=truth)

beam = 2 # the resolution of the EHT image is roughly the shadow size
rat = (beam/(step(bulkgrid.X)))
crcache = ConditionalMarkov(GMRF, bulkgrid; order=2)
cprior = HierarchicalPrior(crcache, truncated(InverseGamma(1.0, -log(0.1)*rat); upper=2*bulkpix))
prior = (
    rpeak = Uniform(1.0, 8.0),
    p1 = Uniform(0.1, 10.0),
    p2 = Uniform(0.1, 10.0),
    m_d = Uniform(μas2rad(2.0), μas2rad(8.0)), 
    #spin = Uniform(-0.99, 0.01),
    #θo = Uniform(1, 40.0),
    #θs = Uniform(50, 90),
    #χ = Uniform(-π/2, π/2),
    #ι = Uniform(0.0, π/2),
    #βv = Uniform(0.01, 0.99),
    #spec = Uniform(0.0, 3.0),
    #η = Uniform(0.0, π),
    c = cprior,
    σimg = Exponential(0.5)
)

skym = SkyModel(ModifiedKerrGMRF, prior, grid; metadata=skymeta)
G = SingleStokesGain() do x
    lg = x.lg
    gp = x.gp
    return exp(lg + 1im*gp)
end

intpr = (
    lg= ArrayPrior(IIDSitePrior(ScanSeg(), Normal(0.0, 0.2)); LM = IIDSitePrior(ScanSeg(), Normal(0.0, 1.0))),
    gp= ArrayPrior(IIDSitePrior(ScanSeg(), DiagonalVonMises(0.0, inv(π^2))); refant=SEFDReference(0.0), phase=true)
)
intmodel = InstrumentModel(G, intpr)

post = VLBIPosterior(skym, intmodel, dvis)

@time begin
    g = imagepixels(fovx, fovy, npix, npix)
    p_sample = prior_sample(post)
    mdl = skymodel(post, p_sample)
    function temp(mdl, g)  
        for _ in 1:1000
           intensitymap(mdl, g)
        end
    end
    mdl.model.model.metadata.bulkimg|> imageviz
    fig = CM.Figure();
    using BenchmarkTools
    impix = imagepixels(fovx, fovy, 10*npix, 10*npix)
    img = intensitymap(mdl, impix)
    img |> imageviz
end

using Optimization
using OptimizationOptimisers
#using Zygote
using Enzyme
using StableRNGs
import Plots 
Enzyme.Compiler.RunAttributor[] = false

rng = StableRNG(1234)
msamples = []
using AdvancedHMC
psample = prior_sample(rng, post)
imageviz(intensitymap(skymodel(post, psample), g), size=(500, 400))

mutable struct Callback 
    counter::Int
    stride::Int
    const f::Function
end
Callback(stride, f) = Callback(0, stride, f)
function (c::Callback)(state, loss, others...)
    c.counter += 1  
    if c.counter % c.stride == 0
        @info "On step $(c.counter) loss = $(loss)"
        return false
    else
        return false
    end
end

xopt, sol = comrade_opt(post, Optimisers.Adam(), Optimization.AutoEnzyme(); initial_params=psample, maxiters=50_000, g_tol=1e-1, callback=Callback(10,()->nothing))

img = intensitymap(skymodel(post, xopt), imagepixels(fovx, fovy, 10*npix, 10*npix))
imageviz(img, size=(500, 400))
residual(post, xopt)
xopt.sky.m_d |> rad2μas
dvis_mdl = deepcopy(dvisamp)

gfour = FourierDualDomain(axisdims(img), arrayconfig(dvisamp), NFFTAlg())
dvis_mdl = deepcopy(dvisamp)
dvis_mdl.measurement .= abs.(visibilitymap(mdl, gfour) )
Plots.plot(dvis_mdl);
Plots.plot!(dvisamp)

dvis_mdl.measurement .= abs.(visibilitymap(skymodel(post, xopt), gfour) )
Plots.plot(dvis_mdl);
skymodel(post,xopt).model.model.metadata.bulkimg |> imageviz
xopt.sky.m_d |> rad2μas