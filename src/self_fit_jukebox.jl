using Pkg;Pkg.activate(dirname(@__DIR__));
using Comrade
using Pyehtim
using Krang
using Enzyme
using Metal
using KernelAbstractions
Enzyme.Compiler.RunAttributor[] = false

include(joinpath((@__DIR__),"modifiers.jl"))
include(joinpath((@__DIR__),"models","JuKeBOX.jl"))
Enzyme.Compiler.RunAttributor[] = false

seed = 1234
phasecal = true
ampcal = true
add_th_noise = false# true
scan_avg = true
fractional_noise = 0.01
npix = 100
fovx = μas2rad(80.0)
fovy = μas2rad(80.0)

inobs = ehtim.obsdata.load_uvfits(joinpath(dirname(@__DIR__), "data", "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits"))
joinpath(dirname(@__DIR__), "data", "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits") |> isfile
grid = imagepixels(fovx, fovy, npix, npix; executor=Serial(), header=ComradeBase.MinimalHeader(string(inobs.source), pyconvert.(Ref(Float64), [inobs.ra, inobs.dec, inobs.mjd, inobs.rf])...))

#truth = (m_d = μas2rad(5.01), spin = -0.94, θo = 30.0, θs = 70.0, rpeak = 4.7, p1 = 0.1, p2 = 3.7, χ = 2.5, ι = 1.6, βv = 0.2, spec = 0.09, η = -2.95)
truth = (m_d = 5.01, spin = -0.94, θo = 30.0, θs = 70.0, rpeak = 4.7, p1 = 0.5, p2 = 3.7, χ = 2.5, ι = 1.6, βv = 0.2, spec = 0.09, η = -2.95)
ModifiedJuKeBOX(θ, meta) = RenormalizedFlux(modify(JuKeBOX(θ), Stretch(μas2rad(θ.m_d), μas2rad(θ.m_d))), 0.6)
mdl = ModifiedJuKeBOX(truth,nothing)
true_intmap = intensitymap(mdl, imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx()))#MetalBackend()))
Comrade.save_fits(joinpath(dirname(@__DIR__), "data", "true_intmap_jukebox.fits"), true_intmap)
inimg = ehtim.image.load_fits(joinpath(dirname(@__DIR__), "data", "true_intmap_jukebox.fits"))
inimg.display()
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

function ModifiedKerrGMRF(θ, meta) 
    RenormalizedFlux(modify(KerrGMRF(θ, meta), Stretch(μas2rad(θ.m_d), μas2rad(θ.m_d))),0.6)
end

grid = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())#ThreadsEx(:Enzyme))
skymeta = (;npix=npix, θ=truth)
prior = (
    m_d = Uniform(2.0, 8.0), 
    spin = Uniform(-0.99, -0.01),
    θo = Uniform(1, 40.0),
    θs = Uniform(50, 90),
    rpeak = Uniform(1.0, 8.0),
    p1 = Uniform(0.1, 5.0),
    p2 = Uniform(0.1, 5.0),
    χ = Uniform(-π, π),
    ι = Uniform(0.0, π),
    βv = Uniform(0.01, 0.99),
    spec = Uniform(0.0, 3.0),
    η = Uniform(-π, π),
)
skym = SkyModel(ModifiedJuKeBOX, prior, grid; metadata=skymeta)

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

post = VLBIPosterior(skym, dcphase, dlcamp; admode=set_runtime_activity(Enzyme.Reverse))

impix = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
start = (sky=NamedTuple{keys(truth)}(values(truth)) ,)
img = intensitymap(mdl, impix)

img |> imageviz |> display  

gfour = FourierDualDomain(grid, arrayconfig(dvisamp), NFFTAlg())
dvis_mdl = deepcopy(dvisamp)
dvis_mdl.measurement .= abs.(visibilitymap(mdl, gfour) )
import Plots
Plots.plot(dvis_mdl, label="model");
Plots.plot!(dvisamp)

using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Enzyme
using StableRNGs
import Plots 

rng = StableRNG(1234)
using AdvancedHMC
psample = prior_sample(rng, post)
imageviz(intensitymap(skymodel(post, psample), grid), size=(500, 400))

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

start = (sky=NamedTuple{keys(truth)}(values(truth)) ,)
post(psample)

xopt = prior_sample(rng, post)
imageviz(intensitymap(skymodel(post, psample), grid), size=(500, 400))

xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(0.1); initial_params=xopt, maxiters=4_00, g_tol=1e-1, callback=Callback(10,()->nothing))

img = intensitymap(skymodel(post, xopt), imagepixels(fovx, fovy, npix, npix))
imageviz(img, size=(500, 400))
imageviz(true_intmap, size=(500, 400))
imageviz(((img .- true_intmap)), size=(500, 400))