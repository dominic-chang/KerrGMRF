using Pkg;Pkg.activate(dirname(@__DIR__));
using Comrade
using Pyehtim
using Krang
using Enzyme
include(joinpath((@__DIR__),"modifiers.jl"))
include(joinpath((@__DIR__),"models","KerrGMRF.jl"))
include(joinpath((@__DIR__),"models","JuKeBOX.jl"))
Enzyme.Compiler.RunAttributor[] = false

seed = 1234
phasecal = true
ampcal = true
add_th_noise = false# true
scan_avg = true
fractional_noise = 0.01
bulkx = 40.0
bulky = 40.0
npix = 40
bulkpix = 80
fovx = μas2rad(100.0)
fovy = μas2rad(100.0)
# TODO: try NMSE

inobs = ehtim.obsdata.load_uvfits(joinpath(dirname(@__DIR__), "data", "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits"))
grid = imagepixels(fovx, fovy, npix, npix; executor=Serial(), header=ComradeBase.MinimalHeader(string(inobs.source), pyconvert.(Ref(Float64), [inobs.ra, inobs.dec, inobs.mjd, inobs.rf])...))

#truth = (m_d = μas2rad(5.01), spin = -0.94, θo = 30.0, θs = 70.0, rpeak = 4.7, p1 = 0.1, p2 = 3.7, χ = 2.5, ι = 1.6, βv = 0.2, spec = 0.09, η = -2.95)
truth = (m_d = 5.01, spin = -0.94, θo = 30.0, θs = 70.0, rpeak = 4.7, p1 = 0.5, p2 = 3.7, χ = 2.5, ι = 1.6, βv = 0.2, spec = 0.09, η = -2.95)
ModifiedJuKeBOX(θ) = RenormalizedFlux(modify(JuKeBOX(θ), Stretch(μas2rad(θ.m_d), μas2rad(θ.m_d))), 0.6)
mdl = ModifiedJuKeBOX(truth)
true_intmap = intensitymap(mdl, imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx()))
Comrade.save_fits(joinpath(dirname(@__DIR__), "data", "true_intmap_jukebox.fits"), true_intmap)
inimg = ehtim.image.load_fits(joinpath(dirname(@__DIR__), "data", "true_intmap_jukebox.fits"))
inimg.rf = inobs.rf
inimg.dec = inobs.dec
inimg.ra = inobs.ra
inimg.display()

obs = inimg.observe_same(inobs, ampcal=ampcal, phasecal=phasecal, add_th_noise=add_th_noise, seed=seed, ttype="fast")
obs = scan_average(obs)#obs.flag_uvdist(uv_min=0.1e9))
#obs = obs.add_fractional_noise(fractional_noise)

dvis = extract_table(obs, Visibilities())
dvisamp = extract_table(obs, VisibilityAmplitudes())
dcphase = extract_table(obs, ClosurePhases())
dlcamp = extract_table(obs, LogClosureAmplitudes())

import CairoMakie as CM
using VLBIImagePriors
using Distributions, DistributionsAD

true_intmap |> imageviz

# # Define model to fit
function ModifiedKerrGMRF(θ, meta) 
    RenormalizedFlux(modify(KerrGMRF(θ, meta), Stretch(μas2rad(θ.m_d), μas2rad(θ.m_d))),0.6)
end

bulkgrid = imagepixels(bulkx, bulky, bulkpix, bulkpix, executor=ThreadsEx())
grid = imagepixels(fovx, fovy, npix, npix; executor=Serial())#ThreadsEx(:Enzyme))

bulkimg = IntensityMap(zeros(Float64, bulkpix, bulkpix), bulkgrid)
skymeta = (;bulkimg = bulkimg, npix=npix, θ=truth)

cprior = corr_image_prior(bulkgrid, 100.0; base=GMRF, order=1, upper=2*bulkpix)
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
    c = cprior,
    σimg = truncated(Exponential(1.0); upper=1.0)
)

skym = SkyModel(ModifiedKerrGMRF, prior, grid; metadata=skymeta)
#skym = SkyModel(ModifiedJuKeBOXTest, prior, grid; metadata=skymeta)
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

#post = VLBIPosterior(skym, intmodel, dvis; admode=set_runtime_activity(Enzyme.Reverse))
post = VLBIPosterior(skym, dcphase, dlcamp; admode=set_runtime_activity(Enzyme.Reverse))
impix = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
img = intensitymap(mdl, impix)
#[
#begin
#    #g = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
#    p_sample = prior_sample(post)
#    mdl = skymodel(post, p_sample)
#    #function temp(mdl, g)  
#    #    for _ in 1:1000
#    #       intensitymap(mdl, g)
#    #    end
#    #end
#    #mdl.model.model.metadata.bulkimg|> imageviz
#    println("Posterior evaluation")
#    #@time post(p_sample)  
#    fig = CM.Figure();
#
#    println("Intensitymap evaluation")
#    #impix = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
#    @time intensitymap!(img, mdl)
#    img |> imageviz |> display  
#end
#for _ in 1:100
#]
intensitymap(mdl, imagepixels(fovx,fovy, npix,npix)) |> flux

#@profview [post(p_sample)  for i in 1:1000]
#@profview post(p_sample)
p_sample = prior_sample(post)
@time chi2(post, p_sample) ./ (length(dcphase), length(dlcamp))
p_sample.sky.σimg
p_sample.sky.c.params
gfour = FourierDualDomain(grid, arrayconfig(dvisamp), NFFTAlg())
dvis_mdl = deepcopy(dvisamp)
dvis_mdl.measurement .= abs.(visibilitymap(skymodel(post, p_sample), gfour) )
import Plots
Plots.plot(dvis_mdl, label="model");
Plots.plot!(dvisamp)

@time intensitymap(mdl, impix)

using Optimization
using OptimizationOptimisers
using Enzyme
using StableRNGs
import Plots 

rng = StableRNG(1234)
using AdvancedHMC
psample = prior_sample(rng, post)
imageviz(intensitymap(skymodel(post, psample), grid), size=(500, 400))

loss_arr = []
gfour = FourierDualDomain(grid, arrayconfig(dvisamp), NFFTAlg())
dvis_mdl = deepcopy(dvisamp)
mutable struct Callback 
    counter::Int
    stride::Int
    const f::Function
end
Callback(stride, f) = Callback(0, stride, f)
fpost = asflat(post)
function (c::Callback)(state, loss, others...)
    global loss_arr
    global fpost
    global dvis_mdl
    c.counter += 1  
    if c.counter % c.stride == 0
        append!(loss_arr, loss)
        #print(sol)

        curr = transform(fpost,state.u)
        dvis_mdl.measurement .= abs.(visibilitymap(skymodel(post, curr), gfour) )
        #dvis_mdl.measurement .= Comrade.closure_phases(measurement(dvis_mdl), Comrade.designmat(arrayconfig(dvis_mdl)))
        plt = Plots.plot(dvis_mdl, label="model");
        Plots.plot!(dvisamp)
        img = Plots.plot(intensitymap(skymodel(post, curr), grid), size=(500, 400))
        true_img = Plots.plot(true_intmap, label="true")

        display(Plots.plot(plt, img, true_img, size=(1000, 800)))   
        @info "On step $(c.counter) loss = $(loss) mass = $(curr.sky.m_d)"
        return false
    else
        return false
    end
end

xopt = prior_sample(rng, post)
imageviz(intensitymap(skymodel(post, xopt), grid), size=(500, 400))
xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(0.1); initial_params=xopt, maxiters=1_000, g_tol=1e-1, callback=Callback(10,()->nothing))

CM.plot(loss_arr)
post(xopt)
img = intensitymap(skymodel(post, xopt), imagepixels(fovx, fovy, npix, npix))
imageviz(img, size=(500, 400))
imageviz(true_intmap, size=(500, 400))
imageviz(((img .- true_intmap)), size=(500, 400))
xopt.sky.σimg
residual(post, xopt) 
#xopt.sky.m_d |> rad2μas
dvis_mdl = deepcopy(dvisamp)
#
gfour = FourierDualDomain(grid, arrayconfig(dvisamp), NFFTAlg())
dvis_mdl = deepcopy(dvisamp)
dvis_mdl.measurement .= abs.(visibilitymap(skymodel(post, xopt), gfour) )
Plots.plot(dvis_mdl, label="model");
Plots.plot!(dvisamp)