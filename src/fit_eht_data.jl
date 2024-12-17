using Pkg;Pkg.activate(dirname(@__DIR__));
using Comrade
using Pyehtim
using Krang
using Enzyme

include(joinpath((@__DIR__),"modifiers.jl"))
include(joinpath((@__DIR__),"models","KerrGMRF.jl"))
include(joinpath((@__DIR__),"models","JuKeBOX.jl"))
#Enzyme.Compiler.RunAttributor[] = false

seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.01
bulkx = 40.0
bulky = 40.0
npix = 60
bulkpix = 80
fovx = μas2rad(120.0)
fovy = μas2rad(120.0)
# TODO: try NMSE

inobs = ehtim.obsdata.load_uvfits(joinpath(dirname(@__DIR__), "data", "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits"))
grid = imagepixels(fovx, fovy, npix, npix; executor=Serial(), header=ComradeBase.MinimalHeader(string(inobs.source), pyconvert.(Ref(Float64), [inobs.ra, inobs.dec, inobs.mjd, inobs.rf])...))

#truth = (m_d = μas2rad(5.01), spin = -0.94, θo = 30.0, θs = 70.0, rpeak = 4.7, p1 = 0.1, p2 = 3.7, χ = 2.5, ι = 1.6, βv = 0.2, spec = 0.09, η = -2.95)
truth = (m_d = 5.01, spin = -0.94, θo = 30.0, θs = 70.0, rpeak = 4.7, p1 = 0.5, p2 = 3.7, χ = 2.5, ι = 1.6, βv = 0.2, spec = 0.09, η = -2.95)
ModifiedJuKeBOX(θ, meta) = RenormalizedFlux(modify(JuKeBOX(θ), Stretch(μas2rad(θ.m_d), μas2rad(θ.m_d))), 0.6)

obs = inobs
obs = scan_average(obs.flag_uvdist(uv_min=0.1e9))
obs = obs.add_fractional_noise(fractional_noise)

dvis = extract_table(obs, Visibilities())
dvisamp = extract_table(obs, VisibilityAmplitudes())
dcphase = extract_table(obs, ClosurePhases())
dlcamp = extract_table(obs, LogClosureAmplitudes())

import CairoMakie as CM
using VLBIImagePriors
using Distributions, DistributionsAD


# # Define model to fit
function ModifiedKerrGMRF(θ, meta) 
    RenormalizedFlux(modify(KerrGMRF(θ, meta), Stretch(μas2rad(θ.m_d), μas2rad(θ.m_d)), Rotate(θ.pa)), θ.f)
end

bulkgrid = imagepixels(bulkx, bulky, bulkpix, bulkpix, executor=Serial())
grid = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx(:Enzyme))#ThreadsEx(:Enzyme))

bulkimg = IntensityMap(zeros(Float64, bulkpix, bulkpix), bulkgrid)
skymeta = (;bulkimg = bulkimg, npix=npix)#, θ=truth)

cprior = corr_image_prior(bulkgrid, 1.0; base=GMRF, order=1, upper=2*bulkpix)
prior = (
    m_d = Uniform(2.0, 8.0), 
    spin = Uniform(0.01, 0.99),
    θo = Uniform(140.0, 179.0),
    #θs = Uniform(50, 90),
    rpeak = Uniform(1.0, 8.0),
    p1 = Uniform(0.1, 5.0),
    p2 = Uniform(0.1, 5.0),
    χ = Uniform(-π, π),
    pa = Uniform(-π, π),
    ι = Uniform(0, π),
    βv = Uniform(0.01, 0.99),
    spec = Uniform(-1.0, 3.0),
    η = Uniform(-π, π),
    f = Uniform(0.1, 2.0),
    c = cprior,
    σimg = truncated(Normal(0.0, 0.5); lower=0.0)
)

skym = SkyModel(ModifiedKerrGMRF, prior, grid; metadata=skymeta)
#skym = SkyModel(ModifiedJuKeBOX, prior, grid; metadata=skymeta)
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

post = VLBIPosterior(skym, intmodel, dvis; admode=set_runtime_activity(Enzyme.Reverse))

p_sample = prior_sample(post)
imageviz(intensitymap(skymodel(post, p_sample), grid), size=(500, 400))
@time chi2(post, p_sample) ./ (length(dcphase), length(dlcamp))
p_sample.sky.σimg
p_sample.sky.c.params
gfour = FourierDualDomain(grid, arrayconfig(dvisamp), NFFTAlg())
dvis_mdl = extract_table(obs, VisibilityAmplitudes())
abs.(visibilitymap(skymodel(post, p_sample), gfour) )
dvis_mdl.measurement .= abs.(visibilitymap(skymodel(post, p_sample), gfour) )
import Plots
Plots.plot(dvis_mdl, label="model");
Plots.plot!(dvisamp)


using Optimization
using OptimizationOptimisers
using OptimizationBBO
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
    global st
    c.counter += 1  
    curr = transform(fpost,state.u)
    if c.counter % c.stride == 0
        append!(loss_arr, loss)

        curr = transform(fpost,state.u)
        dvis_mdl.measurement .= abs.(forward_model(post, curr)) 
        #dvis_mdl.measurement .= Comrade.closure_phases(measurement(dvis_mdl), Comrade.designmat(arrayconfig(dvis_mdl)))
        plt = Plots.plot(dvis_mdl, label="model");
        Plots.plot!(dvisamp)
        img = Plots.plot(intensitymap(skymodel(post, curr), grid), size=(500, 400))

        display(Plots.plot(plt, img, size=(1000, 500)))   
        @info "On step $(c.counter) loss = $(loss) mass = $(curr.sky.m_d)"
        return false
    else
        return false
    end
end

xopt = prior_sample(rng, post)
imageviz(intensitymap(skymodel(post, xopt), grid), size=(500, 400))
xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(0.1); initial_params=xopt, maxiters=10_000, g_tol=1e-1, callback=Callback(10,()->nothing))
#xopt, sol = comrade_opt(post, OptimizationBBO.BBO_adaptive_de_rand_1_bin(), maxiters=5_000, g_tol=1e-1, callback=Callback(10,()->nothing), cube=true)
xopt

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

chain = sample(post, NUTS(0.8), 500; n_adapts=500, progress=true, initial_params=xopt)#, adtype=Val(:Enzyme));#, saveto=DiskStore());
#chain = load_samples("/Volumes/Working/Documents/Harvard/Physics/Research/KerrGMRF/Results")

msamples = []
append!(msamples, skymodel.(Ref(post), chain))
#end
#
using StatsBase
imgs = intensitymap.(msamples, Ref(grid))
mimg = mean(imgs)
simg = std(imgs)
fig = CM.Figure(;resolution=(700, 700));
axs = [CM.Axis(fig[i, j], xreversed=true, aspect=1) for i in 1:2, j in 1:2]
CM.image!(axs[1,1], mimg, colormap=:afmhot); axs[1, 1].title="Mean"
CM.image!(axs[1,2], simg./(max.(mimg, 1e-8)), colorrange=(0.0, 2.0), colormap=:afmhot);axs[1,2].title = "Std"
CM.image!(axs[2,1], rand(imgs),   colormap=:afmhot);
CM.hidedecorations!.(axs)
fig
p = Plots.plot(layout=(2,1));
for s in sample(chain, 10)
    residual!(post, s)
end
p
chain.sky.m_d |> rad2μas|> CM.hist
chain.sky.rpeak |> CM.hist
chain.sky.p1 |> CM.hist
chain.sky.p2 |> CM.hist
chain.sky.σimg |> CM.scatter

intensitymap(skymodel(post, rand(chain[100:end])), grid) |> imageviz
gfour = FourierDualDomain(grid, arrayconfig(dvis), NFFTAlg())
dvis_mdl = deepcopy(dvis)
dvis_mdl.measurement .= visibilitymap(skymodel(post, rand(chain[100:end])), gfour)
dcphase_mdl = deepcopy(dcphase)
dcphase_mdl.measurement .= Comrade.closure_phases(measurement(dvis_mdl), Comrade.designmat(arrayconfig(dcphase)))
Plots.plot(dcphase, label="model");
Plots.plot!(dcphase_mdl)

dvisamp_mdl = deepcopy(dvisamp)
dvisamp_mdl.measurement .= abs.(visibilitymap(skymodel(post, rand(chain[100:end])), gfour))
Plots.plot(dvisamp, label="model");
Plots.plot!(dvisamp_mdl)
