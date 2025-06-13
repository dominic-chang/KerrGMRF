using Pkg;Pkg.activate(dirname(@__DIR__));
using Comrade
using Pyehtim
using Krang
using Enzyme
using StableRNGs
using Accessors
import CairoMakie as CM
using BasicInterpolators
using FINUFFT
Enzyme.API.printall!(true)
Enzyme.API.printperf!(true)

rng = StableRNG(1234)

include(joinpath((@__DIR__),"utils.jl"))
include(joinpath((@__DIR__),"modifiers.jl"))
include(joinpath((@__DIR__),"models","KerrGMRF.jl"))
include(joinpath((@__DIR__),"models","JuKeBOX.jl"))

seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.1
bulkx = 100.0
bulky = 100.0
npix = 100
bulkpix = 100
snrcut = 3.0
uv_min = 0.1e9
fovx = μas2rad(100.0)
fovy = μas2rad(100.0)
path = joinpath(dirname(@__DIR__), "data", "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits")


# Get observer information
obs = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min=0.1e9).add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))

baselineplot(dlcamp, :uvdist, :measwnoise, error=true, label="Visibilities")

using VLBIImagePriors
using Distributions

function ModifiedKerrGMRF(θ, meta) 
    truth = (θs = 75.0, rpeak = 4.7, p1 = 0.5, p2 = 3.7, χ = 2.5, pa=-0.5, ι = 0.6, βv = 0.2, spec = 0.09, η = -2.95,f=0.6)
    θ2 = merge(truth, θ)

    m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa))
    RenormalizedFlux(m, 0.6)
end

using Optimization
using OptimizationOptimisers
using Enzyme
using StableRNGs
import Plots 

bulkgrid = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor=Serial())
transform1, cprior1  = matern(size(bulkgrid))
prior = (
    m_d = Uniform(μas2rad(2.0), μas2rad(6.0)), 
    spin = Uniform(-0.99, -0.01),#-0.99, -0.01),
    θo = Uniform(1, 40.0),
    #θs = Uniform(40, 90),
    #rpeak = Uniform(1.0, 18.0),
    #p1 = Uniform(0.1, 10.0),
    #p2 = Uniform(1.0, 10.0),
    #χ = Uniform(-π, π),
    #pa = Uniform(-π, 0),
    #ι = Uniform(0, π/2),
    #βv = Uniform(0.01, 0.99),
    #spec = Uniform(-1.0, 5.0),
    #η = Uniform(-π, π),
    #f = Uniform(0.2, 1.0),
    c1 = cprior1,
    #c2 = cprior1,
    σimg = truncated(Normal(0.0, 1.2); lower=0.0),
    #lower is related to pixel size
    ρpr = truncated(InverseGamma(1.0, -log(0.1)*10); lower = 5.0, upper=2*max(size(bulkgrid)...)),
    νpr = Uniform(1.0, 5.0)
)

skym = SkyModel(
    ModifiedKerrGMRF, 
    prior, 
    imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx()); 
    metadata=(;bulkgrid, transform1),
    algorithm=FINUFFTAlg(;threads=1)
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode=set_runtime_activity(Enzyme.Reverse))

using Enzyme
tpost = asflat(post)
x = prior_sample(tpost)
Comrade.LogDensityProblems.logdensity_and_gradient(tpost, x)

# xopt_true = ((rng, post)-> begin
#     temp1 =  prior_sample(rng, post)
#     truth = (m_d = μas2rad(4.01), spin = -0.94, θo = 30.0, rpeak = 4.7, p1 = 0.5, p2 = 3.7, χ = 2.5, pa=-0.5, ι = 0.6, βv = 0.2, spec = 0.09, η = -2.95,f=0.6)

#     vals = collect(values(temp1.sky))
#     ks = collect(keys(temp1.sky))

#     for k in filter(x-> !isa(x, Nothing), indexin(collect(keys(truth)), ks))
#         vals[k] = truth[ks[k]]
#     end
#     temp1 = (sky=NamedTuple{Tuple(ks)}(vals),)
#     return temp1
# end)(rng, post)
# imageviz(intensitymap(skymodel(post, xopt_true), skym.grid), size=(500, 400))

# (newdlcamp, newdcphase) = ((post, xopt_true, obs) -> begin
#     return simulate_observation(post, xopt_true;add_thermal_noise=false)
# end)(post, xopt_true, obs)

# newdvis = ((xopt_true, post, obs) -> begin
#     vis = extract_table(obs, Visibilities())
#     vis.measurement .= forward_model(post, xopt_true)
#     return vis
# end)(xopt_true, post, obs)

# newdvisamp = ((xopt_true, post, obs) -> begin
#     visamp = extract_table(obs, VisibilityAmplitudes())
#     visamp.measurement .= abs.(forward_model(post, xopt_true))
#     return visamp
# end)(xopt_true, post, obs)

# post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), newdlcamp, newdcphase; admode=set_runtime_activity(Enzyme.Reverse))
# fpost = Comrade.asflat(post)
# chi2(post, xopt_true)

# using AdvancedHMC
# chain = sample(post, NUTS(0.8), 20_000; n_adapts=10_000, progress=true, initial_params=xopt_true, saveto=DiskStore(;name=joinpath(dirname(@__DIR__),"Results_self_low_res"), stride=10))#, restart=true);#;name=joinpath((dirname(@__DIR__),"Results_smaller_res"))));


# # Import data from disk
# chain = load_samples(joinpath(dirname(@__DIR__),"Results_self_low_res"))
# fchain = Comrade.inverse.(Ref(fpost), chain)
# msamples = skymodel.(Ref(post), chain)
# stats= samplerstats(chain)



# # Check chain spread of random parameter
# rand_dir = rand(1:length(cprior1))
# CM.lines([i[rand_dir] for i in fchain[120:end]], linewidth=0.5)

# # Check data points spread of random sample
# fig = CM.Figure();
# ax = CM.Axis(fig[1, 1])
# baselineplot!(ax, newdlcamp, :uvdist, :measwnoise, error=true, color=:blue)
# for i in 1:1
#     (tempdlcamp, tempdcphase) = ((post, xopt_true, obs) -> begin
#         return simulate_observation(post, rand(chain);add_thermal_noise=false)
#     end)(post, xopt_true, obs)
#     baselineplot!(ax,tempdlcamp, :uvdist, :measwnoise, error=true, color=:orange)
# end
# display(fig)


# (tempdlcamp, tempdcphase) = ((post, xopt_true, obs) -> begin
#     return simulate_observation(post, rand(chain);add_thermal_noise=false)
# end)(post, xopt_true, obs)
# collect(tempdlcamp.measurement ./ sqrt.([tempdlcamp.noise[i,i] for i in length(tempdlcamp.noise[1,:])])) |> Plots.plot
# collect(tempdcphase.measurement ./ sqrt.([tempdcphase.noise[i,i] for i in length(tempdcphase.noise[1,:])])) |> Plots.plot

# collect(newdlcamp.measurement ./ sqrt.([newdlcamp.noise[i,i] for i in length(newdlcamp.noise[1,:])])) |> Plots.plot
# collect(newdcphase.measurement ./ sqrt.([newdcphase.noise[i,i] for i in length(newdcphase.noise[1,:])])) |> Plots.plot

# collect(dlcamp.measurement ./ sqrt.([dlcamp.noise[i,i] for i in length(dlcamp.noise[1,:])])) |> Plots.plot
# collect(dcphase.measurement ./ sqrt.([dcphase.noise[i,i] for i in length(dcphase.noise[1,:])])) |> Plots.plot

# # Check residuals of samples
# resids_list = Comrade.residuals.(Ref(post), [rand(chain) for i in 1:20])
# begin
#     fig = CM.Figure();
#     ax = CM.Axis(fig[1, 1])
#     ax2 = CM.Axis(fig[1, 2])
#     for res in resids_list
#         baselineplot!(ax, res[1], :uvdist, :res, color=:grey, alpha = 0.1)
#         baselineplot!(ax2, res[2], :uvdist, :res, color=:grey, alpha = 0.1)
#     end
#     display(fig)
# end

# CM.lines(rad2μas.((chain.sky.m_d[120:end])))
# CM.lines(moving_average(rad2μas.((chain.sky.m_d[120:end])), 10))
# CM.lines((chain.sky.spin[120:end]))
# CM.lines((chain.sky.θo[120:end]))
# CM.lines((chain.sky.σimg[120:end]))
# begin 
#     fig = CM.Figure();
#     ax = CM.Axis(fig[1, 1], yscale=log10)
#     CM.lines!(ax, stats.step_size[120:end])
#     display(fig)
# end
# begin 
#     fig = CM.Figure();
#     ax = CM.Axis(fig[1, 1], yscale=log10)
#     CM.lines!(ax, moving_average(stats.step_size[120:end],10))
#     display(fig)
# end
# CM.scatter(stats.numerical_error, alpha=0.1)
# CM.scatter(stats.tree_depth, alpha=0.2)
# CM.lines([(mean([i[55] for i in chain.sky.c1][120+i:140+i])) for i in 1:(length(chain.sky.m_d)-140)])
# begin
#     fig = CM.Figure();
#     a, b, c, d = rand(3:dimension(post), 4)
#     CM.lines!(CM.Axis(fig[1, 1]), [i[a] for i in chain.sky.c1][120:end])
#     CM.lines!(CM.Axis(fig[1, 2]), [i[b] for i in chain.sky.c1][120:end])
#     CM.lines!(CM.Axis(fig[2, 1]), [i[c] for i in chain.sky.c1][120:end])
#     CM.lines!(CM.Axis(fig[2, 2]), [i[d] for i in chain.sky.c1][120:end])
#     display(fig)
# end
# CM.scatter(rad2μas.((chain.sky.m_d[120:end])),([i[rand_dir] for i in chain.sky.c1][120:end]), alpha=0.5)
# CM.scatter(([i[1000] for i in chain.sky.c1][120:end]),([i[1200] for i in chain.sky.c1][120:end]), alpha=0.5)
# CM.scatter(rad2μas.((chain.sky.m_d[120:end])), (chain.sky.spin[120:end]), alpha=0.5)

# using StatsBase
# newgrid = imagepixels(2*fovx, 2*fovy, npix, npix; executor=ThreadsEx())
# imgs = intensitymap.(msamples[120:end], Ref(newgrid))
# mimg = mean(imgs)
# simg = std(imgs)
# fig = CM.Figure(;resolution=(700, 700));
# axs = [CM.Axis(fig[i, j], xreversed=true, aspect=1) for i in 1:2, j in 1:2]
# CM.image!(axs[1,1], mimg, colormap=:afmhot); axs[1, 1].title="Mean"
# CM.image!(axs[1,2], simg./(max.(mimg, 1e-8)), colorrange=(0.0, 2.0), colormap=:afmhot);axs[1,2].title = "Std"
# CM.image!(axs[2,1], rand(imgs),   colormap=:afmhot);
# CM.image!(axs[2,2], rand(imgs),   colormap=:afmhot);
# CM.hidedecorations!.(axs)
# fig
# p = Plots.plot(layout=(2,1));
# for s in sample(chain, 10)
#     residual!(post, s)
# end
# p
