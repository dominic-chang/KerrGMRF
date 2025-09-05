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
bulkpix = 20#50
raster_size = 180.0 # in microarcseconds    
snrcut = 3.0
uv_min = 0.1e9
fovx = μas2rad(150.0)
fovy = μas2rad(150.0)
npix = 30
year = 2017


data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 =>"L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits")
path = joinpath(dirname(@__DIR__), "data", data[year])


# Get observer information
obs = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min=0.1e9).add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))

using VLBIImagePriors
using Distributions, DistributionsAD

function ModifiedKerrGMRF(θ2, meta) 
    m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa), Shift(μas2rad(12.0), μas2rad(12.0)))
    return RenormalizedFlux(m, θ2.f) 
    
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
    θs = VLBIImagePriors.DeltaDist(90.0),
    rpeak = Uniform(1.0, 4.0),
    p1 = Uniform(0.1, 5.0),
    p2 = Uniform(1.0, 5.0),
    χ = Uniform(-π, π),
    ι = Uniform(0, π/2),
    βv = Uniform(0.01, 0.99),
    spec = Uniform(-1.0, 5.0),
    η = Uniform(-π, π),
    frac = VLBIImagePriors.DeltaDist(1.0),#Uniform(0.0, 1.0),
    pa =VLBIImagePriors.DeltaDist(-72*π/180),# Uniform(-π, 0),
    f = VLBIImagePriors.DeltaDist(0.6),#Uniform(-π, π),
    σimg = truncated(Normal(0.0, 1.0); lower=0.0),
    ρpr = truncated(InverseGamma(1.0, -log(0.1)*10); lower = 1.0, upper=2*max(size(bulkgrid)...)),
    νpr = Uniform(1.0, 5.0),
    c1 = cprior1,
)

skym = SkyModel(
    ModifiedKerrGMRF, 
    prior, 
    imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx()); 
    metadata=(;bulkgrid, transform1, raster_size)
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode=set_runtime_activity(Enzyme.Reverse))

fpost = asflat(post)
psample = prior_sample(post)
Comrade.transform(fpost, rand(dimension(fpost)))

vals = rand(dimension(fpost))
dvals = zeros(size(vals))

mutable struct Callback 
    counter::Int
    stride::Int
    const f::Function
end
Callback(stride, f) = Callback(0, stride, f)
gfour = FourierDualDomain(skym.grid, arrayconfig(dvis), NFFTAlg())
dvis_mdl = extract_table(obs, VisibilityAmplitudes())
dvis_mdl.measurement .= abs.(visibilitymap(skymodel(post, prior_sample(post)), gfour) )
dcphase_mdl = extract_table(obs, ClosurePhases(snrcut=3.0))
dcphase_mdl.measurement .= Comrade.closure_phases(forward_model(post, psample), Comrade.designmat(dcphase_mdl.config))
loss_arr = []
function (c::Callback)(state, loss, others...)
    global loss_arr
    global fpost
    global dcphase_mdl 
    global st
    global dcphase#newdvisamp 
    c.counter += 1  
    curr = transform(fpost,state.u)
    
    if isnan(loss)
        println("NaN error")
        println(curr)
        throw(ArgumentError("NaN error"))
    end
    f = open(joinpath((@__DIR__),"err.txt"), "w")
    write(f, string(curr))
    close(f)
#
    if c.counter % c.stride == 0
        append!(loss_arr, loss)
#
        curr = transform(fpost,state.u)
#
        dcphase_mdl.measurement .= Comrade.closure_phases(forward_model(post, curr), Comrade.designmat(dcphase_mdl.config))
        plt = residual(post, curr)

        img = Plots.plot(intensitymap(skymodel(post, curr), skym.grid), size=(500, 400),cb=:none, xlabel="√(Convex Quadrangle Area) (Gλ)")#, ylabel="Residual Log Closure Amplitude", label="Model Image")
#
        display(Plots.plot(plt, img, size=(1000, 400), layout=(1,2)))   
#
#
        @info "On step $(c.counter) mass=$(rad2μas(curr.sky.m_d)) inc=$(curr.sky.θo), spin=$(curr.sky.spin), flux_rat=$(curr.sky.frac), σimg=$(curr.sky.σimg), pa=$(unsafe_trunc(Int, curr.sky.pa*180/π)), θs=$(curr.sky.θs), p2=$(curr.sky.p2)"
        return false
    else
        return false
    end
end

#for _ in 1:10
xopt = ((rng, post) -> begin 
    temp = prior_sample(rng, post)
    truth = (m_d = μas2rad(3.8), spin = -0.5, θo = 17.0, rpeak = 1.5, p1 = 3.0, p2 = 2.0, χ =-π/2, ι = π/3, βv = 0.5, spec = 1.0, η = π/2, pa=-72*π/180, σimg=1.00)#, θs=80,)# frac=0.5)

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

imageviz(ComradeBase.intensitymap(m, newgrid),colorscale=log, colorrange=(1e-10, 1e-2))

dlcamp, dcphase = Comrade.simulate_observation(post, xopt; add_thermal_noise=add_th_noise)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode=set_runtime_activity(Enzyme.Reverse))
fpost = asflat(post)

using EnzymeTestUtils
using FiniteDifferences
vals = rand(dimension(fpost))
dvals = zero(vals)
Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), Const(fpost), Enzyme.Active, Enzyme.Duplicated(vals, dvals)) # check that the model is working
FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), fpost, vals)

xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(lr); initial_params=xopt, maxiters=100, g_tol=1e-1, callback=Callback(1,()->nothing))

newgrid = imagepixels(fovx, fovy, npix*3, npix*3; executor=ThreadsEx())
imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid),colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, xopt),μas2rad(20/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, xopt),μas2rad(5/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
plt = residual(postJB, xopt)
plt.subplots[1].attr[:xaxis].plotattributes[:guide] = "√(Convex Quadrangle Area) (Gλ)"
plt.subplots[1].attr[:yaxis].plotattributes[:guide] = "Residual Log Closure Amplitude"

using Pathfinder
result = pathfinder(fpost; init=Comrade.inverse(fpost,xopt), ndraws_elbo=6, ntries=2_000)
inv_metric = result.fit_distribution_transformed.Σ
init_params = result.draws[:, 1]
transform(fpost,init_params)
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost,init_params)), newgrid))
#
transform(fpost,init_params)


using AdvancedHMC
integrator = Leapfrog(0.01)
#metric = DenseEuclideanMetric(dimension(fpost))
metric = DenseEuclideanMetric(Matrix(inv_metric))
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator); 
                         init_buffer=100, term_buffer=200)
smplr = HMCSampler(kernel, metric, adaptor)
out=joinpath(dirname(@__DIR__),"Results_non_diagonal_metric_self_bad_parameterization_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(rad2μas(fovx)))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
#chain = sample(rng, post, smplr, 20_000; n_adapts=10_000, saveto=DiskStore(mkpath(out), 10), initial_params=transform(fpost, init_params))#, restart=true) 
chain = sample(rng, post, smplr, 20_000; n_adapts=10_000, saveto=DiskStore(mkpath(out), 10), initial_params=xopt, restart=true)

#out=joinpath(dirname(@__DIR__),"Results_diagonal_metric_self_bad_parameterization_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(rad2μas(fovx)))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
#chain = sample(post, AdvancedHMC.NUTS(0.8), 20_000; n_adapts=10_000, progress=true, initial_params=xopt, saveto=DiskStore(mkpath(out), 10), restart=true);#;name=joinpath((dirname(@__DIR__),"Results_smaller_res"))));

# Import data from disk
using Serialization
run_name = "Results_non_diagonal_metric_self_bad_parameterization_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(rad2μas(fovx)))_fov_year_$year"
#run_name = "Results_non_diagonal_metric_50_rast_low_res"
out=joinpath(dirname(@__DIR__),run_name)
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
stats= samplerstats(chain)

rand_dir = rand(1:length(cprior1))
CM.lines([i[rand_dir] for i in fchain], linewidth=0.5)
CM.lines((chain.sky.spin))
CM.lines(rad2μas.(chain.sky.m_d))

# Check data points spread of random sample
fig = CM.Figure();
ax = CM.Axis(fig[1, 1])
baselineplot!(ax, dlcamp, :uvdist, :measwnoise, error=true, color=:blue)
for i in 1:100
    (tempdlcamp, tempdcphase) = ((post, obs) -> begin
        return simulate_observation(post, rand(chain);add_thermal_noise=false)
    end)(post, obs)
    baselineplot!(ax,tempdlcamp, :uvdist, :measwnoise, error=false, color=:orange)
end
display(fig)


(tempdlcamp, tempdcphase) = ((post, obs) -> begin
    return simulate_observation(post, rand(chain);add_thermal_noise=false)
end)(post, obs)
collect(tempdlcamp.measurement ./ sqrt.([tempdlcamp.noise[i,i] for i in length(tempdlcamp.noise[1,:])])) |> Plots.plot
collect(tempdcphase.measurement ./ sqrt.([tempdcphase.noise[i,i] for i in length(tempdcphase.noise[1,:])])) |> Plots.plot

collect(dlcamp.measurement ./ sqrt.([dlcamp.noise[i,i] for i in length(dlcamp.noise[1,:])])) |> Plots.plot
collect(dcphase.measurement ./ sqrt.([dcphase.noise[i,i] for i in length(dcphase.noise[1,:])])) |> Plots.plot

# Check residuals of samples
resids_list = Comrade.residuals.(Ref(post), [rand(chain) for i in 1:20])
begin
    fig = CM.Figure();
    ax = CM.Axis(fig[1, 1])
    ax2 = CM.Axis(fig[1, 2])
    for res in resids_list
        baselineplot!(ax, res[1], :uvdist, :res, color=:grey, alpha = 0.1)
        baselineplot!(ax2, res[2], :uvdist, :res, color=:grey, alpha = 0.1)
    end
    display(fig)
end

using Accessors
temp = chain[end]
@reset temp.sky.frac = 0.01
@reset temp.sky.m_d = μas2rad(3.0)
intensitymap(skymodel(post,temp), skym.grid) |> imageviz

CM.lines(stats.log_density)

CM.lines(rad2μas.((chain.sky.m_d)))
CM.lines(moving_average(rad2μas.((chain.sky.m_d)), 10))
CM.lines((chain.sky.spin))
CM.lines((chain.sky.rpeak))
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

begin 
    fig = CM.Figure();
    ax = CM.Axis(fig[1, 1], yscale=log10)
    CM.lines!(ax, stats.step_size)
    display(fig)
end
begin 
    fig = CM.Figure();
    ax = CM.Axis(fig[1, 1], yscale=log10)
    CM.lines!(ax, moving_average(stats.step_size,10))
    display(fig)
end
CM.scatter(stats.numerical_error, alpha=0.1)
CM.scatter(stats.tree_depth, alpha=0.2)
CM.lines([(mean([i[55] for i in chain.sky.c1][120+i:140+i])) for i in 1:(length(chain.sky.m_d)-140)])
begin
    fig = CM.Figure();
    a, b, c, d = rand(3:dimension(post), 4)
    CM.lines!(CM.Axis(fig[1, 1]), [i[a] for i in chain.sky.c1])
    CM.lines!(CM.Axis(fig[1, 2]), [i[b] for i in chain.sky.c1])
    CM.lines!(CM.Axis(fig[2, 1]), [i[c] for i in chain.sky.c1])
    CM.lines!(CM.Axis(fig[2, 2]), [i[d] for i in chain.sky.c1])
    display(fig)
end
CM.scatter(rad2μas.((chain.sky.m_d)),([i[rand_dir] for i in chain.sky.c1]), alpha=0.5)
CM.scatter(([i[1000] for i in chain.sky.c1]),([i[1200] for i in chain.sky.c1]), alpha=0.5)
CM.scatter(rad2μas.((chain.sky.m_d)), (chain.sky.spin), alpha=0.5)
CM.scatter(rad2μas.((chain.sky.m_d)), (chain.sky.rpeak), alpha=0.5)
CM.scatter(rad2μas.((chain.sky.m_d)), (chain.sky.χ), alpha=0.5)
#CM.scatter(rad2μas.((chain.sky.m_d[120:end])), (chain.sky.pa[120:end]), alpha=0.5)
#CM.scatter((chain.sky.spin[120:end]), (chain.sky.pa[120:end]), alpha=0.5)
#CM.scatter((chain.sky.rpeak[120:end]), (chain.sky.pa[120:end]), alpha=0.5)

fig = CM.Figure(resolution=(700, 700));
ax = CM.Axis(fig[1, 1], aspect=1)

CM.plot(post.(chain))

psample = Comrade.inverse(fpost, rand(chain))
chi2(post, Comrade.transform(fpost, psample)) ./ (length(dlcamp), length(dcphase))
post(Comrade.transform(fpost, psample))

using MCMCChains
ess((reduce(hcat, fchain))[1,begin:end])

using StatsBase
using LinearAlgebra

imgs = intensitymap.(msamples, Ref(skym.grid))
mimg = mean(imgs)
simg = std(imgs)
fig = CM.Figure(;resolution=(700, 700));
axs = [CM.Axis(fig[i, j], xreversed=true, aspect=1) for i in 1:2, j in 1:2]
CM.image!(axs[1,1], mimg, colormap=:afmhot); axs[1, 1].title="Mean"
CM.image!(axs[1,2], simg./(max.(mimg, 1e-8)), colorrange=(0.0, 2.0), colormap=:afmhot);axs[1,2].title = "Std"
CM.image!(axs[2,1], rand(imgs),   colormap=:afmhot);
CM.image!(axs[2,2], rand(imgs),   colormap=:afmhot);
CM.hidedecorations!.(axs)
fig
p = Plots.plot(layout=(2,1));
for s in sample(chain, 10)
    residual!(post, s)
end
p

newgrid = imagepixels(fovx, fovy, 2npix, 2npix)
for i in 1:10:length(chain)
    #img = imageviz(intensitymap(smoothed(skymodel(post, chain[i]), μas2rad(7/(2*√(2*log(2))))), skym.grid), size=(500, 400))
    img = imageviz(intensitymap(skymodel(post, chain[i]), newgrid), size=(500, 400))
    println(i)
    sleep(0.1)
    display(img)
end


using PairPlots
using LaTeXStrings

postsamps = Comrade.postsamples(chain)
ks = [:m_d, :spin, :θo,:rpeak, :χ, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]
mat = reduce(hcat,getproperty.(Ref(postsamps.sky), ks))#[2000:end,:]
table = NamedTuple{Tuple(ks)}([mat[:,i] for i in 1:length(ks)])

pairplot(
    #reduce(hcat,getproperty.(Ref(postsamps.sky), [:m_d, :spin, :θo,:rpeak, :χ, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]))=>
    table =>
    (PairPlots.Scatter(),PairPlots.MarginHist()),
    #labels=[:m_d, :spin, :θo,:rpeak, :χ, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]
    labels=Dict(:m_d => L"\theta_g", :spin => L"a", :θo => L"\theta_o", :pa => L"p.a.", :rpeak => L"R", :θs => L"\theta_s"),
)


model = skymodel(post, chain[end])
model.model.model.scene[1].material