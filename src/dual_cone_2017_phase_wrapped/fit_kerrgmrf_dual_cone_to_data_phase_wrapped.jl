using Pkg;Pkg.activate(dirname(dirname(@__DIR__)));
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
LinearAlgebra.BLAS.set_num_threads(48) # to avoid threading conflicts with FINUFFT
rng = StableRNG(1234)

include(joinpath(dirname(@__DIR__),"utils.jl"))
include(joinpath(dirname(@__DIR__),"plotting","utils.jl"))
include(joinpath(dirname(@__DIR__),"modifiers.jl"))
include(joinpath(dirname(@__DIR__),"models","KerrGMRF.jl"))

lr = 0.05
seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.01
bulkx = 1.0
bulky = 1.0
bulkpix = 50#20
raster_size = 120.0# 90 # in microarcseconds    
raster_size = 150.0# 90 # in microarcseconds    
snrcut = 3.0
uv_min = 0.1e9
fovx = μas2rad(150.0)
fovy = μas2rad(150.0)
npix = 50
year = 2017

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 =>"L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(dirname(@__DIR__)), "data", data[year])


# Get observer information
obs = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min=0.1e9).add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))
baselineplot(dvisamp, :uvdist, :measwnoise, error=true, label="Visibility Amplitudes")

using VLBIImagePriors
using Distributions, DistributionsAD

function ModifiedKerrGMRF(θ2, meta) 
    m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa*π/180), Shift(μas2rad(12.0), μas2rad(6.0)))
    return m
end

using Optimization
using OptimizationOptimisers
using Enzyme
using StableRNGs
import Plots 
using LaTeXStrings

bulkgrid = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor=ThreadsEx())
transform1, cprior1  = matern(size(bulkgrid))
transform2, cprior2  = matern(size(bulkgrid))
prior = (
    m_d = Uniform(μas2rad(1.0), μas2rad(8.0)), 
    spin = Uniform(0.01, 0.99),
    θo = Uniform(120.0, 179.0),
    θs = Uniform(40.0, 90.0),
    rpeak = Uniform(1.0, 10.0),
    p1 = Uniform(0.1, 5.0),
    p2 = Uniform(1.0, 5.0),
    χ = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
    pa =Uniform(0.0, 180.0),
    ι = Uniform(0, π/2),
    βv = Uniform(0.01, 0.99),
    spec = Uniform(-1.0, 5.0),
    η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
    frac = VLBIImagePriors.DeltaDist(1.0),#Uniform(0.0, 1.0),
    #pa =VLBIImagePriors.DeltaDist(π-72*π/180),# Uniform(-π, 0),
    f = VLBIImagePriors.DeltaDist(0.6),#Uniform(-π, π),
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
    metadata=(;bulkgrid, transform1, transform2, raster_size, offset),
    algorithm=FINUFFTAlg(;threads=1) 
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode=set_runtime_activity(Enzyme.Reverse))
#post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dvis; admode=set_runtime_activity(Enzyme.Reverse))

fpost = asflat(post)
psample = prior_sample(post)
Comrade.transform(fpost, rand(dimension(fpost)))

vals = rand(dimension(fpost))
dvals = zeros(size(vals))
#function temp1(fpost, val, dval)
#    for _ in 1:1000
#        Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), Const(fpost), Enzyme.Active, Enzyme.Duplicated(val, dval)) # check that the model is working
#    end
#end
#function temp1(fpost, vals)
#    for _ in 1:200
#        Comrade.LogDensityProblems.logdensity_and_gradient(fpost, vals)
#    end
#end
#temp1(fpost, vals)
#@profview temp1(fpost, vals)
using LaTeXStrings
mutable struct Callback 
    counter::Int
    stride::Int
    const f::Function
end
Callback(stride, f) = Callback(0, stride, f)
loss_arr = []

num_data_prods= length(post.data)
function (c::Callback)(state, loss, others...)
    global loss_arr
    global fpost
    global st
    c.counter += 1
    curr = transform(fpost, state.u)
    fig = CM.Figure(;size=(1200, 600));
    if c.counter % c.stride == 0
        append!(loss_arr, loss)
        curr = transform(fpost, state.u)
        resids = Comrade.residuals(post, curr)
        chi2_vals = Comrade.chi2(post, curr) ./ (length(resids[i].measurement) for i in 1:length(resids))
        for (i,resid) in enumerate(resids)
            ax = CM.Axis(fig[i, 1], xlabel=CM.L"\sqrt{\text{(Convex Quadrangle Area)}} (G$\lambda$)", ylabel=LaTeXStrings.latexstring("\\text{Residual $(typeof(resid).parameters[1].name.name)}"), title=LaTeXStrings.latexstring("\\langle\\chi^2\\rangle=$(round(chi2_vals[i], digits=2))"), titlesize=24, xlabelsize=24, ylabelsize=14)
            baselineplot!(ax, resid, :uvdist, :measwnoise)
        end
        imgax = CM.Axis(fig[1:length(resids),2], aspect=1, xticklabelsvisible=false, yticklabelsvisible=false, xticksvisible =false, yticksvisible=false)
        _imgviz!(fig, imgax, intensitymap(skymodel(post, curr), skym.grid))
        CM.text!(imgax,μas2rad(50.0), μas2rad(50.0); text="M/D :$(round(rad2μas(curr.sky.m_d),digits=2))", fontsize=32, color=:white)
        CM.text!(imgax,μas2rad(50.0), μas2rad(30.0); text="θs :$(round(curr.sky.θs,digits=2))", fontsize=32, color=:white)
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

xopt = ((rng, post) -> begin 
    temp = prior_sample(rng, post)
    truth = temp 
    #truth = (m_d = μas2rad(4.0), spin = -0.5, θo = 17.0, rpeak = 4.5, p1 = 2.5, p2 = 4.5, χ = -π/1.5, ι = π/4, βv = 0.6, spec = 1.0, η = π/2, pa=-72*π/180, σimg=0.01, ν = 2.0, ρpr = 10.0)#,frac = 0.1)
    #truth = (m_d = μas2rad(4.0), spin = 0.9, θo = 163.0, σimg = 1e-1, rpeak= 3.0, p1 =2.0, p2=2.0, χ = π/1.5, ι = π/4, βv = 0.6, spec = 1.0, η = π/2, pa=180-72., ν = 2.0, ρpr = 5.0, θs=80.0)#,frac = 0.1)
    #truth = (m_d = μas2rad(3.5))#,frac = 0.1)
    truth = (m_d = μas2rad(4.0), θo = 163.0, σimg = 1e-1, pa = 108.0, rpeak=3.0)#,frac = 0.1)



    #truth = (m_d = μas2rad(2.5), spin = -0.5, θo = 17.0, rpeak = 1.5, p1 = 3.0, p2 = 2.0, χ =-π/2, ι = π/3, βv = 0.5, spec = 1.0, η = π/2, pa=-72*π/180, σimg=1.00)#, θs=80,)# frac=0.5)
    #truth = (m_d_x_rpeak = μas2rad(3.5), m_d_d_rpeak=μas2rad(3.5/1.5), spin = -0.5, θo = 17.0, p1 = 3.0, p2 = 2.0, χ =-π/2, ι = π/3, βv = 0.5, spec = 1.0, η = π/2, pa=-72*π/180, σimg=1.00)#, θs=80,)# frac=0.5)
    for i in keys(truth)
        if i in keys(temp.sky)
            @reset temp.sky[i] = truth[i]
        end
    end
    #@reset temp.sky.σimg = 0.10
    return temp
end)(rng, post)


m = skymodel(post, xopt)
newgrid = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
ComradeBase.intensitymap(m, newgrid)
img = imageviz(ComradeBase.intensitymap(m, newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))
display(img)
#end
imageviz(ComradeBase.intensitymap(m, newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))
#baselineplot(dcphase, :uvdist, :measwnoise, error=true, label="Closure Phases")
#baselineplot(dlcamp, :uvdist, :measwnoise, error=true, label="Visibility Amplitudes")
#rad2μas(sqrt(xopt.sky.m_d_d_rpeak * xopt.sky.m_d_x_rpeak))
imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))
#end

#@profview Comrade.LogDensityProblems.logdensity_and_gradient(fpost, Comrade.inverse(fpost,xopt))

xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(lr); initial_params=xopt, maxiters=200, g_tol=1e-1, callback=Callback(10,()->nothing))

Comrade.residual(post, xopt)

imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))


newgrid = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid),colorscale=log, colorrange=(1e-3, 1e0))
imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, xopt),μas2rad(20/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, xopt),μas2rad(5/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))

using Pathfinder
result = pathfinder(fpost; init=Comrade.inverse(fpost,xopt))#, ndraws_elbo=200, ntries=5)
inv_metric = result.fit_distribution_transformed.Σ
init_params = result.draws[:, 1]
transform(fpost,init_params)
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost,init_params)), newgrid))
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost,init_params)), newgrid),colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, transform(fpost, init_params)),μas2rad(20/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, transform(fpost, init_params)),μas2rad(5/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))

xopt
transform(fpost,init_params)

residual(post, transform(fpost,init_params))
i = 2
σ2 = result.fit_distribution.Σ[i,i]
ave = result.fit_distribution.μ[i]
tempt = deepcopy(init_params)
transform(fpost,tempt)

spinvals = [begin tempt[i] = j; values(transform(fpost, tempt).sky)[i] end for j in -5:0.01:5]

CM.scatter(spinvals, [exp(-(i-ave)^2/σ2) for i in -5:0.01:5])

using AdvancedHMC
#metric = DenseEuclideanMetric(Matrix(inv_metric))
metric = DiagEuclideanMetric(diag(inv_metric))
integrator = Leapfrog(0.01)#find_good_stepsize(Hamiltonian(metric, fpost), init_params))
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator); 
                         init_buffer=100, term_buffer=200)
smplr = HMCSampler(kernel, metric, adaptor)
#out=joinpath(dirname(@__DIR__),"Results_non_diagonal_metric_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
out=joinpath((@__DIR__),"Results_non_diagonal_metric_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$(year)_phase_wrapped")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
println("starting")



#sample(rng, post, smplr, 120_000; n_adapts=15000, initial_params=transform(fpost, init_params), progress=true)
chain = sample(rng, post, smplr, 120_000; n_adapts=15_000, saveto=DiskStore(mkpath(out), 10), initial_params=transform(fpost, init_params), restart=true)

# Import data from disk
using Serialization
run_name = joinpath((@__DIR__),"Results_non_diagonal_metric_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$(year)_phase_wrapped")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")

chain = begin
    temp = load_samples(joinpath(dirname(@__DIR__), run_name),)
    temp#[1:500:end]
    #temp[Int(length(temp)/2):end]
end
checkpoints = deserialize(joinpath(run_name, "checkpoint.jls"))
checkpoints.state[3].state[2].metric.M⁻¹ |> findmax 
checkpoints.state[3].state[2].metric.M⁻¹ |> findmin

post = checkpoints.pt.c.rf.xform.model.logdensity.lpost
fpost = asflat(post)
fchain = Comrade.inverse.(Ref(fpost), chain)

CM.lines([i[1] for i in fchain])
msamples = skymodel.(Ref(post), chain)
stats= samplerstats(chain)

CM.lines((chain.sky.spin))
m_d, rpeak = begin
    m_d = :m_d in keys(chain[1].sky) ? rad2μas.(chain.sky.m_d) : rad2μas.(sqrt.(chain.sky.m_d_x_rpeak .* chain.sky.m_d_d_rpeak))
    rpeak = :rpeak in keys(chain[1].sky) ? (chain.sky.rpeak) : sqrt.(chain.sky.m_d_x_rpeak ./ chain.sky.m_d_d_rpeak)
    (m_d, rpeak)
end
#m_d = rad2μas.(chain.sky.m_d)
#rpeak = sqrt.(chain.sky.m_d_x_rpeak ./ chain.sky.m_d_d_rpeak)
#rpeak = chain.sky.rpeak
CM.lines(m_d)
MCMCChains.autocor(m_d, 1:15_000) |> CM.scatter
 
xopt = chain[end]
using Accessors
using StatsBase
using MCMCChains
temp = chain[end]

CM.set_theme!(CM.theme_dark())
CM.lines(stats.log_density)

hpd(Chains(rad2μas.(chain.sky.θo)))
hpd(Chains((chain.sky.θo)))
CM.lines(m_d)
CM.lines(moving_average(m_d, 1_00))
CM.lines((chain.sky.spin))
CM.lines(rpeak)
CM.lines(chain.sky.pa)
CM.lines((chain.sky.θo))
CM.lines((chain.sky.θs))
CM.lines((chain.sky.χ[5_000:end]))
CM.lines((chain.sky.νpr[5_000:end]))
CM.lines((chain.sky.ρpr[5_000:end]))
CM.lines((chain.sky.σimg[5_000:end]))
CM.lines((chain.sky.p1[5_000:end]))
CM.lines((chain.sky.p2[5_000:end]))
CM.lines((chain.sky.βv[5_000:end]))

CM.hist(m_d[15_000:end])
CM.hist(chain.sky.θo[15_000:end])
CM.hist((chain.sky.spin))
CM.hist((chain.sky.χ[15_000:end]))
CM.hist(rpeak[15_000:end])


for j in 1:length(chain.sky.c1[1])
    CM.lines([chain.sky.c1[i][j] for i in 5000:length(chain.sky.c1)]) |> display
end

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
CM.lines([(mean([i[55] for i in chain.sky.c1][120+i:140+i])) for i in 1:(length(m_d)-140)])
begin
    fig = CM.Figure();
    a, b, c, d = rand(3:dimension(post), 4)
    CM.lines!(CM.Axis(fig[1, 1]), [i[a] for i in chain.sky.c1])
    CM.lines!(CM.Axis(fig[1, 2]), [i[b] for i in chain.sky.c1])
    CM.lines!(CM.Axis(fig[2, 1]), [i[c] for i in chain.sky.c1])
    CM.lines!(CM.Axis(fig[2, 2]), [i[d] for i in chain.sky.c1])
    display(fig)
end
CM.scatter(m_d[45_000:end],([i[rand_dir] for i in chain[45_000:end].sky.c1]), alpha=0.5)
CM.scatter(([i[1000] for i in chain[45_000:end].sky.c1]),([i[1200] for i in chain[45_000:end].sky.c1]), alpha=0.5)
mvs = CM.scatter(m_d[5_000], (chain.sky.spin[5_000]), alpha=0.5)
mvs.axis.xlabel = "Mass (μas)"
mvs.axis.ylabel = "Spin"
CM.xlims!(mvs.axis, (2.0,6.0))
CM.ylims!(mvs.axis, (0.0,1.0))


for i in 5_000:1000:length(m_d)
    #mvs = CM.scatter(m_d[5_000:end], (chain.sky.spin[5_000:end]), alpha=0.5)
    CM.scatter!(mvs.axis, m_d[i], (chain.sky.spin[i]), alpha=0.5)
    #CM.lines!(mvs.axis, [3.6 for i in 1:10], range(-1,1,length=10), color=:red, linestyle=:dash)
    #CM.lines!(mvs.axis, range(2.0,6.0,length=10), [-0.5 for i in 1:10], color=:red, linestyle=:dash)
    #CM.xlims!(mvs.axis, (2.0,6.0))
    #CM.ylims!(mvs.axis, (0.0,1.0))
    #mvs.axis.xlabel = "Mass (μas)"
    #mvs.axis.ylabel = "Spin"
    mvs |>display
end

mvs = CM.scatter(m_d[5_000], (chain.sky.rpeak[5_000]), alpha=0.5)
mvs.axis.xlabel = "Mass (μas)"
mvs.axis.ylabel = "rpeak"
CM.xlims!(mvs.axis, (3.0,4.0))
CM.ylims!(mvs.axis, (1.0,8.0))

for i in 5_000:1000:length(m_d)
    #mvs = CM.scatter(m_d[5_000:end], (chain.sky.spin[5_000:end]), alpha=0.5)
    CM.scatter!(mvs.axis, m_d[i], (chain.sky.rpeak[i]), alpha=0.5)
    #CM.lines!(mvs.axis, [3.6 for i in 1:10], range(-1,1,length=10), color=:red, linestyle=:dash)
    #CM.lines!(mvs.axis, range(2.0,6.0,length=10), [-0.5 for i in 1:10], color=:red, linestyle=:dash)
    #CM.xlims!(mvs.axis, (2.0,6.0))
    #CM.ylims!(mvs.axis, (0.0,1.0))
    #mvs.axis.xlabel = "Mass (μas)"
    #mvs.axis.ylabel = "Spin"
    mvs |>display
end
MCMCChains.autocor(chain.sky.m_d[5_000:end], [50,100,1_000, 10_000])
MCMCChains.autocor(chain.sky.spin[5_000:end], [50,100,1_000, 10_000])
MCMCChains.autocor(chain.sky.χ[5_000:end], [50,100,1_000, 10_000])
MCMCChains.autocor(chain.sky.βv[5_000:end], [50,100,1_000, 10_000])
ess(chain.sky.m_d[5_000:end])
ess(chain.sky.spin[5_000:end])
begin 
    mvs = CM.scatter(m_d[45_000:end], rpeak[45_000:end], alpha=0.5)
    mvs.axis.xlabel = "Mass (μas)"
    mvs.axis.ylabel = "rpeak (GM/c²)"
    mvs
end

cor(m_d, rpeak)

CM.scatter(m_d[45_000:end], (chain.sky.χ[45_000:end]), alpha=0.5)
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

newgrid = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
#imgs = intensitymap.(msamples, Ref(skym.grid))

imgs = intensitymap.(smoothed.(msamples, μas2rad(5/(2.355))), Ref(newgrid))#skym.grid))
fig = CM.Figure();
#CM.image!(CM.Axis(fig[1,1], xreversed=true, aspect=1), intensitymap(smoothed(msamples[end], μas2rad(10/(2.355))), newgrid), colormap=:afmhot)
#CM.image!(CM.Axis(fig[1,1], xreversed=false, aspect=1), intensitymap(msamples[1], newgrid), colormap=:afmhot)
curr_theme = CM.Theme(
    Axis=(
        xgridvisible=false,
        ygridvisible=false,
        xspinesvisible=false,
        yspinesvisible=false,
        yticklabelsvisible=false,
        yticksvisible=false,
        #ylabelfont="Computer Modern Serif"
        #xlabelvisible=false,
        #ylabelvisible=false,
    ),
    Axis3=(
        xgridvisible=false,
        ygridvisible=false,
        zgridvisible=false,
        xspinesvisible=false,
        yspinesvisible=false,
        zspinesvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false,
        zticklabelsvisible=false,
        xticksvisible=false,
        yticksvisible=false,
        zticksvisible=false,
        xlabelvisible=false,
        ylabelvisible=false,
        zlabelvisible=false,
    ),
    #backgroundcolor = GLMk.colorant"rgba(10%, 10%, 10%, 1.0)"
)

#CM.set_theme!(merge(curr_theme, CM.theme_latexfonts()))
CM.set_theme!(CM.theme_dark())

include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
fig
rand(imgs) |> imageviz  
mimg = mean(imgs) 
simg = std(imgs)
fig = CM.Figure(;resolution=(700, 700));
begin
    axs = [CM.Axis(fig[i, j], xreversed=true, aspect=1) for i in 1:2, j in 1:2]
    CM.image!(axs[1,1], mimg, colormap=:inferno); begin a = axs[1,1]; add_scalebar!(a, mimg, fieldofview(mimg).X / 4, :white); a.title="Mean Image";a.titlesize=20;end
    CM.image!(axs[1,2], simg./(max.(mimg, 1e-2)), colorrange=(0.0, 1.0), colormap=:inferno); begin a = axs[1,2]; add_scalebar!(a, simg, fieldofview(simg).X / 4, :white);a.title="Fractional Std. Dev";a.titlesize=20;end
    imgs1, imgs2 = rand(imgs), rand(imgs);
    CM.image!(axs[2,1], imgs1, colormap=:inferno); begin a = axs[2,1]; add_scalebar!(a, imgs1, fieldofview(imgs1).X / 4, :white);a.title="Random Draw 1.";a.titlesize=20;end
    CM.image!(axs[2,2], imgs2, colormap=:inferno);begin a = axs[2,2]; add_scalebar!(a, imgs2, fieldofview(imgs2).X / 4, :white);a.title="Random Draw 2.";a.titlesize=20;end
    CM.hidedecorations!.(axs)
    CM.save(joinpath((@__DIR__),"sample_images_2017.pdf"), fig)
    #for (i,img) in enumerate(imgs)
    #    save_fits(joinpath("/n/home06/dochang/KerrGMRF/src/post_draws", "img_$(i).fits"), img)
    #end
end
fig
import ColorSchemes
begin
    fig = CM.Figure(size = (600, 1210))#, 600))
    ax2017_consensus = CM.Axis(fig[3:4, 1:2]; aspect = 1, xreversed = true)
    jk= CM.Axis(fig[1, 1]; aspect = 1, xreversed = true)
    kg = CM.Axis(fig[1, 2]; aspect = 1, xreversed = true)
    jkblurred = CM.Axis(fig[2, 1]; aspect = 1, xreversed = true)
    kgblurred = CM.Axis(fig[2, 2]; aspect = 1, xreversed = true)
#
    CM.image!(ax2017_consensus, eht_2017_consensus, colormap = ColorSchemes.inferno)
    CairoMakie.text!(
        ax2017_consensus,
        0.475fovx,
        0.4fovy;
        text = "EHT Consensus 2017",
        color = ColorSchemes.inferno[end],
        fontsize = 45,
    )
    image!(jk, img2017, colormap = ColorSchemes.inferno)
    CairoMakie.text!(
        jk,
        0.475fovx,
        0.35fovy;
        text = "Dual-Cone ",
        color = ColorSchemes.inferno[end],
        fontsize = 35,
    )
    image!(kg, img, colormap = ColorSchemes.inferno)
    text!(kg, 0.475fovx, 0.35fovy; text = "Snapshot Model", color = ColorSchemes.inferno[end], fontsize = 35)
    image!(jkblurred, img2017blurred, colormap = ColorSchemes.inferno)
    CairoMakie.text!(
        jkblurred,
        0.475fovx,
        0.35fovy;
        text = "Blurred",
        color = ColorSchemes.inferno[end],
        fontsize = 35,
    )
    CairoMakie.text!(
        jkblurred,
        0.475fovx,
        0.23fovy;
        text = "Dual-Cone",
        color = ColorSchemes.inferno[end],
        fontsize = 35,
    )
    image!(kgblurred, imgblurred, colormap = ColorSchemes.inferno)
    CairoMakie.text!(
        kgblurred,
        0.475fovx,
        0.35fovy;
        text = "Blurred",
        color = ColorSchemes.inferno[end],
        fontsize = 35,
    )
    CairoMakie.text!(
        kgblurred,
        0.475fovx,
        0.23fovy;
        text = "Snapshot Model ",
        color = ColorSchemes.inferno[end],
        fontsize = 35,
    )
    #Base.get_extension(VLBISkyModels, :VLBISkyModelsMakieExt).
    add_scalebar!(jk, img2017, μas2rad(40), ColorSchemes.inferno[end])
    add_scalebar!(kg, img, μas2rad(40), ColorSchemes.inferno[end])
    add_scalebar!(jkblurred, img2017blurred, μas2rad(40), ColorSchemes.inferno[end])
    add_scalebar!(kgblurred, imgblurred, μas2rad(40), ColorSchemes.inferno[end])
    add_scalebar!(
        ax2017_consensus,
        eht_2017_consensus,
        μas2rad(40),
        ColorSchemes.inferno[end],
    )
    #scalebar!(ax2018_consensus,40u"mm"; strokecolor=:white)#, ColorSchemes.inferno[end])
    for i = 1:1#2
        colgap!(fig.layout, i, 10)
    end
    for i = 1:2
        rowgap!(fig.layout, 1, 10)
    end
    display(fig)
end
save(joinpath((@__DIR__), "jk_kg_comp.pdf"), fig)
#

fig
using Plots
p = Plots.plot(layout=(1,2));
for s in sample(chain[30_000:end], 100)
    residual!(post, s)
end
residual(post,chain[end])
p
chi2vals =[chi2(post, chain[i]) ./ (length(dlcamp), length(dcphase)) for i in 47000:49000]
CM.scatter(chi2vals )

intensitymap(skymodel(post, chain[1000]), newgrid) |> imageviz
intensitymap(skymodel(post, chain[end]), newgrid) |> imageviz

newgrid = imagepixels(fovx, fovy, 2npix, 2npix; executor=ThreadsEx())
fig = CM.Figure()
ax1 = CM.Axis(fig[1, 1], title="log density")
ax2 = CM.Axis(fig[2, 1], title="step size", yscale=log10)
ax3 = CM.Axis(fig[1:2, 2], xreversed=true, yticklabelsvisible=false, xticklabelsvisible=false, aspect=1)

for i in 1:100:40000
    CM.empty!(ax1)
    CM.empty!(ax2)
    CM.empty!(ax3)
    CM.lines!(ax1, stats.log_density[begin:i])
    CM.lines!(ax2, stats.step_size[begin:i])
    CM.image!(ax3, intensitymap(skymodel(post, chain[i]), newgrid), colormap=:afmhot)
    sleep(0.1)
    display(fig)
end
temp = chain[end-100]
@reset temp.sky.σimg = 0.01
intensitymap(skymodel(post, temp), newgrid) |> imageviz
intensitymap(smoothed(skymodel(post, temp), μas2rad(20.0)/(2√2*log(2))), newgrid) |> imageviz


c = chain.sky[end-500:end]
mt1 = mean(skym.metadata.transform1.(c.c1, c.ρpr, c.νpr)) 
t1 = CM.image(mt1)
CM.Colorbar(t1.figure[1,2], t1.plot)
t1

st1 = std(skym.metadata.transform1.(c.c1, c.ρpr, c.νpr)) 
t2 = CM.image(st1)
CM.Colorbar(t2.figure[1,2], t2.plot)
t2


for i in 5_000:100:length(chain)
    #img = imageviz(intensitymap(smoothed(skymodel(post, chain[i]), μas2rad(7/(2*√(2*log(2))))), skym.grid), size=(500, 400))
    img = imageviz(intensitymap(skymodel(post, chain[i]), newgrid), size=(500, 400))
    println(i)
    sleep(0.1)
    display(img)
end

using PairPlots
using LaTeXStrings

postsamps = Comrade.postsamples(chain[5_000:end])
ks = [:m_d, :spin, :θo,:rpeak, :χ]#, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]
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