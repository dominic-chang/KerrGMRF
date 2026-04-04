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
include(joinpath(dirname(@__DIR__),"plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__),"modifiers.jl"))
include(joinpath(dirname(@__DIR__),"models","KerrGMRF_shared_variability.jl"))

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

using VLBIImagePriors
using Distributions, DistributionsAD

function ModifiedKerrGMRF(θ2, meta) 
    m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa), Shift(μas2rad(12.0), μas2rad(12.0)))
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
using LaTeXStrings
function _imgviz!(
        fig, ax, img::IntensityMap{<:Real}; scale_length = fieldofview(img).X / 4,
        kwargs...
    )
    colorrange_default = (minimum(img), maximum(img))
    dkwargs = Dict(kwargs)
    crange = get(dkwargs, :colorrange, colorrange_default)
    delete!(dkwargs, :colorrange)
    cmap = get(dkwargs, :colormap, :inferno)
    delete!(dkwargs, :colormap)

    hm = CM.heatmap!(ax, img; colorrange = crange, colormap = cmap, dkwargs...)
    CM.rotate!(hm, -ComradeBase.posang(axisdims(img)))

    color = CM.Makie.to_colormap(cmap)[end]
    add_scalebar!(ax, img, scale_length, color)

    CM.Colorbar(fig[1:num_data_prods, 3], hm; label = "Brightness (Jy/μas²)", tellheight = true)
    CM.colgap!(fig.layout, 15)

    x1, y1 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), first(img.Y))
    x2, y2 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), last(img.Y))
    x3, y3 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), last(img.Y))
    x4, y4 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), first(img.Y))

    xl = min(x1, x2, x3, x4)
    xu = max(x1, x2, x3, x4)
    yl = min(y1, y2, y3, y4)
    yu = max(y1, y2, y3, y4)

    # Flip x l and u for astronomer conventions
    CM.xlims!(ax, (xu, xl))
    CM.ylims!(ax, (yl, yu))
    CM.trim!(fig.layout)

    return CM.Makie.FigureAxisPlot(fig, ax, hm)
end

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

mutable struct Callback 
    counter::Int
    stride::Int
    const f::Function
end
Callback(stride, f) = Callback(0, stride, f)
loss_arr = []
fig = CM.Figure(;size=(1200, 600));
num_data_prods= length(post.data)
function (c::Callback)(state, loss, others...)
    global loss_arr
    global fpost
    global st
    c.counter += 1
    curr = transform(fpost, state.u)
    CM.empty!(fig)
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
        CM.text!(imgax,μas2rad(50.0), μas2rad(30.0); text="θo :$(round(curr.sky.θs,digits=2))", fontsize=32, color=:white)
        display(fig)
        m_d = begin
            if :m_d in keys(curr.sky)
                rad2μas.(curr.sky.m_d)
            else
                rad2μas.(sqrt.(curr.sky.m_d_x_rpeak .* curr.sky.m_d_d_rpeak))
            end
        end
        @info "On step $(c.counter) mass=$(m_d) inc=$(curr.sky.θo), spin=$(curr.sky.spin), flux_rat=$(curr.sky.frac), σimg=$(curr.sky.σimg), pa=$(unsafe_trunc(Int, curr.sky.pa*180/π)), θs=$(curr.sky.θs), p2=$(curr.sky.p2)"
        return false
    else
        return false
    end
end

m = skymodel(post, xopt)
newgrid = imagepixels(fovx, fovy, npix, npix; executor=ThreadsEx())
ComradeBase.intensitymap(m, newgrid)
img = imageviz(ComradeBase.intensitymap(m, newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))
display(img)
imageviz(ComradeBase.intensitymap(m, newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))
xopt =chain[begin]
imageviz(ComradeBase.intensitymap(skymodel(post, chain[begin]), newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))

residual(post, chain[end])
xopt, sol = comrade_opt(post, OptimizationOptimisers.Adam(lr); initial_params=xopt, maxiters=500, g_tol=1e-1, callback=Callback(10,()->nothing))


xopt=transform(fpost, Comrade.inverse(fpost, xopt))#chain[end]))
@reset xopt.sky.σimg = 1e0
@reset xopt.sky.m_d = μas2rad(3.8)
#
begin
	for dir in 1:13
		logdensity = []
		grads = []
		xvals = []
		#dir = 12
		symbol = keys(prior)[dir]
		println(symbol)
		for i in range(-1.0, 1.0, length = 100)
			sample = Comrade.inverse(fpost, xopt)
			sample[dir] += i
			l, grad=Comrade.LogDensityProblems.logdensity_and_gradient(fpost, sample)
			append!(xvals, Comrade.transform(fpost, sample).sky[symbol])
			append!(logdensity, l)
			append!(grads, grad[dir])
		end
        if dir == 1 
            xvals .= rad2μas.(xvals)
            CM.vlines([rad2μas(xopt.sky[symbol])], label = "truth")
        else
		    CM.vlines([xopt.sky[symbol]], label = "truth")
        end
#
		fig = CM.scatter(xvals, logdensity, label = "log_density")
		CM.scatter!(xvals, grads, label = "gradient")
		display(fig)
	end
end

imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid))#,colorscale=log, colorrange=(1e-10, 1e-2))
diffvals =collect((values(xopt.sky)[begin:18] .- values(chain[begin].sky)[begin:18]) ./ values(chain[begin].sky)[begin:18])
CM.scatter(diffvals[begin:14])
diffvals =collect((values(xopt.sky)[19] .- values(chain[begin].sky)[19]) ./ values(chain[begin].sky)[19])
CM.scatter(reshape(diffvals, 40*40))
diffvals =collect((values(xopt.sky)[20] .- values(chain[begin].sky)[20]) ./ values(chain[begin].sky)[20])
CM.scatter(reshape(diffvals, 40*40))
chain[begin].sky
xopt.sky


newgrid = imagepixels(fovx, fovy, npix*3, npix*3; executor=ThreadsEx())
imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid),colorscale=log, colorrange=(1e-3, 1e0))
imageviz(ComradeBase.intensitymap(skymodel(post, xopt), newgrid))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, xopt),μas2rad(20/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, xopt),μas2rad(5/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))

using Pathfinder
result = pathfinder(fpost; init=Comrade.inverse(fpost,xopt), ndraws_elbo=100, ntries=1_00)
inv_metric = result.fit_distribution_transformed.Σ
init_params = result.draws[:, 1]
transform(fpost,init_params)
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost,init_params)), newgrid))
imageviz(ComradeBase.intensitymap(skymodel(post, transform(fpost,init_params)), newgrid),colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, transform(fpost, init_params)),μas2rad(20/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))
imageviz(ComradeBase.intensitymap(smoothed(skymodel(post, transform(fpost, init_params)),μas2rad(5/(2.355))), newgrid), colormap=:afmhot)#,colorscale=log, colorrange=(1e-7, 1e-3))

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
out=joinpath(dirname(@__DIR__),"Results_non_diagonal_metric_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
println("starting")
chain = sample(rng, post, smplr, 120_000; n_adapts=500, saveto=DiskStore(mkpath(out), 10), initial_params=transform(fpost, init_params), restart=true)

function temp(fpost, xopt) 
    for _ in 1:1_000
        Comrade.LogDensityProblems.logdensity_and_gradient(fpost, Comrade.inverse(fpost, xopt))
    end
end
@profview temp(fpost, xopt) 


#chain = sample(rng, post, smplr, 20_000; n_adapts=10_000, saveto=DiskStore(mkpath(out), 10), initial_params=xopt, restart=true)

#chain = sample(post, AdvancedHMC.NUTS(0.8), 20_000; n_adapts=10_000, progress=true, initial_params=xopt, saveto=DiskStore(;name=joinpath(dirname(@__DIR__),"Results"), stride=10), restart=true);#;name=joinpath((dirname(@__DIR__),"Results_smaller_res"))));

# Import data from disk
using Serialization
run_name = "Results_non_diagonal_metric_grmhd_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(rad2μas(fovx)))_fov_year_$(year)"
#run_name = "Results_non_diagonal_metric_50_rast_low_res"
out = "/n/home06/dochang/KerrGMRF/src/dual_cone_2017/Results_non_diagonal_metric_dual_cone_40_180_rast_40_res_150_fov_year_2017"
#out=joinpath((@__DIR__),run_name)
chain = begin
    temp = load_samples(joinpath(dirname(@__DIR__), out))#[10_000:end]
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
msamples = skymodel.(Ref(post), chain[45_000:end])
stats= samplerstats(chain)

rand_dir = rand(1:length(cprior1))
CM.lines([i[rand_dir] for i in fchain], linewidth=0.5)
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
MCMCChains.autocor(m_d[1:100:end], 1:100) |> CM.scatter

xopt = chain[end]
using Accessors
using StatsBase
temp = chain[end]
@reset temp.sky.frac = 0.9
@reset temp.sky.m_d = μas2rad(3.0)
intensitymap(skymodel(post,temp), skym.grid) |> imageviz

CM.lines(stats.log_density[15_000:end])

Chains(chain.sky.m_d[40_000:end])
hpd(Chains(rad2μas.(chain.sky.θo)))
hpd(Chains((chain.sky.θo[30_000:end])))
CM.lines(m_d[45_000:end])
CM.lines(moving_average(m_d[30_000:end], 50))
CM.lines((chain.sky.spin[45_000:end]))
CM.lines(rpeak[45_000:end])
CM.lines((chain.sky.frac))
CM.lines(((π .-chain.sky.pa) .* 180/π))
CM.lines((chain.sky.θo))
CM.lines((chain.sky.θs))
CM.lines((chain.sky.χ))
CM.lines((chain.sky.νpr))
CM.lines((chain.sky.ρpr))
CM.lines((chain.sky.σimg))
CM.lines((chain.sky.p1))
CM.lines((chain.sky.p2))
CM.lines((chain.sky.βv))

CM.hist(m_d[30_000:end])
CM.hist(chain.sky.θo[45_000:end])
CM.hist((chain.sky.spin[45_000:end]))
CM.hist((chain.sky.χ[45_000:end]))
CM.hist(rpeak[45_000:end])


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
begin 
    mvs = CM.scatter(m_d[45_000:end], (chain.sky.spin[45_000:end]), alpha=0.5)
    CM.lines!(mvs.axis, [3.6 for i in 1:10], range(-1,1,length=10), color=:red, linestyle=:dash)
    CM.lines!(mvs.axis, range(2.0,6.0,length=10), [-0.5 for i in 1:10], color=:red, linestyle=:dash)
    CM.xlims!(mvs.axis, (2.0,6.0))
    CM.ylims!(mvs.axis, (0.0,1.0))
    mvs.axis.xlabel = "Mass (μas)"
    mvs.axis.ylabel = "Spin"
    mvs
end
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

newgrid = imagepixels(fovx, fovy, 200, 200; executor=ThreadsEx())
#imgs = intensitymap.(msamples, Ref(skym.grid))

imgs = intensitymap.(msamples, Ref(newgrid))#skym.grid))
fig = CM.Figure();
#CM.image!(CM.Axis(fig[1,1], xreversed=true, aspect=1), intensitymap(smoothed(msamples[end], μas2rad(10/(2.355))), newgrid), colormap=:afmhot)
#CM.image!(CM.Axis(fig[1,1], xreversed=false, aspect=1), intensitymap(msamples[1], newgrid), colormap=:afmhot)
fig
rand(imgs) |> imageviz  
mimg = mean(imgs)
simg = std(imgs)
fig = CM.Figure(;resolution=(700, 700));
axs = [CM.Axis(fig[i, j], xreversed=true, aspect=1) for i in 1:2, j in 1:2]
CM.image!(axs[1,1], mimg, colormap=:afmhot); axs[1, 1].title="Mean"
CM.image!(axs[1,2], simg./(max.(mimg, 1e-1)), colorrange=(0.0, 2.0), colormap=:afmhot);axs[1,2].title = "Std"
CM.image!(axs[2,1], rand(imgs),   colormap=:afmhot);
CM.image!(axs[2,2], rand(imgs),   colormap=:afmhot);
CM.hidedecorations!.(axs)
fig
p = Plots.plot(layout=(1,2));
for s in sample(chain[45000:end], 100)
    residual!(post, s)
end
residual(post,chain[begin])
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

postsamps = Comrade.postsamples(chain[45_000:end])
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

b = sqrt(27)

q = [2*b^2, 0.0]
p = [-b^2, 0.0]

pow(x,y) = x^y

function c_m(x, y) 
    return [x[1] * y[1] - x[2] * y[2], x[1] * y[2] + x[2] * y[1]]
    end

function c_d(x, y) 
    return c_m(x, [y[1], -y[2]]) / (pow(y[1], 2.) + pow(y[2], 2.));
end
function c_pow(x, y) 
    mag = pow(sqrt(x[1] * x[1] + x[2] * x[2]), y)
    angle = y * (atan(x[2], x[1]))
    return [mag * cos(angle), mag * sin(angle)]
end
C = c_pow([q[1]*q[1], 0.] / 4. + [p[1]*p[1]*p[1], 0.] / 27., 0.5);
C1 = c_pow(-q / 2. + C, 1. / 3.)
C2 = c_m(C1,[-1/2.,sqrt(3)/2.])#c_pow(-q / 2. + c_m(C, [-(1. / 2.), sqrt(3)]/ 2.), 1. / 3.);
C3 = c_m(C1,[-1/2.,-sqrt(3)/2.])#c_pow(-q / 2. + c_m(C, [-(1. / 2.), -sqrt(3)]/ 2.), 1. / 3.);

v1 = C2 - c_d(p, 3. * C2)
v3 = C3 - c_d(p, 3. * C3)
v4 = C1 - c_d(p, 3. * C1)

u1 = c_pow(-q / 2. + c_pow(c_pow(q, 2.) / 4. + c_pow(p, 3.) / 27., 0.5), 1. / 3.);
u2 = c_pow(-q / 2. - c_pow(c_pow(q, 2.) / 4. + c_pow(p, 3.) / 27., 0.5), 1. / 3.);

e1 = [-(1.), sqrt(3)]
e2 = [-(1. ), -sqrt(3)]
c_m(e1,u1)/2. + c_m(e2,u2)/2.
c_m(e2,u1)/2. + c_m(e1,u2)/2.
u1 + u2


