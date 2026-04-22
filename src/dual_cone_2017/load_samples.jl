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
using Serialization
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

out=joinpath((@__DIR__),"Results_non_diagonal_metric_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
chain = begin
    temp = load_samples(joinpath((@__DIR__), out))#[10_000:end]
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
msamples = skymodel.(Ref(post), chain[15_000:100:end])
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

CM.hist(m_d[15_000:end])
CM.hist(m_d[55_000:end])
CM.hist(chain.sky.θo[15_000:end])
CM.hist((chain.sky.spin[15_000:end]))
CM.hist((chain.sky.χ[15_000:end]))
CM.hist(rpeak[15_000:end])


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

imgs = intensitymap.(smoothed.(msamples, μas2rad(10.0/(2*√(2*log(2))))), Ref(newgrid))#skym.grid))
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


