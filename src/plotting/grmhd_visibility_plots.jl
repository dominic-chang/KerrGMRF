using Pkg;
Pkg.activate(dirname(dirname(@__DIR__)));
using Comrade
using Pyehtim
using Krang
using StableRNGs
using Accessors
import CairoMakie as CM
using BasicInterpolators
using LinearAlgebra
using FINUFFT
using Enzyme
using Optimization, OptimizationOptimisers
using StableRNGs
using LaTeXStrings
using VLBIImagePriors
using Distributions, DistributionsAD
using Pathfinder
using AdvancedHMC
using Serialization
using MCMCChains
using StatsBase

LinearAlgebra.BLAS.set_num_threads(24) # to avoid threading conflicts with FINUFFT
rng = StableRNG(1234)

include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF.jl"))

function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa))#, Shift(μas2rad(12.0), μas2rad(12.0)))
	return RenormalizedFlux(m, θ2.f)
end

lr = 0.05
seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.01
bulkx = 1.0
bulky = 1.0
#bulkpix = 40
#raster_size = 100.0# in microarcseconds    
#snrcut = 3.0
#uv_min = 0.1e9
#fovx = μas2rad(120.0)
#fovy = μas2rad(120.0)
#npix = 40

bulkpix = 70#20
raster_size = 140.0# 90 # in microarcseconds    
fovx = μas2rad(140.0)
fovy = μas2rad(140.0)
npix = 70
year = 2017

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 => "L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "30nights_subchanneled_mergedobs_244GHz.uvfits")#/n/home06/dochang/KerrGMRF/data/sim140_frame0026_230.5_GHz_synthdata_ngEHTsim.uvfits")#"frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(dirname(@__DIR__)), "data", data[year])
img_path = joinpath(dirname(dirname(@__DIR__)), "data", "image_ma+0.5_1275_163_1_nall.h5")

# Get observer information
obsin = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min = 0.1e9).add_fractional_noise(fractional_noise)
inimg = ehtim.image.load_image(img_path)
inimg.rf, inimg.ra, inimg.dec = obsin.rf, obsin.ra, obsin.dec
inimg = inimg.rotate(π - 72*pi/180)
inimg.imvec *= 0.65/inimg.total_flux()
inimg.display(scale = "log")
inimg.blur_circ(μas2rad(7.0)).display()
inimg.mjd = 57848
obsin.mjd = 57848
obs = inimg.observe_same(obsin, ampcal = ampcal, phasecal = phasecal, add_th_noise = add_th_noise, seed = seed, ttype = "fast")
obs = scan_average(obs.flag_uvdist(uv_min = 0.1e9)).add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))
baselineplot(dvisamp, :uvdist, :measwnoise, error = true, label = "Visibility Amplitudes")

bulkgrid            = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor = ThreadsEx())
transform1, cprior1 = matern(size(bulkgrid))
transform2, cprior2 = matern(size(bulkgrid))
prior               = (
m_d = Uniform(μas2rad(1.0), μas2rad(8.0)),
spin = Uniform(0.01, 0.99),
θo = Uniform(120.0, 179.0),
θs = Uniform(40.5, 90.0),
rpeak = Uniform(1.0, 8.0),
p1 = Uniform(0.1, 5.0),
p2 = Uniform(1.0, 5.0),
χ = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
ι = Uniform(0, π/2),
βv = Uniform(0.01, 0.99),
spec = Uniform(-1.0, 5.0),
η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
frac = VLBIImagePriors.DeltaDist(1.0),
pa = VLBIImagePriors.DeltaDist(π-72*π/180),
f = VLBIImagePriors.DeltaDist(0.65),
σimg = truncated(Normal(0.0, 1.0); lower = 0.0),
ρpr = truncated(InverseGamma(1.0, -log(0.1)*10); lower = 1.0, upper = 2*max(size(bulkgrid)...)),
νpr = Uniform(1.0, 5.0),
c1 = cprior1,
c2 = cprior2
)
offset              = 0.0
skym                = SkyModel(
ModifiedKerrGMRF,
prior,
imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx());
metadata = (; bulkgrid, transform1, transform2, raster_size, offset)
)

post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode = set_runtime_activity(Enzyme.Reverse))
fpost = asflat(post)
run_name=joinpath("/n/home06/dochang/KerrGMRF/src/dual_cone_2017_phase_wrapped_grmhd", "Results_non_diagonal_metric_grmhd_dual_cone_$(bulkpix)_$(Int(raster_size))_rast_$(npix)_res_$(Int(floor(rad2μas(fovx))))_fov_year_$year")#Results_non_diagonal_metric_20_rast_extremely_low_res_pinned_frac")
out=joinpath((@__DIR__), run_name)
chain = load_samples(joinpath(dirname(dirname(@__DIR__)), run_name))#, 120_000:10:120_010)
checkpoints = deserialize(joinpath(out, "checkpoint.jls"))
checkpoints.state[3].state[2].metric.M⁻¹ |> findmax
checkpoints.state[3].state[2].metric.M⁻¹ |> findmin

post = checkpoints.pt.c.rf.xform.model.logdensity.lpost
fpost = asflat(post)
fchain = Comrade.inverse.(Ref(fpost), chain)


using PairPlots
postsamps = Comrade.postsamples(chain[2000:end])
ks = [:m_d, :spin, :θo, :θs, :χ, :νpr, :ρpr]#, :σimg]
map(x -> sin(x), getproperty.(Ref(postsamps.sky), :θs))
mat = begin
	mat = reduce(hcat, getproperty.(Ref(postsamps.sky), ks))
	mat[:, 1] .= rad2μas.(mat[:, 1])
	rh = 1 .+ .√(1 .- mat[:, 2] .^ 2)
	mat[:, 7] .= mat[:, 7] ./ (rad2μas(fovx) ./ (mat[:, 1] .* 5) ./ map(x -> sin(x*π/180), getproperty.(Ref(postsamps.sky), :θs)))
	mat[:, 7] .= mat[:, 7] ./ npix  .* (raster_size .- rh .* mat[:,1]) ./ (mat[:,1] )
	mat
end
table = NamedTuple{Tuple(ks)}([mat[:, i] for i in 1:length(ks)])
curr_theme = CM.Theme(
    font = "CMU Serif", fontsize = 40, margin=0,
    Axis=(
        xticklabelsize=30.,
        yticklabelsize=30.,
    )
    )
CM.set_theme!(
	merge(CM.theme_latexfonts(), curr_theme),
)

clrs = CM.Makie.wong_colors();
prior = post.prior.sky
fig = pairplot(
	#reduce(hcat,getproperty.(Ref(postsamps.sky), [:m_d, :spin, :θo,:rpeak, :χ, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]))=>
	table =>
		(
			#PairPlots.Contourf(; sigmas=0.5:0.5:2),
			PairPlots.Scatter(; color = clrs[1], markersize = 3.0, strokecolor = clrs[1], strokewidth = 0.5, alpha = 0.01),
			PairPlots.MarginHist(; color = clrs[1]),#, bandwidth=0.5),
			PairPlots.MarginQuantileText(;color=:black)
		),
	#PairPlots.Band(
	#	(; χ = (0.0, 1.0π),),
	#	color = :grey,
	#),
	#labels=[:m_d, :spin, :θo,:rpeak, :χ, :η, :ι, :βv, :p1, :p2, :spec, :νpr, :ρpr, :σimg]
	PairPlots.Truth(
		(; m_d = 3.83, spin = 0.5, θo = 163.00),
		linewidth = 3.0,
		linestyle = :dash,
		color = :black,
	),
	axis=(;
        m_d=(;
            ticks=([1.0, 3.0, 5.0, 7.0]),
            lims=(; low=rad2μas(prior.m_d.a), high=rad2μas(prior.m_d.b)),
        ),
        spin=(;
            #ticks=([-0.85, -0.45, -0.05]),
            lims=(; low=prior.spin.a, high=prior.spin.b),
        ),
        θo=(;
            lims=(; low=prior.θo.a, high=prior.θo.b),
            ticks=([130, 150, 170]),
        ),
        θs=(;
            lims=(; low=prior.θs.a, high=prior.θs.b),
            ticks=([45, 65, 85]),
        ),
        rpeak=(;
            lims=(; low=prior.rpeak.a, high=prior.rpeak.b),
            #ticks=([5, 10, 15]),
        ),
        χ=(;
            lims=(; low=-π, high=π),
        ),

    ),
	labels = Dict(:m_d => L"m{/}d\,(\mu as)", :χ => L"\chi\,(\text{rad.})", :spin => L"a", :θo => L"\theta_o\,({}^\circ)", :ρpr => L"\rho\,(G M/c^2)", :rpeak => L"R\,(G M/c^2)", :θs => L"\theta_s\,({}^\circ)", :νpr => L"\nu"),
);


markers = [
	CM.LineElement(color = :black, linestyle = :dash, linewidth = 5),
	CM.MarkerElement(color = clrs[1],marker=:circle),
]
labels = [CM.L"\text{Truth}", CM.L"\text{Samples}", ]

for i in 1:6
	CM.rowgap!(fig.layout, i, 10)
	CM.colgap!(fig.layout, i, 10)
end
fig

CM.save("GMHD_visibility.png", fig)