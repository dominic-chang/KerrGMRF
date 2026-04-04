using VIDA
using Krang
using CairoMakie
using LaTeXStrings
using Comrade
using VLBIImagePriors
using Enzyme
using BasicInterpolators
using Images
using Accessors
using ImageIO
using FileIO
using Optimization
using OptimizationOptimisers
using Enzyme
using StableRNGs
using Plots: Plots
using LaTeXStrings
using Distributions, DistributionsAD

include(joinpath((@__DIR__), "utils.jl"))
include(joinpath((@__DIR__), "plotting", "utils.jl"))
include(joinpath((@__DIR__), "modifiers.jl"))
include(joinpath((@__DIR__), "models", "JuKeBOX.jl"))
include(joinpath((@__DIR__), "models", "KerrGMRF.jl"))

lr = 0.1
seed = 1234
phasecal = true
ampcal = true
add_th_noise = true
scan_avg = true
fractional_noise = 0.01
bulkx = 1.0
bulky = 1.0
#bulkpix = 100#20
bulkpix = 240#20
raster_size = 120.0# 90 # in microarcseconds    
snrcut = 3.0
uv_min = 0.1e9

year = 2017#"bhex"

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 => "L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(@__DIR__), "data", data[year])
files= "/n/holylabs/doeleman_lab/Users/dochang/GRMHDImages/snapshots"
filenames = filter(file->occursin("_10_",file), readdir(files))
spins_and_files=[]
#for fname in filenames
fname = filenames[1]
img_path = joinpath(files, fname)
#img_path = joinpath(dirname(@__DIR__), "data", "image_sa+0.94_981_163_160_nall.h5")#"image_ma+0.5_1275_163_1_nall.h5")
img_path = joinpath(dirname(@__DIR__), "data", "image_ma+0.5_1275_163_1_nall.h5")
#img_path = joinpath(dirname(@__DIR__), "data", "sim140_true.fits")#image_ma+0.5_1000_163_1_nall.h5")
#img_path = joinpath("/n/home06/dochang/KerrGMRF/data/sim140_true.fits")
#img_path = "/n/holylabs/doeleman_lab/Users/dochang/GRMHDImages/snapshots/image_ma+0.5_1000_163_10_nall.h5"
#img_path = joinpath(dirname(@__DIR__), "data", "image_ma+0.5_1445_163_40_nall.h5")



#inimg = VIDA.load_image(img_path)#
inimg = rotated(VIDA.load_image(img_path), π - 72/180*π)#, dims=(1,2))
npix = inimg.X |> length 
fovx, fovy = fieldofview(inimg)
#joinpath(dirname(@__DIR__), "smiley.png")
#smiley = map(
#    x->Float64(Gray(x).val),
#    FileIO.load(joinpath(dirname(@__DIR__), "smiley.png"))[Int.(floor.(1:2.5:1200)), Int.(floor.(1:2.5:1200))]
#)
#inimg .= smiley
imageviz(inimg, colorscale = log10, colorrange = (1e-6, 1e-3), colormap = :inferno)
inimg .= max.(inimg, 0.0)
inimg ./= flux(inimg)
bh = Bhattacharyya(inimg)
nx = NxCorr(inimg)

bulkgrid            = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor = Serial())
transform1, cprior1 = matern(size(bulkgrid))
transform2, cprior2 = matern(size(bulkgrid))
prior               = (
m_d = Uniform(μas2rad(1.0), μas2rad(8.0)),
spin = Uniform(0.01, 0.99),#-0.99, -0.01),
θo = Uniform(120, 179.0),
θs = Uniform(20, 90.0),#,VLBIImagePriors.DeltaDist(90.0),
rpeak = Uniform(1.0, 8.0),# Uniform(1.0, 8.0),
p1 = Uniform(0.1, 5.0),#Uniform(0.1, 5.0),
p2 = Uniform(1.0, 5.0),
χ =VLBIImagePriors.DiagonalVonMises(0, inv(π^2)), 
ι = Uniform(0, π/2),
βv = Uniform(0.01, 0.9),
spec = Uniform(-1.0, 5.0),
η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
frac = VLBIImagePriors.DeltaDist(1.0),#Uniform(0.0, 1.0),
pa = VLBIImagePriors.DeltaDist(π - 72*π/180),# Uniform(-π, 0),
f = VLBIImagePriors.DeltaDist(1.0),#Uniform(-π, π),
σimg = truncated(Normal(0.0, 1.0); lower = 0.0),
ρpr = truncated(InverseGamma(1.0, -log(0.1)*10); lower = 1.0, upper = 2*max(size(bulkgrid)...)),
νpr = Uniform(1.0, 5.0),
c1 = cprior1,
c2 = cprior2,
x = Uniform(μas2rad.((-1,1))...),
y = Uniform(μas2rad.((-1,1))...),
)
offset = 0.0
function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa))#, Shift(θ2.x, θ2.y))
	return RenormalizedFlux(m, θ2.f)
end
global metadata = (; bulkgrid, transform1, transform2, raster_size, offset)
using Pyehtim
obsin = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min = 0.1e9).add_fractional_noise(fractional_noise)
ehtimg = ehtim.image.load_image(img_path).rotate(π-72*π/180)
ehtimg.ivec.shape
ehtimg.rf = obsin.rf
ehtimg.ra = obsin.ra
ehtimg.dec = obsin.dec
ehtimg.ivec
ehtimg.mjd = 57848
ehtimg.display()
obs = ehtimg.observe_same(obsin, ampcal = ampcal, phasecal = phasecal, add_th_noise = add_th_noise, seed = seed, ttype = "fast")
obs = scan_average(obs.flag_uvdist(uv_min = 0.1e9))
obs = obs.add_fractional_noise(fractional_noise)
dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))
skym = SkyModel(
	ModifiedKerrGMRF,
	prior,
	imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx());
	metadata = (; bulkgrid, transform1, transform2, raster_size, offset),
)
post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode = set_runtime_activity(Enzyme.Reverse))
global fpost = Comrade.asflat(post)
psample = (sky = NamedTuple{keys(prior)}(rand.(values(prior))),)
psample = prior_sample(post)
grid = imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx())

function _rand!(arr, i)
	temp = rand(i)
	append!(arr, temp)
end
function sample_prior(prior)
	arr = Vector{Float64}()
	for i in prior
		temp = rand(i)
		if temp isa Vector
			arr = hcat(arr, temp)
		else
			append!(arr, temp)
		end
	end
	return arr
end
function prior_to_named_tuple(sample, prior)
	vals = []
	marker = 1
	for i in prior
		if length(findall(x->x==:dims, fieldnames(typeof(i)))) > 0
			dms = i.dims
			rng = reduce(*, dms)
			append!(vals, [reshape(sample[marker:(marker+rng-1)], dms)])
			marker += rng
		else
			append!(vals, sample[marker])
			marker += 1
		end
	end
	return NamedTuple{keys(prior)}(vals)
end
function mse(img, img2)
	log(mean(((img .- img2) ) .^ 2))
end
psample = prior_to_named_tuple(sample_prior(prior), prior)
inimg2 = intensitymap(ModifiedKerrGMRF(psample, metadata), grid)
inimg

Comrade.transform(fpost, rand(dimension(fpost)))
function loss(θ, fpost, metadata, inimg)
	psample = Comrade.transform(fpost, θ).sky
	mse(inimg, intensitymap(ModifiedKerrGMRF(psample, metadata), grid))
end

smpl = sample_prior(prior)
prior_to_named_tuple(smpl, prior)

using Accessors
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Optim

t = (; fpost, metadata, inimg)
function f(θ, p)
	loss(θ, p.fpost, p.metadata, p.inimg)
end

mutable struct Callback
	counter::Int
	stride::Int
	const f::Function
end
Callback(stride, f) = Callback(0, stride, f)
global loss_arr = []
fig = Figure(resolution = (800, 400))
function (c::Callback)(state, loss, others...)
	global loss_arr
	global grid
	global fpost
	global metadata
	c.counter += 1

	if c.counter % c.stride == 0
		append!(loss_arr, loss)
		println(
			"Iteration: $(c.counter), Loss: $(loss), m_d: $(rad2μas(Comrade.transform(fpost, state.u).sky.m_d)) μas, spin: $(Comrade.transform(fpost, state.u).sky.spin), β: $(Comrade.transform(fpost, state.u).sky.βv), θs: $(Comrade.transform(fpost, state.u).sky.θs) deg, σimg: $(Comrade.transform(fpost, state.u).sky.σimg)",
		)
        tsol = Comrade.transform(fpost, state.u)
		img=imageviz(intensitymap(ModifiedKerrGMRF(Comrade.transform(fpost, state.u).sky, metadata), grid), colorscale = log10, colorrange = (1e-6, 1e-3), colormap = :inferno) 
		text!(img.axis, (77), (65); text = latexstring("M/D: $(round(tsol.sky.m_d  |> rad2μas,digits=2))"), color = :white, fontsize = 35)
	    text!(img.axis, (13), (65); text = "μas", color = :white, fontsize = 35)
	    text!(img.axis, (77), (52); text = latexstring("a: $(round(tsol.sky.spin, digits=2))"), color = :white, fontsize = 35)
	    text!(img.axis, (77), (35); text = latexstring("\\theta_o: $(round(tsol.sky.θo, digits=2))\\degree"), color = :white, fontsize = 35)
		display(img)
		return false
	else
		return false
	end
end
import CairoMakie as CM
num_data_prods = 1

curr_theme = CM.Theme(
	Axis = (
		xgridvisible = false,
		ygridvisible = false,
		xspinesvisible = false,
		yspinesvisible = false,
		yticklabelsvisible = false,
		yticksvisible = false,
		#ylabelfont="Computer Modern Serif"
		#xlabelvisible=false,
		#ylabelvisible=false,
	),
	Axis3 = (
		xgridvisible = false,
		ygridvisible = false,
		zgridvisible = false,
		xspinesvisible = false,
		yspinesvisible = false,
		zspinesvisible = false,
		xticklabelsvisible = false,
		yticklabelsvisible = false,
		zticklabelsvisible = false,
		xticksvisible = false,
		yticksvisible = false,
		zticksvisible = false,
		xlabelvisible = false,
		ylabelvisible = false,
		zlabelvisible = false,
	),
	Text = (fontsize = 40,),
	Colorbar = (
		fontsize = 30,
		ticklabelsize = 30,
		labelsize = 30,
	),
	Heatmap = (
		xreversed = true,
		colormap = :inferno,
		aspectratio = 1,
		rasterize = true,
	),
	Imageviz = (colormap=:inferno),
	#backgroundcolor = GLMk.colorant"rgba(10%, 10%, 10%, 1.0)"
)

CM.set_theme!(merge(curr_theme, CM.theme_latexfonts()))

vals = ((post) -> begin
	temp = prior_sample(post)#transform(fpost, prior_sample(fpost))
	@reset temp.sky.σimg = 1e-7#1e-7
	@reset temp.sky.m_d = μas2rad(6.0)# 1e-7
	return Comrade.inverse(fpost, temp)
end)(post)
optf = OptimizationFunction(f, AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
dvals = similar(vals)
prob = OptimizationProblem(optf, vals, t)#, data2[1]; manifold = M, structural_analysis = true)

sol = solve(prob, OptimizationOptimisers.Adam(0.05), maxiters = 500, callback = Callback(2, ()->nothing))
Comrade.transform(fpost, sol.u).sky[[:m_d, :spin, :θo, :θs]]
intmap = intensitymap(ModifiedKerrGMRF(Comrade.transform(fpost, sol.u).sky, metadata), grid)
imageviz(intmap)
append!(spins_and_files, merge(Comrade.transform(fpost, sol.u).sky[[:m_d, :spin, :θo, :θs]], (nx_corr=1-round(divergence(nx, intmap),digits=3), filename=filename,) ))


CM.set_theme!(CM.theme_latexfonts())

tsol = Comrade.transform(fpost, sol.u)
joinpath((@__DIR__), "plotting", "$(split(img_path, "/")[end])_best_fits.txt")
fpath = open(joinpath((@__DIR__), "plotting", "$(split(img_path, "/")[end])_best_fits.txt"), "w")
write(fpath, "best_fit = " * string(tsol))
close(fpath)
