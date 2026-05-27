using Pkg;
Pkg.activate(dirname(dirname(@__DIR__)));
using VIDA
using Comrade
using Krang
using VLBIImagePriors
using BasicInterpolators
using Enzyme
using Optimization, OptimizationOptimisers, OptimizationOptimJL, Optim
using StableRNGs
using Distributions, DistributionsAD
using Pyehtim
import CairoMakie as CM
using LaTeXStrings
using Accessors

include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF.jl"))

curr_theme = CM.Theme(
	Axis = (
		xgridvisible = false,
		ygridvisible = false,
		xspinesvisible = false,
		yspinesvisible = false,
		yticklabelsvisible = false,
		yticksvisible = false,
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
)
CM.set_theme!(merge(curr_theme, CM.theme_latexfonts()))

function ModifiedKerrGMRF(θ2, meta)
	m = Comrade.modify(KerrGMRF(θ2, meta), Stretch((θ2.m_d), (θ2.m_d)), Rotate(θ2.pa))#, Shift(θ2.x, θ2.y))
	return RenormalizedFlux(m, θ2.f)
end

function mse(img, img2)
	nx = NxCorr(img)
	#log(divergence(nx, img2))
	log(mean((((img ./ flux(img)) .- (img2 ./ flux(img2)))) .^ 2))
end

function loss(θ, fpost, metadata, inimg)
	psample = Comrade.transform(fpost, θ).sky
	grid = fpost.lpost.skymodel.grid
	mse(inimg, intensitymap(ModifiedKerrGMRF(psample, metadata), grid))
end

function f(θ, p)
	loss(θ, p.fpost, p.metadata, p.inimg)
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
bulkpix = 240
raster_size = 240.0# in microarcseconds    
snrcut = 3.0
year = 2017

data = Dict(2017=>"SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits", 2018 => "L2V1_M87_2018_111_b3_hops_netcal_10s_StokesI.uvfits", "bhex" => "frame0008_230.5_GHz_synthdata_ngEHTsim.uvfits")
path = joinpath(dirname(dirname(@__DIR__)), "data", data[year])
files = "/Users/dominicchang/Desktop/CenterComparison/data/snapshots"#/n/holylabs/doeleman_lab/Users/dochang/GRMHDImages/snapshots"
#filenames = filter(file->occursin("_160_", file), readdir(files))
filenames = filter(file->occursin("_160_", file), readdir(files))


function (c::Callback)(state, loss, others...)
	fpost = c.fpost
	metadata = fpost.lpost.skymodel.metadata
	loss_arr = c.loss_arr
	grid = fpost.lpost.skymodel.grid
	c.counter += 1

	if c.counter % c.stride == 0
		append!(loss_arr, loss)
		println(
			"Iteration: $(c.counter), Loss: $(loss), m_d: $(rad2μas(Comrade.transform(fpost, state.u).sky.m_d)) μas, spin: $(Comrade.transform(fpost, state.u).sky.spin), β: $(Comrade.transform(fpost, state.u).sky.βv), θs: $(Comrade.transform(fpost, state.u).sky.θs) deg, σimg: $(Comrade.transform(fpost, state.u).sky.σimg)",
		)
		tsol = Comrade.transform(fpost, state.u)
		img=imageviz(intensitymap(ModifiedKerrGMRF(Comrade.transform(fpost, state.u).sky, metadata), grid), colormap = :inferno)
		CairoMakie.text!(img.axis, (77-20), (65-20); text = latexstring("M/D: $(round(tsol.sky.m_d  |> rad2μas,digits=2))\\ \\mu as"), color = :white, fontsize = 35)
		CairoMakie.text!(img.axis, (77-20), (52-20); text = latexstring("a: $(round(tsol.sky.spin, digits=2))"), color = :white, fontsize = 35)
		CairoMakie.text!(img.axis, (77-20), (35-20); text = latexstring("\\theta_o: $(round(tsol.sky.θo, digits=2))\\degree"), color = :white, fontsize = 35)
		display(img)
		return false
	else
		return false
	end
end


using Images
img = Gray.(Images.load("/Users/dominicchang/Desktop/KerrGMRF/src/image_domain/image.png") )
Float64.(img) ./ maximum(img) 
#for fname in filenames[(99+9):149]
#fname = filenames[99]
fname = filenames[2]

	img_path = joinpath(files, fname)

	inimg = rotated(VIDA.load_image(img_path), 108.0/180*π)
	inimg .= max.(inimg, maximum(inimg)/1_000)
	#inimg = regrid(inimg, imagepixels(μas2rad(120), μas2rad(120), 120, 120))
	inimg = regrid(inimg, imagepixels(μas2rad(160), μas2rad(160), 160, 160))

	#img = Gray.(Images.load("/Users/dominicchang/Desktop/KerrGMRF/src/image_domain/image.png") )
	#inimg .= reverse((Float64.(img) )', dims = 1)
	#inimg = regrid(inimg, imagepixels(μas2rad(120), μas2rad(120), 100, 100))
	#inimg ./= sum(inimg)



	npix = inimg.X |> length
	fovx, fovy = fieldofview(inimg)

	imageviz(inimg, colorscale = log10, colorrange = (1e-6, 1e-3), colormap = :inferno) |> display
	imageviz(inimg, colormap = :inferno) |> display


	bulkgrid = imagepixels(bulkx, bulky, bulkpix, bulkpix; executor = Serial())
	transform1, cprior1 = matern(size(bulkgrid))
	transform2, cprior2 = matern(size(bulkgrid))
	prior = (
		#m_d = Uniform(μas2rad(1.0), μas2rad(8.0)),
		m_d = Uniform(μas2rad(0.01), μas2rad(8.0)),
		spin = Uniform(0.01, 0.99),
		θo = Uniform(120.0, 179.0),
		θs = Uniform(20, 90.0),
		rpeak = Uniform(1.0, 8.0),
		p1 = Uniform(0.1, 5.0),
		p2 = Uniform(0.1, 5.0),
		χ = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
		ι = Uniform(-π/2, π/2),
		βv = Uniform(0.01, 0.99),
		spec = Uniform(-1.0, 5.0),
		η = VLBIImagePriors.DiagonalVonMises(0, inv(π^2)),
		frac = VLBIImagePriors.DeltaDist(1.0),
		pa = VLBIImagePriors.DeltaDist(108.0*π/180),
		f = VLBIImagePriors.DeltaDist(1.0),
		σimg = truncated(Normal(0.0, 1.0); lower = 0.0),
		ρpr = truncated(InverseGamma(1.0, -log(0.1)*10); lower = 1.0, upper = 2*max(size(bulkgrid)...)),
		νpr = Uniform(1.0, 5.0),
		c1 = cprior1,
		c2 = cprior2,
	)
	offset = 0.0
	metadata = (; bulkgrid, transform1, transform2, raster_size, offset)
	obsin = (ehtim.obsdata.load_uvfits(path) |> scan_average).flag_uvdist(uv_min = 0.1e9).add_fractional_noise(fractional_noise)
	ehtimg = ehtim.image.load_image(img_path).rotate(108*π/180)
	ehtimg.rf, ehtimg.ra, ehtimg.dec, ehtimg.mjd = obsin.rf, obsin.ra, obsin.dec, 57848

	obs = scan_average(ehtimg.observe_same(obsin, ampcal = ampcal, phasecal = phasecal, add_th_noise = add_th_noise, seed = seed, ttype = "fast"))
	dvis, dvisamp, dcphase, dlcamp = extract_table(obs, Visibilities(), VisibilityAmplitudes(), ClosurePhases(; snrcut = 3.0), LogClosureAmplitudes(; snrcut = 3.0))
	skym = SkyModel(
		ModifiedKerrGMRF,
		prior,
		imagepixels(fovx, fovy, npix, npix; executor = ThreadsEx());
		metadata = (; bulkgrid, transform1, transform2, raster_size, offset),
	)
	post = VLBIPosterior(skym, Comrade.IdealInstrumentModel(), dlcamp, dcphase; admode = set_runtime_activity(Enzyme.Reverse))
	fpost = Comrade.asflat(post)

	t = (; fpost, metadata, inimg)
	curr =
		tsol = (
			sky = (
				m_d = 1.852782699827925e-11,
				spin = 0.34261943710501536,
				θo = 159.66325910346768,
				θs = 72.55221410100953,
				rpeak = 1.0320609725554761,
				p1 = 4.997485810130196,
				p2 = 3.4271109236512496,
				χ = 0.784485331711496,
				ι = 1.5672026597904312,
				βv = 0.3511992071422593,
				spec = 2.745807873295397,
				η = -0.2979777845654403,
				frac = 1.0,
				pa = 1.8849555921538759,
				f = 1.0,
			),
		)
	curr = (sky = NamedTuple{keys(curr.sky)}(tsol.sky[keys(curr.sky)]),)
		vals = ((post) -> begin
			temp = prior_sample(post)#transform(fpost, prior_sample(fpost))
			newvals = []
			for key in keys(temp.sky)
				if key in keys(curr.sky)
					push!(newvals, getproperty(curr.sky, key))
				else
					push!(newvals, getproperty(temp.sky, key))
				end
			end
			outvals = (sky = NamedTuple{keys(temp.sky)}(newvals),)
			@reset outvals.sky.σimg = 1e-10
			return Comrade.inverse(fpost, outvals)
		end)(post)
		xvals = Comrade.transform(fpost, vals)

	newgrid = imagepixels(fovx, fovy, 1000,1000)
	intmap = intensitymap(ModifiedKerrGMRF(xvals.sky, metadata), newgrid) 
	intmap ./= flux(intmap)
	|> imageviz

	#joinpath(dirname(@__DIR__), "plotting", "$(split(img_path, "/")[end])_best_fits.txt")
	#fpath = open(joinpath(dirname(@__DIR__), "plotting", "$(split(img_path, "/")[end])_best_fits.txt"), "r")
	#tsol = eval(Meta.parse(read(fpath, String)))
	for _ in 1:100
		curr = (sky = NamedTuple{keys(curr.sky)}(tsol.sky[keys(curr.sky)]),)
		vals = ((post) -> begin
			temp = prior_sample(post)#transform(fpost, prior_sample(fpost))
			newvals = []
			for key in keys(temp.sky)
				if key in keys(curr.sky)
					push!(newvals, getproperty(curr.sky, key))
				else
					push!(newvals, getproperty(temp.sky, key))
				end
			end
			outvals = (sky = NamedTuple{keys(temp.sky)}(newvals),)
			@reset outvals.sky.σimg = 1e0
			return Comrade.inverse(fpost, outvals)
		end)(post)
		xvals = Comrade.transform(fpost, vals)
		intensitymap(ModifiedKerrGMRF(xvals.sky, metadata), skym.grid) |> imageviz
		optf = OptimizationFunction(f, AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
		dvals = similar(vals)
		prob = OptimizationProblem(optf, vals, t)

		#sol = solve(prob, OptimizationOptimisers.Adam(0.05), maxiters = 100, callback = Callback(5, ()->nothing))
		sol = solve(prob, OptimizationOptimisers.Adam(0.2), maxiters = 100, callback = Callback(5, fpost, ()->nothing))

		tsol = Comrade.transform(fpost, sol.u)

		joinpath((@__DIR__), "$(split(img_path, "/")[end])_best_fits.txt")
		fpath = open(joinpath((@__DIR__), "$(split(img_path, "/")[end])_best_fits.txt"), "w")
		write(fpath, "best_fit = " * string(tsol))
		close(fpath)
	end

	# Do one last optimization
	vals = ((post) -> begin
		temp = prior_sample(post)
		newvals = []
		for key in keys(temp.sky)
			if key in keys(curr.sky)
				push!(newvals, getproperty(curr.sky, key))
			else
				push!(newvals, getproperty(temp.sky, key))
			end
		end
		outvals = (sky = NamedTuple{keys(temp.sky)}(newvals),)
		@reset outvals.sky.σimg = 1e0#1
		return Comrade.inverse(fpost, outvals)
	end)(post)
	xvals = Comrade.transform(fpost, vals)
	optf = OptimizationFunction(f, AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
	dvals = similar(vals)
	prob = OptimizationProblem(optf, vals, t)
	sol = solve(prob, OptimizationOptimisers.Adam(0.05), maxiters = 500, callback = Callback(5, fpost, ()->nothing))
	tsol = Comrade.transform(fpost, sol.u)
	#intensitymap(ModifiedKerrGMRF(tsol.sky, metadata), skym.grid) |> imageviz
	joinpath((@__DIR__), "$(split(img_path, "/")[end])_best_fits.txt")
	fpath = open(joinpath((@__DIR__), "$(split(img_path, "/")[end])_best_fits.txt"), "w")
	write(fpath, "best_fit = " * string(tsol))
	close(fpath)
end
