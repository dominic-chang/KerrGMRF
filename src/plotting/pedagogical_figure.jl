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
import WGLMakie as CM
import WGLMakie as CairoMakie
using WGLMakie
using LaTeXStrings
using Accessors
using LinearAlgebra

include(joinpath(dirname(@__DIR__), "utils.jl"))
include(joinpath(dirname(@__DIR__), "plotting", "utils.jl"))
include(joinpath(dirname(@__DIR__), "modifiers.jl"))
include(joinpath(dirname(@__DIR__), "models", "KerrGMRF.jl"))

#clrs = [WGLMakie.RGBAf(i.r,i.g,i.b,0.1) for i in CM.Makie.wong_colors()]
clrs = CM.Makie.wong_colors()

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


#using Images
#img = Gray.(Images.load("/Users/dominicchang/Desktop/KerrGMRF/src/image_domain/image.png") )
#Float64.(img) ./ maximum(img) 
#for fname in filenames[(99+9):149]
#fname = filenames[99]
fname = filenames[2]

img_path = joinpath(files, fname)
inimg = rotated(VIDA.load_image(img_path), 108.0/180*π)
#inimg2 = rotated(VIDA.load_image(img_path), 108.0/180*π)
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
intmap |> imageviz
#joinpath(dirname(@__DIR__), "plotting", "$(split(img_path, "/")[end])_best_fits.txt")
#fpath = open(joinpath(dirname(@__DIR__), "plotting", "$(split(img_path, "/")[end])_best_fits.txt"), "r")
#tsol = eval(Meta.parse(read(fpath, String)))
foo = ((post) -> begin
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
		end)
sol = let fpost=fpost, metadata=metadata, inimg=inimg, curr=curr, foo=foo
	temp = nothing
	for _ in 1:10
		curr = (sky = NamedTuple{keys(curr.sky)}(tsol.sky[keys(curr.sky)]),)
		vals = foo(post)
		xvals = Comrade.transform(fpost, vals)
		intensitymap(ModifiedKerrGMRF(xvals.sky, metadata), skym.grid) |> imageviz
		optf = OptimizationFunction(f, AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
		dvals = similar(vals)
		t = (; fpost, metadata, inimg)
		prob = OptimizationProblem(optf, vals, t)

		#sol = solve(prob, OptimizationOptimisers.Adam(0.05), maxiters = 100, callback = Callback(5, ()->nothing))
		temp = solve(prob, OptimizationOptimisers.Adam(0.05), maxiters = 200, callback = Callback(20, fpost, ()->nothing))	
	end
	temp
end


v = Comrade.transform(fpost,sol)
m = begin 
	t = skymodel(post)
	t.f(v.sky, t.metadata)
end

g = m.model.model.metadata.bulkgrid
met = Krang.Kerr(v.sky.spin)
m_d = v.sky.m_d
rpeak = v.sky.rpeak
p1 = v.sky.p1
p2 = v.sky.p2
θs = v.sky.θs * π/180
θo = 163
χ = v.sky.χ
βv = v.sky.βv
η = v.sky.η
ι = v.sky.ι
spectral_index = v.sky.spec
meanarr1, meanarr2, arr1, arr2, rs_vals, ϕks_vals = begin
	arr1 = zeros(size(g))
	arr2 = zeros(size(g))
	meanarr1 = zeros(size(g))
	meanarr2 = zeros(size(g))


	rs_vals = Float64[]
	ϕks_vals = Float64[]
	for (i,α) in enumerate(LinRange(-fovx/m_d, fovx/m_d, 240))
		for (j,β) in enumerate(LinRange(-fovx/m_d, fovx/m_d, 240))
		
			pix = Krang.SlowLightIntensityPixel(met, -α, β, θo * π / 180)
			rs, ϕs, νr, νθ = Krang.emission_coordinates_fast_light(pix, θs, β<0, 0)
			ϕks = Krang.ϕ_kerr_schild(met, rs, ϕs)
			rh = Krang.horizon(met)
			rs_grid = (rs - rh) * rad2μas(m_d) / (raster_size - rh * rad2μas(m_d)) # convert to microarcseconds
			fluid_velocity = Krang.SVector(βv, π / 2.0, χ)
			magnetic_field1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
			magnetic_field2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
			norm1, redshift1, lp1 = @inline Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magnetic_field1, fluid_velocity, νr, νθ)
			norm2, redshift2, lp2 = @inline Krang.synchrotronIntensity(met, α, β, rs, π-θs, θo, magnetic_field2, fluid_velocity, νr, νθ)

			if rs >  rh
				rs_h = rs # / rh
				# grid has maximum radius of 30 units
					
				dim = (X = rs_grid * cos(ϕks), Y = rs_grid * sin(ϕks))
				rat = (rs_h / rpeak)
				cp1 = Comrade.intensity_point(m.model.model.scene[1].material.bulkmodel, dim)
				cp2 = Comrade.intensity_point(m.model.model.scene[2].material.bulkmodel, dim)
				meanarr1[i,j] = norm1^(1 + spectral_index) * min(lp1, 1e2) * max(rat^p1 / (1 + rat^(p1 + p2)) * max(redshift1, eps())^(3 + spectral_index), 0)
				meanarr2[i,j] = norm2^(1 + spectral_index) * min(lp2, 1e2) * max(rat^p1 / (1 + rat^(p1 + p2)) * max(redshift2, eps())^(3 + spectral_index), 0)
				arr2[i,j] = norm1^(1 + spectral_index) * min(lp1, 1e2) * max(rat^p1 / (1 + rat^(p1 + p2)) * max(redshift1, eps())^(3 + spectral_index) * cp1, 0)
				arr1[i,j] = norm2^(1 + spectral_index) * min(lp2, 1e2) * max(rat^p1 / (1 + rat^(p1 + p2)) * max(redshift2, eps())^(3 + spectral_index) * cp2, 0)
			end
			if rs < 10.0
				append!(ϕks_vals, ϕs)
				append!(rs_vals, rs)
			end
		end
	end
	meanarr1, meanarr2, arr1, arr2, rs_vals, ϕks_vals
end
#rs_vals = rs_vals[begin:10:end]
#ϕks_vals = ϕks_vals[begin:10:end]
fig = CM.Figure()
ax = CM.Axis(fig[1,1],aspect=1)
hidedecorations!(ax)
heatmap!(ax, arr1, colorscale=log10, colorrange = (1e0,1e3))
fig

_, nz = size(arr1)
arr1_cone_texture = map(x-> x<0.00 ? NaN : x, arr1)
arr2_cone_texture = map(x-> x<0.00 ? NaN : x, arr2)
rh = Krang.horizon(met)

cone_x = 2.0 * maximum(rs_vals) .* reduce(hcat, [LinRange(-1,1,nz) for j in 1:nz])
cone_y = 2.0 * maximum(rs_vals) .* reduce(hcat, [LinRange(-1,1,nz) for j in 1:nz])'
cone_z1 = cos(θs) .* hypot.(cone_x, cone_y)
cone_z2 = .-cone_z1
sphere_θ = LinRange(0, 2π, 96)
sphere_ϕ = LinRange(0, π, 48)
sphere_x = rh .* [sin(sphere_ϕ[j]) * cos(sphere_θ[i]) for i in eachindex(sphere_θ), j in eachindex(sphere_ϕ)]
sphere_y = rh .* [sin(sphere_ϕ[j]) * sin(sphere_θ[i]) for i in eachindex(sphere_θ), j in eachindex(sphere_ϕ)]
sphere_z = rh .* [cos(sphere_ϕ[j]) for i in eachindex(sphere_θ), j in eachindex(sphere_ϕ)]

cone_xmask= map(x->abs(x)<7 ? 1 : NaN, cone_x)
cone_ymask= map(x->abs(x)<7 ? 1 : NaN, cone_y)
meanarr1 .*= cone_xmask
meanarr1 .*= cone_ymask
arr1_cone_texture .*= cone_xmask
arr1_cone_texture .*= cone_ymask
arr2_cone_texture .*= cone_xmask
arr2_cone_texture .*= cone_ymask
meanarr2 .*= cone_xmask
meanarr2 .*= cone_ymask

function gen_uv(shift, cone_z2)
    return vec(map(CartesianIndices(size(cone_z2))) do ci
        tup = ((ci[1], ci[2]) .- 1) ./ ((size(cone_z2) .* shift) .- 1)
        return Vec2f(reverse(tup))
    end)
end

CM.set_theme!(CM.theme_latexfonts())
#CM.set_theme!(CM.theme_dark())

obs_el = 2π/15
obs_az = 3π/4 + π

afm = [RGBAf(i.r,i.g,i.b, 5min(t/100, 1.0)^2) for (t,i) in enumerate(WGLMakie.Makie.ColorSchemes.afmhot)]
gs = [RGBAf(i.r,i.g,i.b, (t/256)^0.75) for (t,i) in enumerate(WGLMakie.Makie.ColorSchemes.grays)]

inimg2 = VIDA.load_image(img_path)
inimg2 = rotated(inimg2, 108.0/180*π)
inimg2 = regrid(inimg2, imagepixels(μas2rad(60), μas2rad(60), 480, 480))
cone_fig = CM.Figure(size = (900, 1300))
begin
cone_ax = CM.Axis3(
	cone_fig[1, 1],
	xgridvisible=false,
	ygridvisible=false,
	zgridvisible=false,
	xlabel="x (GM/c^2)",
	ylabel="y (GM/c^2)",
	zlabel="z (GM/c^2)",
	elevation=obs_el, 
	azimuth=obs_az,
	aspect=(1,1,1),
	width=900,
	height=1300
)
	CM.xlims!(cone_ax, -15, 15);
	CM.ylims!(cone_ax, -15, 15);
	CM.zlims!(cone_ax, -5, 25);
	CM.hidedecorations!(cone_ax)
	CM.hidespines!(cone_ax)

	image_surface = surface!(
		cone_ax,
		0.35 .* cone_x,
		0.35 .* cone_y,
		zeros(size(cone_x));
		color = inimg2, 
		colormap = :afmhot,
		nan_color=:transparent,
		#overdraw=true
	)
	#CM.rotate!(image_surface, ((0,1,0)), (π/10))
	CM.translate!(image_surface, ((7,-1.5,18)))
	CM.rotate!(image_surface, ((0,1,0)), (15/180*π))

	surface!(cone_ax, sphere_x, sphere_y, sphere_z; color = :black)
	cone_surface1 = surface!(
		cone_ax,
		cone_x,
		cone_y,
		cone_z1;
		color = meanarr1,
		colormap = gs,
		nan_color=:transparent,
		transparency = true
	)
	cone_surface2 = surface!(
		cone_ax,
		cone_x ,
		cone_y,
		cone_z2;
		color = meanarr2,
		colormap = gs,
		nan_color=:transparent,
		transparency=true
	)
	cone_surface1 = surface!(
		cone_ax,
		cone_x,
		cone_y,
		cone_z1;
		color = arr1_cone_texture,
		colormap = afm,
		#colorrange=(0.0,100.0),
		colorscale=sqrt,
		nan_color=:transparent,
		transparency=true
	)
	cone_surface2 = surface!(
		cone_ax,
		cone_x ,
		cone_y,
		cone_z2;
		color = arr2_cone_texture,
		#colorscale = log10,
		colormap = afm,
		#colorrange = (1e0, 1e3),
		#colorrange=(0.0,200.0),
		colorscale=sqrt,
		nan_color=:transparent,
		transparency = true
	)
	s = 30
	 
	mesh_x = reduce(hcat, [LinRange(-7, 7, s) for _ in 1:s])# convert to microarcseconds
	mesh_y = reduce(hcat, [LinRange(-7, 7, s) for _ in 1:s])'
	mesh_z1 = hypot.(mesh_x, mesh_y) .* cos(θs)
	mesh_z2 = .-mesh_z1

	points = vec([Point3f(xv, yv, zv) for (xv, yv, zv) in zip(mesh_x, mesh_y, mesh_z1)])
	_faces = decompose(WGLMakie.Makie.GeometryBasics.QuadFace{WGLMakie.GLIndex}, WGLMakie.Makie.Tessellation(Rect(0, 0, 1, 1), size(mesh_z1)))
	_normals = normalize.(points)
	uv = gen_uv(0.0, mesh_z1)
	uv_buff = WGLMakie.Buffer(uv)
	gb_mesh = WGLMakie.Makie.GeometryBasics.Mesh(points, _faces; uv = uv_buff, normal = _normals)
	wireframe!(cone_ax, gb_mesh, color=WGLMakie.RGBAf(0.5,0.5,0.5,0.4), linewidth=1, transparency=true)

	points = vec([Point3f(xv, yv, zv) for (xv, yv, zv) in zip(mesh_x, mesh_y, mesh_z2)])
	_faces = decompose(WGLMakie.Makie.GeometryBasics.QuadFace{WGLMakie.GLIndex}, WGLMakie.Makie.Tessellation(Rect(0, 0, 1, 1), size(mesh_z2)))
	_normals = normalize.(points)
	uv = gen_uv(0.0, mesh_z2)
	uv_buff = WGLMakie.Buffer(uv)
	gb_mesh = WGLMakie.Makie.GeometryBasics.Mesh(points, _faces; uv = uv_buff, normal = _normals)
	wireframe!(cone_ax, gb_mesh, color=WGLMakie.RGBAf(0.5,0.5,0.5,0.4), linewidth=1, transparency=true)

	#Geodesic 1
	pix = Krang.SlowLightIntensityPixel(met, 4.0, 2.3, 17*π/180)
	τi = Krang.mino_time(pix, 15.4*π/180, true, 0)
	τf = Krang.mino_time(pix, θs, true, 0)
	τvals = LinRange(τi, τf, 100)
	outvals = Krang.emission_coordinates.(Ref(pix), τvals)

	rvals = []
	θvals = []
	ϕvals = []
	for i in 1:length(outvals)
	    push!(rvals, outvals[i][2])
	    push!(θvals, outvals[i][3] )
	    push!(ϕvals, outvals[i][4])
	end
	xvals = filter(x->!isnan(x),collect(rvals .* sin.(θvals) .* cos.(ϕvals)))
	yvals = filter(x->!isnan(x),collect(rvals .* sin.(θvals) .* sin.(ϕvals)))
	zvals = filter(x->!isnan(x),collect(rvals .* cos.(θvals)))
	obs_dir = [cos(obs_el)*cos(obs_az), cos(obs_el)*sin(obs_az), sin(obs_el)]
	north_dir = LinearAlgebra.normalize([0.,0.,1.0] .- (LinearAlgebra.dot(obs_dir,[0.,0.,1.0]) .* obs_dir))
	dxvals = xvals[begin+1:end] .- xvals[begin:end-1]
	dyvals = yvals[begin+1:end] .- yvals[begin:end-1]
	dzvals = zvals[begin+1:end] .- zvals[begin:end-1]
	pvals = collect.(zip(dxvals, dyvals, dzvals))

	tick_dir = LinearAlgebra.normalize.(Ref(north_dir) .- (LinearAlgebra.dot.(pvals, Ref(north_dir)) .* pvals ./ map(x->hypot(x...), pvals))) ./ 2
	x_tick_vals = [i[1] for i in tick_dir]
	y_tick_vals = [i[2] for i in tick_dir]
	z_tick_vals = [i[3] for i in tick_dir]

	CM.arrows!(
	    cone_ax, 
	    xvals[begin:end-2], 
	    yvals[begin:end-2], 
	    zvals[begin:end-2], 
	    dxvals[begin:end-1], 
	    dyvals[begin:end-1], 
	    dzvals[begin:end-1], 
		minshaftlength = 0.0,
		tiplength = 0.0,
		markerscale=0.2, 
	    shading=false,
		color=clrs[2]
	)

	xi,yi,zi = xvals[end], yvals[end], zvals[end]
	xf,yf,zf = xvals[begin], yvals[begin], zvals[begin]
	

	#Geodesic 2
	pix = Krang.SlowLightIntensityPixel(met, -3.5, 3.7, 17*π/180)
	τi = Krang.mino_time(pix, 11.8π/180, true, 0)
	τf = Krang.mino_time(pix, π - θs, true, 1)
	τvals = LinRange(τi, τf, 100)
	outvals = Krang.emission_coordinates.(Ref(pix), τvals)

	rvals = []
	θvals = []
	ϕvals = []
	for i in 1:length(outvals)
	    push!(rvals, outvals[i][2])
	    push!(θvals, outvals[i][3] )
	    push!(ϕvals, outvals[i][4])
	end
	xvals = filter(x->!isnan(x),collect(rvals .* sin.(θvals) .* cos.(ϕvals)))
	yvals = filter(x->!isnan(x),collect(rvals .* sin.(θvals) .* sin.(ϕvals)))
	zvals = filter(x->!isnan(x),collect(rvals .* cos.(θvals)))
	obs_dir = [cos(obs_el)*cos(obs_az), cos(obs_el)*sin(obs_az), sin(obs_el)]
	north_dir = LinearAlgebra.normalize([0.,0.,1.0] .- (LinearAlgebra.dot(obs_dir,[0.,0.,1.0]) .* obs_dir))
	dxvals = xvals[begin+1:end] .- xvals[begin:end-1]
	dyvals = yvals[begin+1:end] .- yvals[begin:end-1]
	dzvals = zvals[begin+1:end] .- zvals[begin:end-1]
	pvals = collect.(zip(dxvals, dyvals, dzvals))

	tick_dir = LinearAlgebra.normalize.(Ref(north_dir) .- (LinearAlgebra.dot.(pvals, Ref(north_dir)) .* pvals ./ map(x->hypot(x...), pvals))) ./ 2
	x_tick_vals = [i[1] for i in tick_dir]
	y_tick_vals = [i[2] for i in tick_dir]
	z_tick_vals = [i[3] for i in tick_dir]

	CM.arrows!(
	    cone_ax, 
	    xvals[begin:end-2], 
	    yvals[begin:end-2], 
	    zvals[begin:end-2], 
	    dxvals[begin:end-1], 
	    dyvals[begin:end-1], 
	    dzvals[begin:end-1], 
		minshaftlength = 0.0,
		tiplength = 0.0,
		markerscale=0.2, 
	    shading=false,
		color=clrs[2]
	)
	scatter!(
		cone_ax,
		[xi,],
		[yi,],
		[zi,],
		markersize = 20.0,
		color=clrs[2],
		overdraw=true
	)
	scatter!(
		cone_ax,
		[xvals[end],],
		[yvals[end],],
		[zvals[end],],
		markersize = 20.0,
		color=clrs[2],
		overdraw=true
	)
	scatter!(
		cone_ax,
		[xf,],
		[yf,],
		[zf,],
		markersize = 20.0,
		color=clrs[2],
		overdraw=true
	)
	scatter!(
		cone_ax,
		[xvals[begin],],
		[yvals[begin],],
		[zvals[begin],],
		markersize = 20.0,
		color=clrs[2],
		overdraw=true
	)
	CM.arrows!(
	    cone_ax, 
	    [0.0,], 
	    [0.0,], 
	    [0.0,], 
	    [0.0,], 
	    [0.0,], 
	    [5.0,], 
		minshaftlength = 0.0,
		tiplength = 0.4,
		markerscale=0.3, 
	    shading=false,
		color=clrs[1]
	)
	CM.arrows!(
	    cone_ax, 
	    -sin.(LinRange(-π, 0, 40) .- π/2), 
	    cos.(LinRange(-π, 0, 40)  .- π/2), 
	    [3.0,], 
		-0.1cos.(LinRange(-π, 0, 40)  .- π/2), 
	    -0.1sin.(LinRange(-π, 0, 40)  .- π/2), 
	    [0.0,], 
		minshaftlength = 0.0,
		tiplength = 0.0,
		markerscale=0.1, 
	    shading=false,
		color=:black
	)
	CM.arrows!(
	    cone_ax, 
	    [-sin(-π/2),],
	    [cos(-π/2),],
	    [3.0,], 
		[-0.5cos(-π/2),],
	    [-0.5sin(-π/2),],
	    [0.0,], 
		minshaftlength = 0.0,
		tiplength = 0.2,
		markerscale=0.4, 
	    shading=false,
		color=:black
	)

	#CM.text!(
	#	cone_ax,
	#	4.0,
	#	0.0,
	#	24.0,
	#	text="⏿",
	#	fontsize=100,
	#	color=RGBAf(0.7,0.7,0.7,1.0),
	#	rotation=(3π/2-15/180*π)
	#)
end

save(joinpath((@__DIR__), "pedagogical.png"), cone_fig)

CM.Colorbar(cone_fig[1, 2], cone_surface; label = "arr")
display(cone_fig)
cone_fig
