struct _EmissivityModel{T, B} 
	magnetic_field::Krang.SVector{3, T}
	fluid_velocity::Krang.SVector{3, T}
	bulkmodel::B
	spectral_index::T
	rpeak::T
	p1::T
	p2::T
	m_d::T
	raster_size::T
	offset::T

	function _EmissivityModel(magfield, vel, bulkmodel::B, spec::T, rpeak::T, p1::T, p2::T, m_d::T, raster_size::T, offset::T) where {T, B}
		new{T, B}(magfield, vel, bulkmodel, spec, rpeak, p1, p2, m_d, raster_size, offset)
	end
end
struct EmissivityModel<: Krang.AbstractMaterial
    model::_EmissivityModel
    subimgs::NTuple{1,Int}
    function EmissivityModel(model, n)
        new(model, (n,))
    end
end

Krang.isFastLight(material::EmissivityModel) = true
Krang.isAxisymmetric(material::EmissivityModel) = false

@inline function (prof::EmissivityModel)(pix::Krang.AbstractPixel, intersection::Krang.Intersection) 
	(; m_d, magnetic_field, fluid_velocity, bulkmodel, spectral_index, rpeak, p1, p2, raster_size, offset) = prof.model
	(; rs, ϕs, θs, νr, νθ) = intersection

	θo = Krang.inclination(pix)
	met = Krang.metric(pix)
	α, β = @inline Krang.screen_coordinate(pix)
    T = typeof(met.spin)

	norm, redshift, lp = @inline Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magnetic_field, fluid_velocity, νr, νθ)

	rh = Krang.horizon(met)
	rs_h = rs # / rh
	rs_grid = (rs - rh) * rad2μas(m_d) / (raster_size - rh * rad2μas(m_d)) # convert to microarcseconds
	if rs_grid < 0
		return zero(T)
	end

	ϕks = Krang.ϕ_kerr_schild(met, rs, ϕs)
	dim = (X = rs_grid * cos(ϕks) + offset, Y = rs_grid * sin(ϕks) + offset)
	rat = (rs_h / rpeak)

	bulkpix = bulkmodel.img.X.len
	cp = exp(ComradeBase.intensity_point(bulkmodel, dim) / (bulkpix^2))
	ans = rat^p1 / (one(T) + rat^(p1 + p2)) * max(redshift, eps(T))^(T(3) + spectral_index) * cp
	ans = norm^(1 + spectral_index) * min(lp, 1e2) * ans
	return ans
end

struct KerrGMRF{A, S, F} <: ComradeBase.AbstractModel
	met::Krang.Kerr{A}
	θo::A
	scene_n0::S
    scene_n1::S
	metadata::F
	function KerrGMRF(θ, metadata)
		(; frac, m_d, spin, θs, θo, χ, ι, βv, spec, η, spec, rpeak, p1, p2, ρpr, νpr, c1, c2) = θ
		(; bulkgrid, transform1, transform2, raster_size, offset) = metadata
		A = typeof(θo)
		bulkint1 = transform1(c1, ρpr, νpr)
		bulkmodel1 = bulk(bulkint1, θ.σimg, bulkgrid)

		bulkint2 = transform2(c2, ρpr, νpr)
		bulkmodel2 = bulk(bulkint2, θ.σimg, bulkgrid)

		vel = Krang.SVector(βv, A(π / 2), χ)

		magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
		_material1 = _EmissivityModel(magfield1, vel, bulkmodel1, spec, rpeak, p1, p2, m_d, raster_size, offset)
		geometry1 = Krang.ConeGeometry(θs * π / 180, (; frac,))
		mesh1_n0 = Krang.Mesh(geometry1, EmissivityModel(_material1, 0))
		mesh1_n1 = Krang.Mesh(geometry1, EmissivityModel(_material1, 1))

		magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
		_material2 = _EmissivityModel(magfield2, vel, bulkmodel2, spec, rpeak, p1, p2, m_d, raster_size, offset)
		geometry2 = Krang.ConeGeometry(π-θs * π / 180, (; frac,))
		mesh2_n0 = Krang.Mesh(geometry2, EmissivityModel(_material2, 0))
        mesh2_n1 = Krang.Mesh(geometry2, EmissivityModel(_material2, 1))

		scene_n0 = Krang.Scene((mesh1_n0, mesh2_n0))
		scene_n1 = Krang.Scene((mesh1_n1, mesh2_n1))

		new{typeof(θo), typeof(scene_n0), typeof(metadata)}(Krang.Kerr(spin), θo, scene_n0, scene_n1, metadata)
	end
end

Comrade.imanalytic(::Type{<:KerrGMRF}) = Comrade.NotAnalytic()
Comrade.visanalytic(::Type{<:KerrGMRF}) = Comrade.NotAnalytic()
function bulk(transformedc, σimg, bulkgrid)
	bulkimg = IntensityMap(σimg * transformedc, bulkgrid)
	return BicubicInterpolatedImage(bulkimg)#, bulkint)
end
struct N0_wrapper <: ComradeBase.AbstractModel
    model 
end
struct N1_wrapper <: ComradeBase.AbstractModel
    model 
end
@inline function intensity_point_and_mask(m::N0_wrapper, p) 
	(; X, Y) = p

	(; met, scene_n0, θo) = m.model

	pix = Krang.SlowLightIntensityPixel(met, -X, Y, θo * π / 180.)
	return _render(pix, scene_n0)
end

@inline function intensity_point_and_mask(m::N1_wrapper, p) 
	(; X, Y) = p
	(; met, scene_n1, θo) = m.model

	pix = Krang.SlowLightIntensityPixel(met, -X, Y, θo * π / 180.)
	return _render(pix, scene_n1)[1]
end


#function Comrade.ComradeBase.intensitymap(
#        s::M,
#        dims::Comrade.AbstractDomain
#    ) where {M <: Comrade.AbstractModel}
#    return Comrade.ComradeBase.create_imgmap(Comrade.intensitymap(Comrade.imanalytic(M), s, dims), dims)
#end

function transform_intensitymap(model, transform::Shift, img)
    (;Δx, Δy) = transform
    data = getfield(img, :data)
    grid = axisdims(img)
    dims, executor, header, posang = getfield.(Ref(grid), (:dims, :executor, :header, :posang))
    X, Y = dims
    @reset X.val.data.start -= Δx
    @reset X.val.data.stop -= Δx

    @reset Y.val.data.start -= Δy
    @reset Y.val.data.stop -= Δy

    newgrid = Comrade.RectiGrid(dims; executor, header, posang)
    return IntensityMap(data, newgrid)
end
@inline scale_intensitymap(m, ::Shift, img) = VLBISkyModels.unitscale(img[1], m)

@inline function transform_intensitymap(model, transform::Stretch, img)
    (;α, β) = transform
    data = getfield(img, :data)
    grid = axisdims(img)
    dims, executor, header, posang = getfield.(Ref(grid), (:dims, :executor, :header, :posang))
    X, Y = dims
    @reset X.val.data.start /= α
    @reset X.val.data.stop /= α
    @reset X.val.span.step /= α

    @reset Y.val.data.start /= β
    @reset Y.val.data.stop /= β
    @reset Y.val.span.step /= β

    dims = (X, Y)
    newgrid = Comrade.RectiGrid(dims; executor, header, posang)
    return IntensityMap(data, newgrid)
end
function scale_intensitymap(m, transform::Stretch, img)
    (;α, β) = transform
    T = typeof(α)
    return inv(α * β) * VLBISkyModels.unitscale(T, m)
end

@inline function transform_intensitymap(model, transform::Rotate, img)
    (;s, c) = transform
    g = axisdims(img)
    dims, executor, header, posang = getfield.(Ref(g), (:dims, :executor, :header, :posang))
    interpolated_data = BicubicInterpolator(g.X, g.Y, img, NoBoundaries())
    new_data = getfield(img, :data)
    for (ix,X) in enumerate(g.X)
        for (iy,Y) in enumerate(g.Y)
            new_data[ix, iy] = interpolated_data(c * X + s * Y, s * X + c * Y)
        end
    end

    newgrid = Comrade.RectiGrid(dims; executor, header, posang)
    return IntensityMap(new_data, newgrid)
end
@inline scale_intensitymap(::NotPolarized, transform::Rotate, img) = one(typeof(getparam(transform, :s, img[1,1])))

function modify_intensitymap(model, t::Tuple, img, scale)
    imgt = transform_intensitymap(model, last(t), img)
    scalet = scale_intensitymap(model, last(t), img)
    return modify_intensitymap(model, Base.front(t), imgt, scalet * scale)
end
modify_intensitymap(model, ::Tuple{}, img, scale) = img, scale

function Comrade.ComradeBase.intensitymap_numeric!(img::Comrade.IntensityMap, m::M) where {M <: VLBISkyModels.ModifiedModel}
    mbase = m.model
    transform = m.transform
    ispol = VLBISkyModels.ispolarized(M)
    imgt, scale = modify_intensitymap(ispol, transform, img, VLBISkyModels.unitscale(eltype(img[1,1]), ispol))

    return scale .* Comrade.ComradeBase.intensitymap_numeric!(imgt, mbase)
end

function custom_intmap_n0!(img::IntensityMap, mask, m)
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
	grids = axisdims(img)
    bimg = baseimage(img)

	m_n0 = N0_wrapper(m)
	sub_g = @view g[begin:4:end, begin:4:end]
	sub_bimg = @view bimg[begin:4:end, begin:4:end]
	sub_mask = @view mask[begin:4:end, begin:4:end]
    Threads.@threads for I in eachindex(sub_g, sub_bimg)
		tempimg, tempmax = intensity_point_and_mask(m_n0, sub_g[I])
        sub_bimg[I], sub_mask[I] =  tempimg * dx * dy, tempmax>0
    end
	interpolated_data = BicubicInterpolator(grids.X[begin:4:end], grids.Y[begin:4:end], sub_bimg, NoBoundaries())
	interpolated_mask = BicubicInterpolator(grids.X[begin:4:end], grids.Y[begin:4:end], sub_mask, NoBoundaries())
	Threads.@threads for I in eachindex(g, bimg)
        bimg[I] = interpolated_data(g[I]...)
		mask[I] = interpolated_mask(g[I]...) > 0.0
    end

    return nothing
end

function custom_intmap_n1!(img::IntensityMap, mask, m)
    dx, dy = pixelsizes(img)
    g = domainpoints(img)
    bimg = baseimage(img)

	m_n1 = N1_wrapper(m)
	sub_g = @view g[mask]
	sub_bimg = @view bimg[mask]
    Threads.@threads for I in eachindex(sub_g, sub_bimg)
		temp = intensity_point_and_mask(m_n1, sub_g[I])
        sub_bimg[I] +=  temp * dx * dy
    end

    return nothing
end

function Comrade.ComradeBase.intensitymap_numeric!(img::Comrade.IntensityMap, m::KerrGMRF)
	mask = zeros(Bool, size(img))
    @inline custom_intmap_n0!(img, mask, m) 
	@inline custom_intmap_n1!(img, mask, m) 
    return img 
end

function Comrade.ComradeBase.visibilitymap_numeric(m::Comrade.VLBISkyModels.ModifiedModel, grid::Comrade.VLBISkyModels.AbstractFourierDualDomain)
	img = IntensityMap(rand(size(grid.imgdomain)...), grid.imgdomain)
	Comrade.intensitymap!(img, m)
    vis = Comrade.VLBISkyModels.applyft(Comrade.VLBISkyModels.forward_plan(grid), img)
    return vis
end

@inline function Krang.emission_coordinates_fast_light(
    pix::Krang.AbstractPixel,
    θs::T,
    isindir,
    n,
) where {T}
    # NB: I do not return θs since it is already known, and returning it might encourage bad coding errors
    α, β = Krang.screen_coordinate(pix)
    θo = Krang.inclination(pix)
    met = Krang.metric(pix)
    isincone = θo ≤ θs ≤ (π - θo) || (π - θo) ≤ θs ≤ θo
    err_return = (zero(T), zero(T), false, false, false, 0.0)
    if !isincone
        αmin = Krang.αboundary(met, θs)
        βbound = (abs(α) >= (αmin + eps(T)) ? Krang.βboundary(met, α, θo, θs) : zero(T))
        ((abs(β) + eps(T)) < βbound) && return err_return
    end

    τ, Gs, Go, Ghat, isvortical, issuccess = @inline Krang.Gθ(pix, θs, isindir, n)
    issuccess || return err_return

    nmax = floor((Krang.total_mino_time(pix) - τ)/Ghat)
    # is θ̇s increasing or decreasing?
    νθ = isincone ? (θo > θs) ⊻ (n % 2 == 1) : !isindir
    if isincone
        νθ = (θo > θs) ⊻ (n % 2 == 1)
    end

    rs, νr, _, issuccess = Krang.emission_radius(pix, τ)
    issuccess || return err_return

    ϕs = @inline Krang.emission_azimuth(pix, θs, rs, τ, νr, isindir, n)

    return rs, ϕs, νr, νθ, issuccess, nmax
end

@inline function _raytrace(
    pix::Krang.AbstractPixel,
    mesh::Krang.Mesh{<:Krang.ConeGeometry{T,A},<:Krang.AbstractMaterial};
) where {T,A}
    observation = 0.0
    geometry = mesh.geometry
    material = mesh.material
    θs = geometry.opening_angle
    subimgs = material.subimgs

    nmax = 0.0
    isindir = false
    for _ = 1:2 # Looping over isindir this way is needed to get Metal to work
        isindir ⊻= true
        for n in subimgs
            rs, ϕs, νr, νθ, issuccess, ntemp =
                @inline Krang.emission_coordinates_fast_light(pix, θs, isindir, n)
            intersection = Krang.Intersection(zero(rs), rs, θs, ϕs, νr, νθ)
            if issuccess && (Krang.horizon(Krang.metric(pix)) < rs < T(Inf))
                observation += @inline(material(pix, intersection))
            end
            nmax = max(ntemp, nmax)
        end
    end

    return observation, nmax
end

@inline function _render(pixel::Krang.AbstractPixel, scene::Krang.Scene) 
    observation = 0.0
    mesh = scene[1]
    nmax = 0.0
    for itr = 1:length(scene)
        mesh = scene[itr]
        temp, ntemp = @inline _raytrace(pixel, mesh)
        observation += temp
        nmax = max(nmax, ntemp)
    end
    return observation, nmax
end
