struct EmissivityModel{N, T, B} <: Krang.AbstractMaterial
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
	subimgs::NTuple{N, Int}

	function EmissivityModel(magfield, vel, bulkmodel::B, spec::T, rpeak::T, p1::T, p2::T, m_d::T, raster_size::T, offset::T) where {T, B}
		new{2, T, B}(magfield, vel, bulkmodel, spec, rpeak, p1, p2, m_d, raster_size, offset, (0, 1))
	end
end
Krang.isFastLight(::EmissivityModel) = true
Krang.isAxisymmetric(::EmissivityModel) = false

@inline function (prof::EmissivityModel{N, B})(pix::Krang.AbstractPixel, intersection; n = 0) where {N, B}
	(; m_d, magnetic_field, fluid_velocity, bulkmodel, spectral_index, rpeak, p1, p2, raster_size, offset) = prof
	(; rs, ϕs, θs, νr, νθ) = intersection

	θo = Krang.inclination(pix)
	met = Krang.metric(pix)
	T = typeof(met.spin)
	α, β = @inline Krang.screen_coordinate(pix)

	norm, redshift, lp = @inline Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magnetic_field, fluid_velocity, νr, νθ)

	rh = Krang.horizon(met)

	rs_grid = (rs - rh) * rad2μas(m_d) / (raster_size - rh * rad2μas(m_d)) # convert to microarcseconds
	if rs_grid < 0
		return zero(T)
	end

	ϕks = Krang.ϕ_kerr_schild(met, rs, ϕs)
	dim = (X = rs_grid * cos(ϕks) + offset, Y = rs_grid * sin(ϕks) + offset)
	rat = (rs / rpeak)

	bulkpix = bulkmodel.img.X.len
	cp = exp(ComradeBase.intensity_point(bulkmodel, dim) / (bulkpix^2))
	ans = rat^p1 / (one(T) + rat^(p1 + p2)) * max(redshift, eps(T))^(T(3) + spectral_index) * cp
	ans = norm^(1 + spectral_index) * min(lp, 1e2) * ans
	# Add a clamp to lp to help remove hot pixels
	return ans
end

