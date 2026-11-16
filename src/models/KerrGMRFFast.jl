# Faster, point-sampled version of the modified KerrGMRF model. It renders the same image as
#     modify(KerrGMRF(θ, meta), Stretch(m_d, m_d), Rotate(pa))
# (up to floating-point rounding), but is cheaper to evaluate and much cheaper to differentiate:
#
# 1. The image is rendered by a direct threaded loop over the pixel array (an intensitymap_analytic!
#    overload), with the Stretch and Rotate applied per ray. Comrade's generic executor path is ~2.4x
#    slower to differentiate at 90x90 (see benchmark_executor.jl, now in
#    KerrGMRF_fast_validation/attic/completed_experiments_2026-10-21.tar.gz), and wrapping the
#    model in modify(…) would bypass this overload. The model struct holds only what the rays need
#    (no metadata, transforms or bulk grids).
# 2. Rays use FastLightPixel, which caches only the quantities fast-light tracing reads.
#    Krang.SlowLightIntensityPixel also computes the radial t and I1/I2/I± integrals and Gt,
#    which only slow-light (time-dependent) tracing uses.
# 3. The GMRF field is interpolated with the same 4-point Lagrange (bicubic) stencils as
#    BasicInterpolators.BicubicInterpolator, but with direct cell indexing and closed-form
#    weights on the uniform bulk grid (no bisection, no per-evaluation copies of the grid); see
#    GMRFFieldInterpolation.jl, which this file includes.
# 4. The emissivity power laws and exp(GMRF) are combined into a single exp of a sum of logs.
#
# Use in a fit as
#     RenormalizedFlux(KerrGMRFFast(θ, meta; pa), θ.f)
# with the same metadata as KerrGMRF, m_d in radians and pa in DEGREES (including any 180° flip for
# θo < 90°). Do not wrap it in modify(…, Stretch, Rotate): that would apply the transforms twice.
# This file does not change KerrGMRF.jl or the files it includes, which running fits depend on.
#
# Optional smoothing (metadata `smooth = Val(true)`; without it the model is unchanged): removes jumps
# and kinks in the emissivity that NUTS sees as divergences.
#   - The GMRF field is tapered to zero over the outer part of its disk (radius 0.4 → 0.5 in grid
#     units, C¹ smoothstep in ρ²). Otherwise exp(g) jumps from exp(σimg·c) to 1 where rays leave the grid.
#   - The lp clamp min(lp, 100) becomes the soft minimum lp - softplus(lp - 100).
# Both are selected by the field's type (TaperedField), so the default path compiles exactly as before.
#
# JacobiElliptic algorithms: the per-function choice of src/fast_model/jacobi_algs.jl (Fukushima where it is
# faster in the reverse pass, Carlson for the incomplete E, Π and J) is applied here, ~4% off the gradient with the
# same values to ~1e-11. Set JE_ALGS=none to use JacobiElliptic's defaults.

import JacobiElliptic
@isdefined(_je_spec) || include(joinpath(dirname(@__DIR__), "fast_model", "jacobi_algs.jl"))

# ---------------------------------------------------------------------------------------------
# GMRF field: interpolation of the bulk grid (square or polar), in GMRFFieldInterpolation.jl
# ---------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "GMRFFieldInterpolation.jl"))

# ---------------------------------------------------------------------------------------------
# Pixel: only what fast-light tracing needs
# ---------------------------------------------------------------------------------------------

struct FastLightPixel{T, R, G1, G2} <: Krang.AbstractPixel
	metric::Krang.Kerr{T}
	screen_coordinate::NTuple{2, T}
	roots::R
	θo::T
	η::T
	λ::T
	I0_inf::T
	τtotal::T
	Iϕ_inf::T
	absGθo_Gθhat::G1
	absGϕo_Gϕhat::G2
end

# Rounding can put the second radial root of a captured ray (two real roots) at or above the inner horizon r₋,
# which it only touches in exact arithmetic: on the image line λ = (r₋² + a²)/a, with r₂ - r₋ quadratic in the
# distance to it, so a band ~1e-7 wide in α rounds to r₂ ≥ r₋. There Krang's Iϕ_inf_case3 evaluates 0 × ∞ (NaN),
# and one ulp above r₋ gives a wrong value. Even where the pixel emits nothing, Enzyme's reverse pass carries the
# NaN into the gradient (0 × NaN), which crashed AutoMALA's step-size search at npix 90 (2026-10-12). Moving r₂
# ~100 ulps below r₋ reproduces the nearby exact values of Iϕ_inf to 1e-8 with a finite gradient
# (diagnose_nan_gradient.jl, now in KerrGMRF_fast_validation/attic/completed_experiments_2026-10-21.tar.gz). Other pixels are unchanged.
# The test only decides the branch, so it is excluded from differentiation, and it cannot throw even on an
# out-of-range spin: Enzyme's reverse pass of the threaded pixel loop recomputed a √(1 - a²) here with a spin
# outside [0, 1] (DomainError), presumably reading gradient storage under runtime activity. The fix moves r₂ down
# by 128 ulps (a few ulps below r₋, since the test fires within 64 ulps of it) without recomputing r₋.
function _r2_at_inner_horizon(a::T, r2::T) where {T}
	rm = 1 - √(max(1 - a * a, zero(T)))
	return r2 ≥ rm - 64 * eps(rm)
end
Enzyme.EnzymeRules.inactive(::typeof(_r2_at_inner_horizon), args...) = nothing
@inline function _roots_below_rm(met::Krang.Kerr{T}, roots) where {T}
	sum(Krang._isreal2.(roots)) == 2 || return roots
	r1, r2, r3, r4 = roots
	_r2_at_inner_horizon(met.spin, real(r2)) || return roots
	return (r1, r2 - 128 * eps(real(r2)), r3, r4)
end

# Same quantities, computed the same way, as Krang.SlowLightIntensityPixel (including its root order), except for
# the root guard above.
function FastLightPixel(met::Krang.Kerr{T}, α, β, θo) where {T}
	η = Krang.η(met, α, β, θo)
	λ = Krang.λ(met, α, θo)
	roots = _roots_below_rm(met, Krang.get_radial_roots(met, η, λ))
	return FastLightPixel(met, (α, β), roots, θo, η, λ,
		Krang.Ir_inf(met, roots), Krang.total_mino_time(met, roots), Krang.Iϕ_inf(met, roots, λ),
		Krang._absGθo_Gθhat(met, θo, η, λ), Krang._absGϕo_Gϕhat(met, θo, η, λ))
end
@inline Krang.η(p::FastLightPixel) = p.η
@inline Krang.λ(p::FastLightPixel) = p.λ
@inline Krang.roots(p::FastLightPixel) = p.roots
@inline Krang.I0_inf(p::FastLightPixel) = p.I0_inf
@inline Krang.Ir_inf(p::FastLightPixel) = p.I0_inf
@inline Krang.total_mino_time(p::FastLightPixel) = p.τtotal
@inline Krang.Iϕ_inf(p::FastLightPixel) = p.Iϕ_inf
@inline Krang.absGθo_Gθhat(p::FastLightPixel) = p.absGθo_Gθhat
@inline Krang.absGϕo_Gϕhat(p::FastLightPixel) = p.absGϕo_Gϕhat

# ---------------------------------------------------------------------------------------------
# Emissivity (EmissivityModel.jl, rewritten)
# ---------------------------------------------------------------------------------------------

struct FastEmissivity{T, B}
	magnetic_field::Krang.SVector{3, T}
	fluid_velocity::Krang.SVector{3, T}
	bulk::B
	spectral_index::T
	rpeak::T
	p1::T
	p2::T
	uas_per_M::T # rad2μas(m_d)
	raster_size::T
	offset::T
	boost::Krang.StaticArrays.SMatrix{4, 4, T, 16} # ZAMO → fluid boost, from fluid_velocity (the same for every ray)
	ϕshift::T # added to the emission azimuth before the field lookup (π: the field rotated by 180° about the spin axis)
end
# The fluid boost depends only on the fluid velocity (βv, θz, ϕz), so it is built once per model instead of once per
# sub-image (Krang.jac_fluid_u_zamo_d does not use the metric). Krang applies the boost to a vector; it is linear,
# so its columns are the boosted basis vectors.
function _fluid_boost(v::Krang.SVector{3, T}) where {T}
	met = Krang.Kerr(zero(T))
	e(i) = Krang.SVector(ntuple(j -> T(j == i), 4))
	return hcat(ntuple(i -> Krang.jac_fluid_u_zamo_d(met, v[1], v[2], v[3], e(i)), 4)...)
end
FastEmissivity(B, v, bulk, spec, rpeak, p1, p2, uas, raster, offset, ϕshift = zero(offset)) =
	FastEmissivity(B, v, bulk, spec, rpeak, p1, p2, uas, raster, offset, _fluid_boost(v), ϕshift)

# Krang.synchrotronIntensity, reusing η and λ stored on the pixel instead of recomputing them.
# Fast path (used by the model): the photon momentum p_μ = (-1, ±√R/Δ, ±√Θ, λ) is raised with the closed-form
# inverse Kerr metric (only the t-ϕ block couples; Krang.metric_uu inverts the full 4×4 matrix), projected onto
# the ZAMO frame with the nonzero entries of Krang.jac_zamo_u_bl_d written out, and boosted with the precomputed
# fluid boost. Same result as the Krang path below to rounding (fast_model/test_emission_opt.jl).
@inline function _synchrotron(pix::FastLightPixel, ri, θs, magnetic_field, boost::Krang.StaticArrays.SMatrix{4, 4}, νr, θsign)
	met = pix.metric
	T = typeof(ri)
	a = met.spin
	a2 = a * a
	s, c = sin(θs), cos(θs)
	s2 = s * s
	r2 = ri * ri
	Σt = r2 + a2 * c * c
	Δt = r2 - 2ri + a2
	At = (r2 + a2)^2 - a2 * Δt * s2
	λ = pix.λ
	pr_d = (νr ? one(T) : -one(T)) * √max(zero(T), Krang.r_potential(met, pix.η, λ, ri)) / Δt
	pθ_d = (θsign ? one(T) : -one(T)) * √max(zero(T), Krang.θ_potential(met, pix.η, λ, θs))
	ΣΔ = Σt * Δt
	# p^μ = g^{μν} p_ν with p_t = -1, p_ϕ = λ: g^tt = -A/(ΣΔ), g^tϕ = -2ar/(ΣΔ), g^ϕϕ = (Δ - a² sin²θ)/(ΣΔ sin²θ)
	pt_u = (At - 2a * ri * λ) / ΣΔ
	pr_u = Δt / Σt * pr_d
	pθ_u = pθ_d / Σt
	pϕ_u = (2a * ri + (Δt - a2 * s2) * λ / s2) / ΣΔ
	# ZAMO components (Krang.jac_zamo_u_bl_d, output order t, r, ϕ, θ)
	p_zamo_u = Krang.SVector(√(ΣΔ / At) * pt_u, √(Σt / Δt) * pr_u,
		-(2a * ri * s) / √(At * Σt) * pt_u + √(At / Σt) * s * pϕ_u, -√Σt * pθ_u)
	p_fluid_u = boost * p_zamo_u
	bx, by, bz = magnetic_field
	_, ux, uy, uz = p_fluid_u ./ p_fluid_u[1]
	vx, vy, vz = uy * bz - uz * by, uz * bx - ux * bz, ux * by - uy * bx
	nrm = √(vx * vx + vy * vy + vz * vz) + eps(T)
	return nrm, inv(p_fluid_u[1]), abs(p_fluid_u[1] / p_fluid_u[4])
end

# Krang path (Krang.synchrotronIntensity's frames, reusing η and λ from the pixel): kept for comparison and for
# callers that pass the fluid velocity instead of the boost.
@inline function _synchrotron(pix::FastLightPixel, ri, θs, magnetic_field, βfluid::Krang.SVector{3}, νr, θsign)
	met = pix.metric
	T = typeof(ri)
	βv, θz, ϕz = βfluid
	curr_p_bl_d = Krang.p_bl_d(met, ri, θs, pix.η, pix.λ, νr, θsign)
	curr_p_bl_u = Krang.metric_uu(met, ri, θs) * curr_p_bl_d
	p_zamo_u = Krang.jac_zamo_u_bl_d(met, ri, θs, curr_p_bl_u)
	p_fluid_u = Krang.jac_fluid_u_zamo_d(met, βv, θz, ϕz, p_zamo_u)
	bx, by, bz = magnetic_field
	_, ux, uy, uz = p_fluid_u ./ p_fluid_u[1]
	vx, vy, vz = uy * bz - uz * by, uz * bx - ux * bz, ux * by - uy * bx
	nrm = √(vx * vx + vy * vy + vz * vz) + eps(T)
	return nrm, inv(p_fluid_u[1]), abs(p_fluid_u[1] / p_fluid_u[4])
end

# log(1 + exp(x)) without overflow
@inline _softplus(x) = x > zero(x) ? x + log1p(exp(-x)) : log1p(exp(x))

# Clamp on lp to help remove hot pixels: hard by default, soft with smoothing
@inline _clamp_lp(_, lp::T) where {T} = min(lp, T(1e2))
@inline _clamp_lp(::TaperedField, lp::T) where {T} = lp - _softplus(lp - T(1e2))
@inline _clamp_lp(::PolarField, lp::T) where {T} = lp - _softplus(lp - T(1e2))

@inline function _emission(mat::FastEmissivity{T}, pix::FastLightPixel, rs, θs, ϕs, νr, νθ, rh) where {T}
	(; uas_per_M, raster_size, offset) = mat
	rs_grid = (rs - rh) * uas_per_M / (raster_size - rh * uas_per_M)
	rs_grid < 0 && return zero(T)

	g = _field_at(mat.bulk, rs_grid, Krang.ϕ_kerr_schild(pix.metric, rs, ϕs) + mat.ϕshift, offset)
	return _emission_profile(mat, pix, rs, θs, νr, νθ, g)
end

# The emission without the field (exp(g) = 1): axisymmetric, so it needs no azimuth (n1_field = Val(false))
@inline function _emission_nofield(mat::FastEmissivity{T}, pix::FastLightPixel, rs, θs, νr, νθ, rh) where {T}
	(; uas_per_M, raster_size) = mat
	(rs - rh) * uas_per_M / (raster_size - rh * uas_per_M) < 0 && return zero(T)
	return _emission_profile(mat, pix, rs, θs, νr, νθ, zero(T))
end

@inline function _emission_profile(mat::FastEmissivity{T}, pix::FastLightPixel, rs, θs, νr, νθ, g) where {T}
	(; spectral_index, rpeak, p1, p2) = mat
	nrm, redshift, lp = _synchrotron(pix, rs, θs, mat.magnetic_field, mat.boost, νr, νθ)

	# rat^p1 / (1 + rat^(p1+p2)) * redshift^(3+s) * norm^(1+s) * exp(g), as one exp
	lr = log(rs / rpeak)
	lg = p1 * lr - _softplus((p1 + p2) * lr) +
		(3 + spectral_index) * log(max(redshift, eps(T))) + (1 + spectral_index) * log(nrm) + g
	return exp(lg) * _clamp_lp(mat.bulk, lp)
end

# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

struct KerrGMRFFast{A, M1, M2, N1, DK} <: ComradeBase.AbstractModel
	met::Krang.Kerr{A}
	θo::A # radians
	θs1::A # cone opening angles (radians)
	θs2::A
	mat1::M1
	mat2::M2
	m_d::A # radians
	s::A # sin(position angle)
	c::A # cos(position angle)
	n1_field::N1 # Val(true): the n = 1 image is modulated by the field (default); Val(false): mean emissivity only
	disk::DK # Val(true): intensitymap traces only the pixels inside the disk inscribed in the grid, the rest are 0 (opt-in)
end
# the positional forms without the switches (n = 1 modulated, full square, as before)
KerrGMRFFast(met, θo, θs1, θs2, mat1, mat2, m_d, s, c) = KerrGMRFFast(met, θo, θs1, θs2, mat1, mat2, m_d, s, c, Val(true))
KerrGMRFFast(met, θo, θs1, θs2, mat1, mat2, m_d, s, c, n1) = KerrGMRFFast(met, θo, θs1, θs2, mat1, mat2, m_d, s, c, n1, Val(false))

function KerrGMRFFast(θ, metadata; pa = θ.pa, ϕshift = 0) # pa in degrees; ϕshift (radians) rotates the field about the spin axis
	(; m_d, spin, θs, θo, χ, ι, βv, spec, η, rpeak, p1, p2, ρpr, νpr, c1, c2, σimg) = θ
	(; transform1, transform2, raster_size, offset) = metadata
	A = typeof(θo)
	sm = _smoothing(metadata)
	bulk1, bulk2 = _fields(metadata, σimg * transform1(c1, ρpr, νpr), σimg * transform2(c2, ρpr, νpr), sm)

	vel = Krang.SVector(βv, A(π / 2), χ)
	magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
	magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
	uas = rad2μas(m_d)
	mat1 = FastEmissivity(magfield1, vel, bulk1, spec, rpeak, p1, p2, uas, A(raster_size), A(offset), A(ϕshift))
	mat2 = FastEmissivity(magfield2, vel, bulk2, spec, rpeak, p1, p2, uas, A(raster_size), A(offset), A(ϕshift))

	s, c = sincos(pa * A(π) / 180)
	n1 = haskey(metadata, :n1_field) ? metadata.n1_field : Val(true)
	dk = haskey(metadata, :disk) ? metadata.disk : Val(false)
	return KerrGMRFFast(Krang.Kerr(spin), θo * A(π) / 180, θs * A(π) / 180, A(π) - θs * A(π) / 180, mat1, mat2, m_d, s, c, n1, dk)
end

ComradeBase.imanalytic(::Type{<:KerrGMRFFast}) = ComradeBase.IsAnalytic()
ComradeBase.visanalytic(::Type{<:KerrGMRFFast}) = ComradeBase.NotAnalytic()
ComradeBase.ispolarized(::Type{<:KerrGMRFFast}) = ComradeBase.NotPolarized()

# n = 0 and 1 images of one cone, both isindir branches (as in overloads.jl). The four sub-images are written out as
# separate calls rather than a loop: inside a loop Enzyme caches each iteration's coordinates and emission
# intermediates in heap buffers allocated with malloc on every pixel, which took ~45% of the reverse pass; unrolled,
# the cache is fixed-size (reverse pass of a pixel 99 → 56 ms per 60×60 image on one thread, values identical,
# profile_full_gradient.jl, 2026-10-17, now in KerrGMRF_fast_validation/attic/completed_experiments_2026-10-21.tar.gz).
# The angular solve (Gθ at θs, the cone-boundary test) depends on the cone, not on the sub-image, so it is done once per
# cone (Krang.AngularCache) and shared by the four sub-images.
@inline function _sub_emission(pix::FastLightPixel, cache::Krang.AngularCache{T}, mat, rh, isindir, n) where {T}
	rs, ϕs, νr, νθ, issuccess = @inline Krang.emission_coordinates_fast_light(pix, cache, isindir, n)
	return (issuccess && (rh < rs < T(Inf))) ? _emission(mat, pix, rs, cache.θs, ϕs, νr, νθ, rh) : zero(T)
end
@inline function _fast_cone(pix::FastLightPixel, θs::T, mat, rh, ::Val{true} = Val(true)) where {T}
	cache = @inline Krang.AngularCache(pix, θs)
	return _sub_emission(pix, cache, mat, rh, true, 0) + _sub_emission(pix, cache, mat, rh, false, 0) +
		_sub_emission(pix, cache, mat, rh, true, 1) + _sub_emission(pix, cache, mat, rh, false, 1)
end

# metadata `n1_field = Val(false)`: the n = 1 image carries no field (as in the 2017 data fits,
# KerrGMRF_no_n1_variability.jl before commit 3cd5af1), so it is axisymmetric and needs only the emission
# radius (Krang's axisymmetric path), not the azimuth. Unrolled as above.
@inline function _sub_emission_nofield(pix::FastLightPixel, cache::Krang.AngularCache{T}, mat, rh, isindir) where {T}
	rs, νr, νθ, _, issuccess = @inline Krang.emission_radius(pix, cache, isindir, 1)
	return (issuccess && (rh < rs < T(Inf))) ? _emission_nofield(mat, pix, rs, cache.θs, νr, νθ, rh) : zero(T)
end
@inline function _fast_cone(pix::FastLightPixel, θs::T, mat, rh, ::Val{false}) where {T}
	cache = @inline Krang.AngularCache(pix, θs)
	return _sub_emission(pix, cache, mat, rh, true, 0) + _sub_emission(pix, cache, mat, rh, false, 0) +
		_sub_emission_nofield(pix, cache, mat, rh, true) + _sub_emission_nofield(pix, cache, mat, rh, false)
end

# Intensity at sky position (X, Y) in radians, with the Stretch by m_d and the Rotate by pa applied
# per ray: the same screen coordinates as Stretch(m_d) ∘ Rotate(pa) followed by KerrGMRF's (-X, Y).
@inline function ComradeBase.intensity_point(m::KerrGMRFFast, p)
	α = -(m.c * p.X - m.s * p.Y) / m.m_d
	β = (m.s * p.X + m.c * p.Y) / m.m_d
	pix = FastLightPixel(m.met, α, β, m.θo)
	rh = Krang.horizon(m.met)
	return (_fast_cone(pix, m.θs1, m.mat1, rh, m.n1_field) + _fast_cone(pix, m.θs2, m.mat2, rh, m.n1_field)) / m.m_d^2
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap, m::KerrGMRFFast)
	g = axisdims(img)
	X, Y = g.X, g.Y
	bimg = baseimage(img)
	scale = abs(step(X) * step(Y))
	r2 = _disk_radius2(m.disk, X, Y)
	ComradeBase.@threaded executor(img) for I in CartesianIndices(bimg)
		x, y = X[I[1]], Y[I[2]]
		bimg[I] = x * x + y * y > r2 ? zero(eltype(bimg)) : ComradeBase.intensity_point(m, (X = x, Y = y)) * scale
	end
	return nothing
end

# metadata `disk = Val(true)`: the squared radius of the disk inscribed in the grid (half its full extent on the
# shorter side, about the grid centre at 0); pixel centres outside it are not traced
_disk_radius2(::Val{false}, X, Y) = Inf
_disk_radius2(::Val{true}, X, Y) = (min(length(X) * abs(step(X)), length(Y) * abs(step(Y))) / 2)^2
