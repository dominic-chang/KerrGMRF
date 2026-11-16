# GMRF field interpolation for KerrGMRFFast (split out of KerrGMRFFast.jl, 2026-10-21; included by it, so include
# KerrGMRFFast.jl, not this file). The field multiplies the emissivity by exp(g), with g looked up at each sub-image's
# emission point:
#   UniformLagrange   the square bulk grid: 4-point Lagrange (bicubic) stencils as BasicInterpolators'
#                     BicubicInterpolator, with direct cell indexing and closed-form weights
#   UniformLagrangeRule  the same with a custom Enzyme reverse rule (opt-in: metadata `field_rule = Val(true)`)
#   TaperedField      the smoothing option's taper to zero at the grid's edge (metadata `smooth = Val(true)`)
#   PolarField        the field on a polar (disk) grid (opt-in: metadata `polar = (; rmax)`)
#   _field_at         the field value of a sub-image at grid radius rs_grid and azimuth ϕ
#   _fields           the two fields (one per cone) from the transformed coefficients and the metadata

# ---------------------------------------------------------------------------------------------
# GMRF field: cubic Lagrange interpolation on a uniform grid
# ---------------------------------------------------------------------------------------------

import Enzyme

struct UniformLagrange{T, M <: AbstractMatrix{T}}
	Z::M
	x0::T
	y0::T
	xend::T
	yend::T
	dx::T
	dy::T
	nx::Int
	ny::Int
	scale::T # value multiplier, matching KerrGMRF's intensity_point(bulk) / bulkpix^2
end

function UniformLagrange(Z::AbstractMatrix{T}, bulkgrid) where {T}
	X, Y = bulkgrid.X, bulkgrid.Y
	nx, ny = length(X), length(Y)
	dx, dy = T(step(X)), T(step(Y))
	return UniformLagrange(Z, T(first(X)), T(first(Y)), T(last(X)), T(last(Y)), dx, dy, nx, ny, inv(dx * dy * nx^2))
end

# First stencil node and Lagrange weights for coordinate q, matching BasicInterpolators' findcell
# and its boundary stencils (1:4 in the first cell, n-3:n in the last).
@inline function _stencil(q::T, q0, h, n) where {T}
	i = clamp(floor(Int, (q - q0) / h) + 1, 1, n - 1)
	i0 = i == 1 ? 1 : (i == n - 1 ? n - 3 : i - 1)
	t = (q - q0) / h - (i0 - 1)
	tm1, tm2, tm3 = t - 1, t - 2, t - 3
	w = (-tm1 * tm2 * tm3 / 6, t * tm2 * tm3 / 2, -t * tm1 * tm3 / 2, t * tm1 * tm2 / 6)
	return i0, w
end

# The grid constants of a UniformLagrange (no field values), passed to the lookup as one constant argument
struct _LagrangeGrid{T}
	x0::T
	y0::T
	xend::T
	yend::T
	dx::T
	dy::T
	nx::Int
	ny::Int
	scale::T
end
_grid(b::UniformLagrange) = _LagrangeGrid(b.x0, b.y0, b.xend, b.yend, b.dx, b.dy, b.nx, b.ny, b.scale)

@inline (b::UniformLagrange)(x, y) = _lagrange_body(b.Z, x, y, _grid(b))

# The same field with the custom reverse rule of _lagrange_value below (opt-in: metadata `field_rule = Val(true)`).
# The rule cut the emission step's single-thread reverse pass by 42%, but the full gradient was ~5% SLOWER at 12-16
# threads and level at 48 (atomic additions into the shared field gradient; per-thread buffers were slower still,
# 2026-10-14), so it is off by default.
struct UniformLagrangeRule{U <: UniformLagrange}
	field::U
end
@inline (b::UniformLagrangeRule)(x, y) = _lagrange_value(b.field.Z, x, y, _grid(b.field))
_maybe_rule(f, ::Val{false}) = f
_maybe_rule(f, ::Val{true}) = UniformLagrangeRule(f)

# The lookup itself
@inline _lagrange_value(Z::AbstractMatrix, x, y, c::_LagrangeGrid) = _lagrange_body(Z, x, y, c)
@inline function _lagrange_body(Z::AbstractMatrix{T}, x, y, c::_LagrangeGrid) where {T}
	# A NaN coordinate (e.g. a NaN emission azimuth) fails both bounds checks and would make the integer
	# cell index in _stencil throw; return NaN so the log density is NaN (an HMC divergence) instead.
	(isnan(x) || isnan(y)) && return T(NaN)
	(c.x0 > x || x > c.xend) && return zero(T)
	(c.y0 > y || y > c.yend) && return zero(T)
	i0, wx = _stencil(x, c.x0, c.dx, c.nx)
	j0, wy = _stencil(y, c.y0, c.dy, c.ny)
	acc = zero(T)
	@inbounds for j in 1:4
		col = wx[1] * Z[i0, j0 + j - 1] + wx[2] * Z[i0 + 1, j0 + j - 1] +
			wx[3] * Z[i0 + 2, j0 + j - 1] + wx[4] * Z[i0 + 3, j0 + j - 1]
		acc += wy[j] * col
	end
	return acc * c.scale
end

# ---- Custom Enzyme reverse rule for the lookup (2026-10-14). The reverse pass of the emission step was ~34× its
# forward (profile_reverse_stages.jl, now in KerrGMRF_fast_validation/attic/completed_experiments_2026-10-21.tar.gz), much of it Enzyme recording and replaying the stencil indices, the weight
# polynomials and the 16-point sum for every sub-image. The lookup is linear in Z and its weights are cubic
# polynomials, so the adjoint is closed form and needs nothing stored: the reverse pass recomputes the stencil and
#     x̄ += ȳ·scale·Σᵢⱼ wᵢ'(x) wⱼ(y) Z[i,j]/dx,   ȳ_y += ȳ·scale·Σᵢⱼ wᵢ(x) wⱼ'(y) Z[i,j]/dy,
#     Z̄[i,j] += ȳ·scale·wᵢ(x) wⱼ(y)   (atomically: the pixel loop runs in parallel).
# Same values as Enzyme's own derivative to rounding (fast_model/test_field_rule.jl, now in
# KerrGMRF_fast_validation/attic/completed_experiments_2026-10-21.tar.gz).
@inline function _dstencil(t)
	tm1, tm2, tm3 = t - 1, t - 2, t - 3
	return (-(tm2 * tm3 + tm1 * tm3 + tm1 * tm2) / 6, (tm2 * tm3 + t * tm3 + t * tm2) / 2,
		-(tm1 * tm3 + t * tm3 + t * tm1) / 2, (tm1 * tm2 + t * tm2 + t * tm1) / 6)
end
@inline function _atomic_add!(A::Array{T}, i::Int, v::T) where {T}
	GC.@preserve A Base.unsafe_modify!(pointer(A, i), +, v, :monotonic)
	return nothing
end
@inline _atomic_add!(A::AbstractArray, i::Int, v) = (A[i] += v; nothing)

function Enzyme.EnzymeRules.augmented_primal(config::Enzyme.EnzymeRules.RevConfigWidth{1},
		func::Enzyme.Const{typeof(_lagrange_value)}, ::Type{RT}, Z, x, y, c) where {RT}
	v = func.val(Z.val, x.val, y.val, c.val)
	return Enzyme.EnzymeRules.AugmentedReturn(Enzyme.EnzymeRules.needs_primal(config) ? v : nothing,
		Enzyme.EnzymeRules.needs_shadow(config) ? zero(v) : nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(config::Enzyme.EnzymeRules.RevConfigWidth{1},
		::Enzyme.Const{typeof(_lagrange_value)}, dret, tape, Z, x, y, c)
	xv, yv, g, Zv = x.val, y.val, c.val, Z.val
	ȳ = dret isa Enzyme.Active ? dret.val : zero(eltype(Zv))
	gx, gy = zero(xv), zero(yv)
	inside = !(isnan(xv) || isnan(yv)) && g.x0 <= xv <= g.xend && g.y0 <= yv <= g.yend
	if inside && !iszero(ȳ)
		i0, wx = _stencil(xv, g.x0, g.dx, g.nx)
		j0, wy = _stencil(yv, g.y0, g.dy, g.ny)
		dwx = _dstencil((xv - g.x0) / g.dx - (i0 - 1))
		dwy = _dstencil((yv - g.y0) / g.dy - (j0 - 1))
		s = ȳ * g.scale
		addZ = Z isa Enzyme.Duplicated && Z.dval !== Zv # runtime activity: a shadow aliasing the primal is inactive
		nrow = size(Zv, 1)
		@inbounds for j in 1:4, i in 1:4
			z = Zv[i0 + i - 1, j0 + j - 1]
			gx += dwx[i] * wy[j] * z
			gy += wx[i] * dwy[j] * z
			addZ && _atomic_add!(Z.dval, (j0 + j - 2) * nrow + i0 + i - 1, s * wx[i] * wy[j])
		end
		gx *= s / g.dx
		gy *= s / g.dy
	end
	# The grid constants carry no gradient, but Enzyme can still pass them as Active (e.g. captured in a closure):
	# an Active argument needs a (zero) tangent of its own type.
	return (nothing, x isa Enzyme.Active ? gx : nothing, y isa Enzyme.Active ? gy : nothing, _zero_grid(c))
end
_zero_grid(c::Enzyme.Active{_LagrangeGrid{T}}) where {T} = (z = zero(T); _LagrangeGrid(z, z, z, z, z, z, c.val.nx, c.val.ny, z))
_zero_grid(_) = nothing

# Smoothing wrapper (see the header). Assumes offset = 0, so that (x, y) is centered on the hole.
struct TaperedField{B}
	field::B
end
@inline function (t::TaperedField)(x::T, y) where {T}
	ρ² = x * x + y * y
	u = clamp((T(0.25) - ρ²) / T(0.25 - 0.16), zero(T), one(T)) # 1 inside ρ = 0.4, 0 outside ρ = 0.5
	return t.field(x, y) * (u * u * (3 - 2u))
end
_maybe_taper(field, ::Val{false}) = field
_maybe_taper(field, ::Val{true}) = TaperedField(field)
_smoothing(metadata) = haskey(metadata, :smooth) ? metadata.smooth : Val(false)

# ---------------------------------------------------------------------------------------------
# GMRF field on a polar (disk) grid (opt-in: metadata `polar = (; rmax)`)
# ---------------------------------------------------------------------------------------------
# The square field grid above is addressed by rs_grid·(cos ϕ, sin ϕ), so its corners never touch the emission.
# The polar field instead lives on an nu × nφ grid in
#     u = rs_grid / rmax ∈ [0, 1]                    (radial)
#     φ = the Kerr-Schild emission azimuth ∈ [0, 2π) (periodic)
# where rs_grid = (rs - r_h)·(M/D) / (raster_size - r_h·(M/D)) is the same embedding as the square grid's: it is
# set by the sky scale (μas, through raster_size), not by the mass, so the field keeps its angular resolution when
# the black hole appears smaller. rmax is in those grid units (0.5 matches the square grid's extent, i.e. emission
# radius times M/D up to about raster_size/2). Cubic Lagrange interpolation in u (as UniformLagrange), periodic
# cubic interpolation in φ, and a smooth taper to zero over the outer 20% of u (emission beyond rmax is
# unmodulated). Its prior is the same Matern GMRF built on the (nu, nφ) grid: periodic in φ as it should be; the
# periodic wrap in u only couples the horizon edge to the tapered outer edge, whose values are multiplied by ~0.
# Cells are rmax·raster/(nu - 1) μas radially and 2π/nφ azimuthally (2π·ρ/nφ μas at sky radius ρ), so
# nu ≈ nφ·rmax·raster/(2π·ρ) gives square cells at ρ (ρ ≈ 20 μas at the ring). Correlation lengths (ρpr) are in
# grid cells. The lp clamp is always soft with this field (as with the smoothing option).
struct PolarField{T, M <: AbstractMatrix{T}}
	Z::M # nu × nφ
	rmax::T # outer radius, in the square grid's units (rs_grid)
	nu::Int
	nφ::Int
end
PolarField(Z::AbstractMatrix{T}, rmax) where {T} = PolarField(Z, T(rmax), size(Z, 1), size(Z, 2))

@inline function (b::PolarField{T})(rs_grid, ϕ) where {T}
	(isnan(rs_grid) || isnan(ϕ)) && return T(NaN)
	u = rs_grid / b.rmax
	(u < 0 || u >= 1) && return zero(T)
	nu, nφ = b.nu, b.nφ
	i0, wu = _stencil(u, zero(u), inv(T(nu - 1)), nu)
	p = mod(ϕ, 2 * T(π)) * nφ / (2 * T(π)) # azimuth in cells; nodes at 0, 1, …, nφ - 1
	j = floor(Int, p)
	t = p - j
	wφ = (-t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2, -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6)
	Z = b.Z
	acc = zero(T)
	@inbounds for k in 1:4
		jj = mod(j + k - 2, nφ) + 1 # nodes j-1, j, j+1, j+2 (periodic)
		col = wu[1] * Z[i0, jj] + wu[2] * Z[i0 + 1, jj] + wu[3] * Z[i0 + 2, jj] + wu[4] * Z[i0 + 3, jj]
		acc += wφ[k] * col
	end
	s = clamp((1 - u) / T(0.2), zero(T), one(T))
	return acc * (s * s * (3 - 2s))
end

# Field value of a sub-image at grid radius rs_grid and azimuth ϕ: square grid (as before) or polar grid
@inline function _field_at(b, rs_grid, ϕ, offset)
	sϕ, cϕ = sincos(ϕ)
	return b(rs_grid * cϕ + offset, rs_grid * sϕ + offset)
end
@inline _field_at(b::PolarField, rs_grid, ϕ, offset) = b(rs_grid, ϕ)

# The two fields from the transformed coefficients
function _fields(metadata, Z1, Z2, sm)
	if haskey(metadata, :polar)
		return PolarField(Z1, metadata.polar.rmax), PolarField(Z2, metadata.polar.rmax)
	end
	fr = haskey(metadata, :field_rule) ? metadata.field_rule : Val(false)
	f1 = _maybe_rule(UniformLagrange(Z1, metadata.bulkgrid), fr)
	f2 = _maybe_rule(UniformLagrange(Z2, metadata.bulkgrid), fr)
	return _maybe_taper(f1, sm), _maybe_taper(f2, sm)
end
