# Vectorized, branch-free KerrGMRFFast for Reactant (XLA). Renders the same image as
#     RenormalizedFlux(KerrGMRFFast(θ, meta; pa), θ.f)        (src/models/KerrGMRFFast.jl)
# but as array code that Reactant can trace once and compile into a single XLA program in which every
# parameter (and the GMRF fields) is a traced input.
#
# Layout: the 8 ray families (cone 1/2 × n = 0/1 × isindir = true/false) are stacked along a third
# array axis, so the pixel grid (npix, npix) becomes (npix, npix, 8) and one broadcast of the scalar ray
# kernel covers every family. Per family the kernel returns
#     E  = the emission without the GMRF field (0 for failed rays),
#     gx, gy = the field-grid coordinates where the ray lands,
# and the GMRF is then interpolated with a vectorized gather (16 stencil values per ray, 4-point
# Lagrange weights identical to UniformLagrange, including its boundary stencils; zero outside the grid).
# The image is sum_families E·exp(g) / m_d², times the pixel area, renormalized to total flux f.
#
# Inputs (see `kerrgmrf_image`):
#   p   : NamedTuple of scalars (m_d [rad], spin, θo [deg], θs [deg], pa [deg], χ, ι, βv, spec, η, rpeak,
#         p1, p2, f) — pass ConcreteRNumbers so one compiled program serves all values. pa must already
#         include any 180° flip (as ModifiedKerrGMRF does).
#   Z1, Z2 : the GMRF fields on the bulk grid, σimg * transform(c, ρpr, νpr) (computed outside for now).
#   X, Y   : the image pixel centers in radians, as (npix, npix) arrays (X varies along dim 1).
#   g      : static grid description (FieldGrid), raster_size and offset (plain Julia numbers).
#
# Requires: Krang (with the KrangReactantExt overlays), Reactant, VLBISkyModels (rad2μas).

# ---------------------------------------------------------------------------------------------
# Static description of the bulk (GMRF) grid: same quantities as UniformLagrange(Z, bulkgrid)
# ---------------------------------------------------------------------------------------------

struct FieldGrid
	x0::Float64
	y0::Float64
	xend::Float64
	yend::Float64
	dx::Float64
	dy::Float64
	nx::Int
	ny::Int
	scale::Float64
end
function FieldGrid(bulkgrid)
	X, Y = bulkgrid.X, bulkgrid.Y
	nx, ny = length(X), length(Y)
	dx, dy = Float64(step(X)), Float64(step(Y))
	return FieldGrid(first(X), first(Y), last(X), last(Y), dx, dy, nx, ny, inv(dx * dy * nx^2))
end

# Family layout along axis 3: cone (1, 2), n, isindir
const FAM_CONE = reshape([1, 1, 1, 1, 2, 2, 2, 2], 1, 1, 8)
const FAM_N = reshape([0, 0, 1, 1, 0, 0, 1, 1], 1, 1, 8)
const FAM_DIR = reshape([true, false, true, false, true, false, true, false], 1, 1, 8)

# ---------------------------------------------------------------------------------------------
# Scalar ray kernel (traced elementwise by broadcast). Branch-free.
# ---------------------------------------------------------------------------------------------

# log(1 + exp(x)) without overflow, branch-free
@inline _softplus_bf(x) = max(x, zero(x)) + log1p(exp(-abs(x)))

@inline function _ray(X, Y, cone, n, isindir,
	spin, θo_deg, θs_deg, m_d, pa_deg, χ, ι, βv, spec, ηB, rpeak, p1, p2, raster_size, offset, smooth::Bool)
	T = typeof(spin)
	πT = T(π)
	met = Krang.Kerr(spin)
	θo = θo_deg * πT / 180
	θs1 = θs_deg * πT / 180
	iscone1 = cone == 1
	θs = Base.ifelse(iscone1, θs1, πT - θs1)
	sgn = Base.ifelse(iscone1, one(T), -one(T)) # cone 2 flips the in-plane magnetic field

	# Stretch by m_d and Rotate by pa, per ray, as in KerrGMRFFast.intensity_point
	s, c = sincos(pa_deg * πT / 180)
	α = -(c * X - s * Y) / m_d
	β = (s * X + c * Y) / m_d
	pix = Krang.SlowLightIntensityPixel(met, α, β, θo)
	rh = Krang.horizon(met)

	rs, ϕs, νr, νθ, ok = Krang.emission_coordinates_fast_light(pix, θs, isindir, n)

	uas = rad2μas(m_d)
	rs_grid = (rs - rh) * uas / (raster_size - rh * uas)
	valid = ok & (rh < rs) & (rs < T(Inf)) & (rs_grid >= zero(T))
	# Evaluate everything at a harmless radius for invalid rays and mask at the end
	rsafe = Base.ifelse(valid, rs, rh + 4 * one(T))
	rgsafe = Base.ifelse(valid, rs_grid, zero(T))

	# _synchrotron
	magfield = Krang.SVector(sgn * sin(ι) * cos(ηB), sgn * sin(ι) * sin(ηB), cos(ι))
	p_bl_d = Krang.p_bl_d(met, rsafe, θs, Krang.η(pix), Krang.λ(pix), νr, νθ)
	p_bl_u = Krang.metric_uu(met, rsafe, θs) * p_bl_d
	p_zamo_u = Krang.jac_zamo_u_bl_d(met, rsafe, θs) * p_bl_u
	p_fluid_u = Krang.jac_fluid_u_zamo_d(met, βv, πT / 2, χ) * p_zamo_u
	bx, by, bz = magfield
	ux, uy, uz = p_fluid_u[2] / p_fluid_u[1], p_fluid_u[3] / p_fluid_u[1], p_fluid_u[4] / p_fluid_u[1]
	vx, vy, vz = uy * bz - uz * by, uz * bx - ux * bz, ux * by - uy * bx
	nrm = √(vx * vx + vy * vy + vz * vz) + eps(T)
	redshift = inv(p_fluid_u[1])
	lp = abs(p_fluid_u[1] / p_fluid_u[4])

	# field-grid coordinates
	sϕ, cϕ = sincos(Krang.ϕ_kerr_schild(met, rsafe, ϕs))
	gx = rgsafe * cϕ + offset
	gy = rgsafe * sϕ + offset

	lr = log(rsafe / rpeak)
	lE = p1 * lr - _softplus_bf((p1 + p2) * lr) +
		(3 + spec) * log(max(redshift, eps(T))) + (1 + spec) * log(nrm)
	lpc = smooth ? lp - _softplus_bf(lp - T(1e2)) : min(lp, T(1e2))
	E = Base.ifelse(valid, exp(lE) * lpc, zero(T))
	# Rays with no emission are sent outside the field grid (g = 0 there)
	big = T(1e3)
	return E, Base.ifelse(valid, gx, big), Base.ifelse(valid, gy, big)
end

# Broadcastable kernel selecting one output; `smooth` is static (not traced).
struct RayKernel{K}
	smooth::Bool
end
(k::RayKernel{K})(args...) where {K} = _ray(args..., k.smooth)[K]
struct RayKernelAll
	smooth::Bool
end
(k::RayKernelAll)(args...) = _ray(args..., k.smooth)

_istraced(x) = x isa Reactant.AnyTracedRArray || x isa Reactant.TracedRNumber

# (E, gx, gy) over the broadcast shape of args. Under Reactant the scalar kernel is traced ONCE and
# batched with all three outputs (TracedUtils.elem_apply on inputs expanded to the full shape); a
# plain broadcast would trace and compile it once per output. On plain arrays: three broadcasts.
function _ray_fields(smooth::Bool, args)
	if any(_istraced, args)
		sz = Broadcast.broadcast_shape(map(size, args)...)
		targs = map(a -> Reactant.broadcast_to_size(a, sz), args)
		return Reactant.TracedUtils.elem_apply(RayKernelAll(smooth), targs...)
	end
	return RayKernel{1}(smooth).(args...), RayKernel{2}(smooth).(args...), RayKernel{3}(smooth).(args...)
end

# ---------------------------------------------------------------------------------------------
# Vectorized bicubic (4-point Lagrange) interpolation, matching UniformLagrange
# ---------------------------------------------------------------------------------------------

# First stencil node (1-based) and Lagrange weights; arrays in, arrays out. q must be inside the grid.
function _stencil_arrays(q, q0, h, n)
	u = (q .- q0) ./ h
	uc = clamp.(u, 0.0, Float64(n)) # keeps the Int conversion finite; in-grid values are unchanged
	i = clamp.(floor.(Int, uc) .+ 1, 1, n - 1)
	i0 = clamp.(i .- 1, 1, n - 3) # 1 in the first cell, n-3 in the last, i-1 otherwise
	t = u .- (i0 .- 1)
	tm1, tm2, tm3 = t .- 1, t .- 2, t .- 3
	w1 = .-tm1 .* tm2 .* tm3 ./ 6
	w2 = t .* tm2 .* tm3 ./ 2
	w3 = .-t .* tm1 .* tm3 ./ 2
	w4 = t .* tm1 .* tm2 ./ 6
	return i0, (w1, w2, w3, w4)
end

# Zstack: (nx, ny, 2) with the two cones' fields. gx, gy, cone: same-shaped arrays.
function _interp_field(Zstack, gx, gy, cone, fg::FieldGrid; smooth::Bool = false)
	(; x0, y0, xend, yend, dx, dy, nx, ny, scale) = fg
	inside = (gx .>= x0) .& (gx .<= xend) .& (gy .>= y0) .& (gy .<= yend)
	xs = ifelse.(inside, gx, x0)
	ys = ifelse.(inside, gy, y0)
	i0, wx = _stencil_arrays(xs, x0, dx, nx)
	j0, wy = _stencil_arrays(ys, y0, dy, ny)
	Zflat = vec(Zstack)
	base = i0 .+ (j0 .- 1) .* nx .+ (cone .- 1) .* (nx * ny) # linear index of node (i0, j0)
	acc = zero(gx)
	for dj in 0:3
		col = zero(gx)
		for di in 0:3
			col = col .+ wx[di + 1] .* Zflat[base .+ (di + dj * nx)]
		end
		acc = acc .+ wy[dj + 1] .* col
	end
	g = ifelse.(inside, acc .* scale, zero(gx))
	if smooth # TaperedField: C¹ smoothstep in ρ², 1 inside ρ = 0.4, 0 outside ρ = 0.5 (offset = 0)
		ρ² = gx .* gx .+ gy .* gy
		uu = clamp.((0.25 .- ρ²) ./ (0.25 - 0.16), 0.0, 1.0)
		g = g .* (uu .* uu .* (3 .- 2 .* uu))
	end
	return g
end

# ---------------------------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------------------------

# X, Y: (npix, npix) pixel centers in radians; returns the (npix, npix) image (sum = p.f).
# fam = (cone, n, isindir) arrays of shape (1, 1, 8); pass FAM_CONE etc (as traced or plain arrays).
# area: pixel area |ΔX ΔY| in rad² (static).
function kerrgmrf_image(p, Z1, Z2, X, Y, fam, fg::FieldGrid, raster_size, offset, area; smooth::Bool = false)
	cone, nn, dir = fam
	X3 = reshape(X, size(X)..., 1)
	Y3 = reshape(Y, size(Y)..., 1)
	args = (X3, Y3, cone, nn, dir,
		p.spin, p.θo, p.θs, p.m_d, p.pa, p.χ, p.ι, p.βv, p.spec, p.η, p.rpeak, p.p1, p.p2,
		raster_size, offset)
	E, gx, gy = _ray_fields(smooth, args)
	conefull = cone .+ zeros(Int, size(E)) # cone index at every ray
	Zstack = cat(reshape(Z1, size(Z1)..., 1), reshape(Z2, size(Z2)..., 1); dims = 3)
	g = _interp_field(Zstack, gx, gy, conefull, fg; smooth)
	img = dropdims(sum(E .* exp.(g); dims = 3); dims = 3)
	img = img .* (area / p.m_d^2)
	return img .* (p.f / sum(img))
end
