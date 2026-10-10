# Milestone 2: the vectorized KerrGMRFFast (KerrGMRFReactant.jl) must reproduce the CPU KerrGMRFFast image
# (src/models/KerrGMRFFast.jl, followed by the RenormalizedFlux normalization).
#   Step A: the vectorized code run on plain CPU arrays (checks the algorithm, no Reactant).
#   Step B: the same code compiled ONCE with Reactant (all parameters and fields traced), evaluated at
#           several parameter points.
# ENV: NPIX (default 80), SMOOTH=1 to test the smoothing mode, SKIP_CPU_VEC=1 to skip step A.
using Pkg; Pkg.activate(@__DIR__)
using Comrade, Krang, Reactant, Printf, StableRNGs
using LinearAlgebra
import Reactant: to_rarray, ConcreteRNumber
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(dirname(@__DIR__), "src", "models", "KerrGMRFFast.jl"))
include(joinpath(@__DIR__, "KerrGMRFReactant.jl"))

const NPIX = parse(Int, get(ENV, "NPIX", "80"))
const SMOOTH = get(ENV, "SMOOTH", "0") == "1"
const RASTER = 120.0
const OFFSET = 0.0
bulkpix = 70
bulkgrid = imagepixels(1.0, 1.0, bulkpix, bulkpix)
grid = imagepixels(μas2rad(120.0), μas2rad(120.0), NPIX, NPIX)
const FG = FieldGrid(bulkgrid)
const AREA = abs(step(grid.X) * step(grid.Y))
Xs = [x for x in grid.X, y in grid.Y]
Ys = [y for x in grid.X, y in grid.Y]
idt(c, ρ, ν) = c
meta = SMOOTH ? (; bulkgrid, transform1 = idt, transform2 = idt, raster_size = RASTER, offset = OFFSET, smooth = Val(true)) :
	(; bulkgrid, transform1 = idt, transform2 = idt, raster_size = RASTER, offset = OFFSET)

rng = StableRNG(42)
base = (m_d = μas2rad(4.0), spin = 0.5, θo = 163.0, θs = 80.0, pa = 108.0, χ = π / 1.5, ι = π / 4, βv = 0.6,
	spec = 1.0, η = π / 2, rpeak = 3.5, p1 = 2.0, p2 = 2.0, f = 1.0, σimg = 0.5, ρpr = 2.0, νpr = 1.0)
points = [
	base,
	(; base..., m_d = μas2rad(3.2), spin = 0.9, θo = 30.0, θs = 60.0, pa = 108.0 + 180.0, χ = -0.7, ι = 1.2, βv = 0.3,
		spec = 0.2, η = 2.0, rpeak = 6.0, p1 = 4.0, p2 = 1.5, σimg = 1.0, f = 0.7),
	(; base..., m_d = μas2rad(5.5), spin = 0.2, θo = 100.0, θs = 45.0, pa = 20.0, χ = 2.5, ι = 0.3, βv = 0.85,
		spec = 3.0, η = -1.0, rpeak = 2.0, p1 = 0.5, p2 = 7.0, σimg = 0.2, f = 1.3),
]
fields = [(randn(rng, bulkpix, bulkpix), randn(rng, bulkpix, bulkpix)) for _ in points]

function cpu_image(θ, c1, c2)
	m = KerrGMRFFast((; θ..., c1, c2), meta; pa = θ.pa)
	img = IntensityMap(zeros(NPIX, NPIX), grid)
	ComradeBase.intensitymap_analytic!(img, m)
	b = baseimage(img)
	return b .* (θ.f / sum(b))
end

params(θ) = (; m_d = θ.m_d, spin = θ.spin, θo = θ.θo, θs = θ.θs, pa = θ.pa, χ = θ.χ, ι = θ.ι, βv = θ.βv,
	spec = θ.spec, η = θ.η, rpeak = θ.rpeak, p1 = θ.p1, p2 = θ.p2, f = θ.f)

report(name, a, ref) = @printf("%-34s max|Δ|/max %.2e  max rel (pix > 1e-3 max) %.2e  sum %.6f vs %.6f  NaN %d\n",
	name, maximum(abs.(a .- ref)) / maximum(ref),
	maximum(abs.(a .- ref)[ref .> 1e-3 * maximum(ref)] ./ ref[ref .> 1e-3 * maximum(ref)]), sum(a), sum(ref), count(isnan, a))

refs = Matrix{Float64}[]
for (k, θ) in enumerate(points)
	c1, c2 = fields[k]
	local t = time(); ref = cpu_image(θ, c1, c2); tcpu = time() - t
	push!(refs, ref)
	@printf("point %d: CPU KerrGMRFFast %.3f s\n", k, tcpu)
	if get(ENV, "SKIP_CPU_VEC", "0") != "1"
		local t = time()
		v = kerrgmrf_image(params(θ), θ.σimg .* c1, θ.σimg .* c2, Xs, Ys, (FAM_CONE, FAM_N, FAM_DIR), FG, RASTER, OFFSET, AREA; smooth = SMOOTH)
		report(@sprintf("  A vectorized CPU (%.2f s)", time() - t), v, ref)
	end
	flush(stdout)
end

# ---- Step B: Reactant, compiled once ----
img_fn(p, Z1, Z2, X, Y, fam) = kerrgmrf_image(p, Z1, Z2, X, Y, fam, FG, RASTER, OFFSET, AREA; smooth = SMOOTH)
cr(θ) = map(ConcreteRNumber, params(θ))
famr = (to_rarray(FAM_CONE), to_rarray(FAM_N), to_rarray(FAM_DIR))
Xr, Yr = to_rarray(Xs), to_rarray(Ys)
θ1 = points[1]
t = time()
compiled = @compile img_fn(cr(θ1), to_rarray(θ1.σimg .* fields[1][1]), to_rarray(θ1.σimg .* fields[1][2]), Xr, Yr, famr)
@printf("Reactant compile: %.1f s\n", time() - t); flush(stdout)

worst = 0.0
for (k, θ) in enumerate(points)
	c1, c2 = fields[k]
	Z1r, Z2r = to_rarray(θ.σimg .* c1), to_rarray(θ.σimg .* c2)
	pr = cr(θ)
	out = Array(compiled(pr, Z1r, Z2r, Xr, Yr, famr)) # first call
	local t = time(); nrep = 5
	for _ in 1:nrep
		out = Array(compiled(pr, Z1r, Z2r, Xr, Yr, famr))
	end
	report(@sprintf("  B Reactant point %d (%.4f s/eval)", k, (time() - t) / nrep), out, refs[k])
	global worst = max(worst, maximum(abs.(out .- refs[k])) / maximum(refs[k]))
	flush(stdout)
end
println(worst < 1e-6 ? "PASS (worst max|Δ|/max = $worst)" : "CHECK: worst max|Δ|/max = $worst")
