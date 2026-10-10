# Milestone 3 (exploratory): reverse-mode gradients of the Reactant KerrGMRFFast image through Enzyme,
# compared with central finite differences of the CPU KerrGMRFFast (src/models/KerrGMRFFast.jl).
# Loss: L = sum(W .* img) for a fixed random weight W.
using Pkg; Pkg.activate(@__DIR__)
using Comrade, Krang, Reactant, Enzyme, Printf, StableRNGs
using LinearAlgebra
import Reactant: to_rarray, ConcreteRNumber
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(dirname(@__DIR__), "src", "models", "KerrGMRFFast.jl"))
include(joinpath(@__DIR__, "KerrGMRFReactant.jl"))

const NPIX = parse(Int, get(ENV, "NPIX", "40"))
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
meta = (; bulkgrid, transform1 = idt, transform2 = idt, raster_size = RASTER, offset = OFFSET)

rng = StableRNG(7)
θ0 = (m_d = μas2rad(3.2), spin = 0.9, θo = 30.0, θs = 60.0, pa = 288.0, χ = -0.7, ι = 1.2, βv = 0.3,
	spec = 0.2, η = 2.0, rpeak = 6.0, p1 = 4.0, p2 = 1.5, f = 0.7, σimg = 1.0, ρpr = 2.0, νpr = 1.0)
c1, c2 = randn(rng, bulkpix, bulkpix), randn(rng, bulkpix, bulkpix)
W = randn(rng, NPIX, NPIX)
const PNAMES = (:m_d, :spin, :θo, :θs, :pa, :χ, :ι, :βv, :spec, :η, :rpeak, :p1, :p2, :f)

function cpu_loss(θ, Z1, Z2)
	m = KerrGMRFFast((; θ..., σimg = 1.0, c1 = Z1, c2 = Z2), meta; pa = θ.pa)
	img = IntensityMap(zeros(NPIX, NPIX), grid)
	ComradeBase.intensitymap_analytic!(img, m)
	b = baseimage(img)
	return sum(W .* (b .* (θ.f / sum(b))))
end

Wr = to_rarray(W)
famr = (to_rarray(FAM_CONE), to_rarray(FAM_N), to_rarray(FAM_DIR))
Xr, Yr = to_rarray(Xs), to_rarray(Ys)
loss(p, Z1, Z2, X, Y, fam, Wm) = sum(Wm .* kerrgmrf_image(p, Z1, Z2, X, Y, fam, FG, RASTER, OFFSET, AREA))
grad(p, Z1, Z2, X, Y, fam, Wm) = Enzyme.gradient(Enzyme.Reverse, Const(loss), p, Z1, Z2, Const(X), Const(Y), Const(fam), Const(Wm))

pr = map(ConcreteRNumber, NamedTuple{PNAMES}(map(k -> θ0[k], PNAMES)))
Z1r, Z2r = to_rarray(c1), to_rarray(c2)
t = time()
gc = @compile grad(pr, Z1r, Z2r, Xr, Yr, famr, Wr)
@printf("gradient compile: %.1f s\n", time() - t); flush(stdout)
g = gc(pr, Z1r, Z2r, Xr, Yr, famr, Wr)
t = time(); g = gc(pr, Z1r, Z2r, Xr, Yr, famr, Wr); @printf("gradient eval: %.3f s\n", time() - t)
gp = map(x -> Float64(Reactant.to_number(x)), g[1])
gZ1 = Array(g[2])

@printf("CPU loss %.10e\n", cpu_loss(θ0, c1, c2))
for k in PNAMES
	h = 1e-6 * max(abs(θ0[k]), 1e-3)
	fd = (cpu_loss((; θ0..., k => θ0[k] + h), c1, c2) - cpu_loss((; θ0..., k => θ0[k] - h), c1, c2)) / 2h
	@printf("  d/d%-6s reactant % .6e  FD % .6e  rel %.1e\n", k, gp[k], fd, abs(gp[k] - fd) / max(abs(fd), 1e-12))
end
# a few field entries with the largest Reactant gradient
idx = partialsortperm(vec(abs.(gZ1)), 1:5; rev = true)
for I in idx
	h = 1e-6
	cp_, cm_ = copy(c1), copy(c1); cp_[I] += h; cm_[I] -= h
	fd = (cpu_loss(θ0, cp_, c2) - cpu_loss(θ0, cm_, c2)) / 2h
	@printf("  d/dZ1[%5d] reactant % .6e  FD % .6e  rel %.1e\n", I, gZ1[I], fd, abs(gZ1[I] - fd) / max(abs(fd), 1e-12))
end
@printf("NaN in gradients: params %d, Z1 %d, Z2 %d\n", count(isnan, collect(gp)), count(isnan, gZ1), count(isnan, Array(g[3])))
