# Milestone 1 (single compile): Krang.emission_coordinates_fast_light under Reactant must match the CPU
# version. Both cones of the PHIBI geometry (θs and π - θs), n = 0 and 1, both isindir branches.
# spin, θo, θs, isindir and n are traced scalars, so one compiled program serves all 8 combinations.
using Pkg; Pkg.activate(@__DIR__)
using Krang, Reactant, Printf
import Reactant: to_rarray, ConcreteRNumber

const N = 40
αs = [α for α in range(-12, 12; length = N), β in range(-12, 12; length = N)]
βs = [β for α in range(-12, 12; length = N), β in range(-12, 12; length = N)]

coords(spin, α, β, θo, θs, isindir, n) =
	Krang.emission_coordinates_fast_light(Krang.SlowLightIntensityPixel(Krang.Kerr(spin), α, β, θo), θs, isindir, n)
f_r(args...) = coords(args...)[1]
f_ϕ(args...) = coords(args...)[2]
function f_ok(args...)
	c = coords(args...)
	return Base.ifelse(c[5], one(c[1]), zero(c[1])) # both branches have the (traced) radius type
end

# One kernel: radius, azimuth and success flag over the pixel grid.
kernel(α, β, spin, θo, θs, isindir, n) = (
	f_r.(spin, α, β, θo, θs, isindir, n),
	f_ϕ.(spin, α, β, θo, θs, isindir, n),
	f_ok.(spin, α, β, θo, θs, isindir, n),
)

αr, βr = to_rarray(αs), to_rarray(βs)
cr(x) = ConcreteRNumber(x)
spin0, θo0 = 0.5, 163 * π / 180
t0 = time()
kc = @compile kernel(αr, βr, cr(spin0), cr(θo0), cr(70 * π / 180), cr(true), cr(0))
@printf("compile time %.1f s\n", time() - t0); flush(stdout)

function compare(spin, θo, θs, n, isindir)
	cpu = [coords(spin, α, β, θo, θs, isindir, n) for (α, β) in zip(αs, βs)]
	r_cpu = [c[1] for c in cpu]; ϕ_cpu = [c[2] for c in cpu]; ok_cpu = [c[5] for c in cpu]
	t = time()
	r_re, ϕ_re, ok_re = Array.(kc(αr, βr, cr(spin), cr(θo), cr(θs), cr(isindir), cr(n)))
	trun = time() - t
	ok_re = ok_re .== 1
	both = ok_cpu .& ok_re
	mism = count(ok_cpu .!= ok_re)
	# hits on the CPU must have r above the horizon; misses are zeros in both
	dr = any(both) ? maximum(abs.(r_cpu[both] .- r_re[both])) : 0.0
	drel = any(both) ? maximum(abs.(r_cpu[both] .- r_re[both]) ./ abs.(r_cpu[both])) : 0.0
	dϕ = any(both) ? maximum(abs.(rem2pi.(ϕ_cpu[both] .- ϕ_re[both], RoundNearest))) : 0.0
	nanre = count(isnan, r_re) + count(isnan, ϕ_re)
	@printf("spin %.2f θo %5.1f° θs %5.1f° n %d isindir %-5s | hits cpu %4d re %4d mism %3d | max|Δr| %.2e rel %.2e | max|Δϕ| %.2e | NaN(re) %d | run %.3f s\n",
		spin, θo * 180 / π, θs * 180 / π, n, isindir, count(ok_cpu), count(ok_re), mism, dr, drel, dϕ, nanre, trun)
	flush(stdout)
	if mism > 0
		idx = findall(ok_cpu .!= ok_re)[1:min(5, mism)]
		for I in idx
			@printf("    mismatch α %.3f β %.3f: cpu ok %s r %.4f ϕ %.4f | re ok %s r %.4f ϕ %.4f\n",
				αs[I], βs[I], ok_cpu[I], r_cpu[I], ϕ_cpu[I], ok_re[I], r_re[I], ϕ_re[I])
		end
	end
	return max(drel, dϕ), mism
end

worst, totmism = 0.0, 0
for (spin, θo) in ((0.5, 163 * π / 180), (0.9, 30 * π / 180), (0.2, 100 * π / 180)),
		θsd in (70.0, 110.0), n in (0, 1), isindir in (true, false)
	d, m = compare(spin, θo, θsd * π / 180, n, isindir)
	global worst = max(worst, d); global totmism += m
end
println(worst < 1e-8 && totmism == 0 ? "PASS" : "CHECK: worst difference $worst, hit/miss mismatches $totmism")
