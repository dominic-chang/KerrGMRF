# Milestone 1: Krang.emission_coordinates_fast_light under Reactant must match the CPU version.
# Both cones of the PHIBI geometry (θs and π - θs), n = 0 and 1, both isindir branches.
using Pkg; Pkg.activate(@__DIR__)
using Krang, Reactant, Printf
import Reactant: to_rarray

met = Krang.Kerr(0.5)
θo = 163 * π / 180
αs = [α for α in range(-12, 12; length = 40), β in range(-12, 12; length = 40)]
βs = [β for α in range(-12, 12; length = 40), β in range(-12, 12; length = 40)]
αr, βr = to_rarray(αs), to_rarray(βs)

coords(met, α, β, θo, θs, isindir, n) =
	Krang.emission_coordinates_fast_light(Krang.SlowLightIntensityPixel(met, α, β, θo), θs, isindir, n)
f_r(args...) = coords(args...)[1]
f_ϕ(args...) = coords(args...)[2]

worst = 0.0
for θs in (70 * π / 180, 110 * π / 180), n in (0, 1), isindir in (true, false)
	cpu = [coords(met, α, β, θo, θs, isindir, n) for (α, β) in zip(αs, βs)]
	r_cpu = [c[1] for c in cpu]; ϕ_cpu = [c[2] for c in cpu]; ok_cpu = [Float64(c[5]) for c in cpu]
	r_re = Array(@jit f_r.(Ref(met), αr, βr, θo, θs, isindir, n))
	ϕ_re = Array(@jit f_ϕ.(Ref(met), αr, βr, θo, θs, isindir, n))
	ok_re = Float64.(r_re .> 0) # failed rays return r = 0, hits have r > horizon
	both = (ok_cpu .== 1) .& (ok_re .== 1)
	mism = count(ok_cpu .!= ok_re)
	dr = any(both) ? maximum(abs.(r_cpu[both] .- r_re[both]) ./ max.(abs.(r_cpu[both]), 1)) : 0.0
	dϕ = any(both) ? maximum(abs.(rem2pi.(ϕ_cpu[both] .- ϕ_re[both], RoundNearest))) : 0.0
	global worst = max(worst, dr, dϕ)
	@printf("θs %5.1f°  n %d  isindir %-5s  hits cpu %4d reactant %4d  success mismatches %3d  max rel Δr %.2e  max Δϕ %.2e\n",
		θs * 180 / π, n, isindir, count(ok_cpu .== 1), count(ok_re .== 1), mism, dr, dϕ)
end
println(worst < 1e-8 ? "PASS" : "CHECK: worst difference $worst")
