# Diagnostic: why does Krang.emission_radius fail (rs = Inf) for case-1/2 rays under Reactant when the
# metric spin is a plain constant (it works when spin is traced)?
using Pkg; Pkg.activate(@__DIR__)
using Krang, Reactant, Printf
import Reactant: to_rarray, ConcreteRNumber
const ext = Base.get_extension(Krang, :KrangReactantExt)
const N = 40
αs = [α for α in range(-12, 12; length = N), β in range(-12, 12; length = N)]
βs = [β for α in range(-12, 12; length = N), β in range(-12, 12; length = N)]
αr, βr = to_rarray(αs), to_rarray(βs)
const θo = 163 * π / 180
const θs = 70 * π / 180
b2f(b, x) = Base.ifelse(b, one(x), zero(x))
pixel(spin, α, β) = Krang.SlowLightIntensityPixel(Krang.Kerr(spin), α, β, θo)
τof(pix) = ext._reactant_Gθ_minotime(pix, θs, true, 0)[1]
k1(α, β) = (p = pixel(0.5, α, β); b2f(Krang._rs_case1_and_2(p, Krang.horizon(Krang.Kerr(0.5)), τof(p))[3], α))   # constant rh
k2(spin, α, β) = (p = pixel(spin, α, β); b2f(Krang._rs_case1_and_2(p, Krang.horizon(Krang.Kerr(spin)), τof(p))[3], α)) # traced rh
k3(α, β) = (p = pixel(0.5, α, β); b2f(Krang.emission_radius(p, τof(p))[4], α))                 # constant spin
k4(spin, α, β) = (p = pixel(spin, α, β); b2f(Krang.emission_radius(p, τof(p))[4], α))        # traced spin
cpu = [(p = pixel(0.5, α, β); Krang._rs_case1_and_2(p, Krang.horizon(Krang.Kerr(0.5)), Krang.Gθ(p, θs, true, 0)[1])[3]) for (α, β) in zip(αs, βs)]
nr = [sum(Krang._isreal2.(Krang.roots(pixel(0.5, α, β)))) for (α, β) in zip(αs, βs)]
c4 = nr .== 4
@printf("CPU case1/2 success on numreals=4 rays: %d of %d\n", count(cpu[c4]), count(c4)); flush(stdout)
s = ConcreteRNumber(0.5)
for (name, f) in (("k1 _rs_case1_and_2, const rh", () -> @jit(k1.(αr, βr))),
		("k2 _rs_case1_and_2, traced rh", () -> @jit(k2.(s, αr, βr))),
		("k3 emission_radius, const spin", () -> @jit(k3.(αr, βr))),
		("k4 emission_radius, traced spin", () -> @jit(k4.(s, αr, βr))))
	ok = Array(f()) .== 1
	@printf("%-34s success on numreals=4 rays: %d of %d\n", name, count(ok[c4]), count(c4)); flush(stdout)
end
