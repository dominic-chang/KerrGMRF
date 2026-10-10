# Diagnostic: where do Reactant fast-light hits disagree with the CPU ones? Compares the intermediate
# quantities of emission_coordinates_fast_light (root count, Gθ Mino time and validity, emission radius
# and its success, Gϕ validity, βboundary blocking) for one combination.
using Pkg; Pkg.activate(@__DIR__)
using Krang, Reactant, Printf
import Reactant: to_rarray
const ext = Base.get_extension(Krang, :KrangReactantExt)

const N = 40
αs = [α for α in range(-12, 12; length = N), β in range(-12, 12; length = N)]
βs = [β for α in range(-12, 12; length = N), β in range(-12, 12; length = N)]
αr, βr = to_rarray(αs), to_rarray(βs)
const MET = Krang.Kerr(0.5)
const θo = 163 * π / 180
const θs = 70 * π / 180
const isindir = true
const n = 0

pixel(α, β) = Krang.SlowLightIntensityPixel(MET, α, β, θo)
b2f(b, x) = Base.ifelse(b, one(x), zero(x))

# --- Reactant-side quantities (each its own small kernel) ---
function re_numreals(α, β)
	r = Krang.roots(pixel(α, β))
	return b2f(Krang._isreal2(r[1]), α) + b2f(Krang._isreal2(r[2]), α) + b2f(Krang._isreal2(r[3]), α) + b2f(Krang._isreal2(r[4]), α)
end
re_maximag(α, β) = (r = Krang.roots(pixel(α, β)); max(abs(imag(r[1])), abs(imag(r[2])), abs(imag(r[3])), abs(imag(r[4]))))
re_τ(α, β) = ext._reactant_Gθ_minotime(pixel(α, β), θs, isindir, n)[1]
re_τok(α, β) = b2f(ext._reactant_Gθ_minotime(pixel(α, β), θs, isindir, n)[2], α)
function re_rs(α, β)
	pix = pixel(α, β)
	τ, _ = ext._reactant_Gθ_minotime(pix, θs, isindir, n)
	return Krang.emission_radius(pix, τ)[1]
end
function re_rsok(α, β)
	pix = pixel(α, β)
	τ, _ = ext._reactant_Gθ_minotime(pix, θs, isindir, n)
	return b2f(Krang.emission_radius(pix, τ)[4], α)
end
re_gϕok(α, β) = b2f(ext._reactant_Gϕ_minotime(pixel(α, β), θs, isindir, n)[2], α)
re_ok(α, β) = b2f(Krang.emission_coordinates_fast_light(pixel(α, β), θs, isindir, n)[5], α)

# --- CPU-side ---
cpu_pix = [pixel(α, β) for (α, β) in zip(αs, βs)]
cpu_numreals = [sum(Krang._isreal2.(Krang.roots(p))) for p in cpu_pix]
cpu_maximag = [maximum(abs.(imag.(Krang.roots(p)))) for p in cpu_pix]
cpu_G = [Krang.Gθ(p, θs, isindir, n) for p in cpu_pix]
cpu_τ = [g[1] for g in cpu_G]; cpu_τok = [g[6] for g in cpu_G]
cpu_er = [Krang.emission_radius(p, τ) for (p, τ) in zip(cpu_pix, cpu_τ)]
cpu_rs = [e[1] for e in cpu_er]; cpu_rsok = [e[4] for e in cpu_er]
cpu_gϕok = [Krang.Gϕ(p, θs, isindir, n)[6] for p in cpu_pix]
cpu_ok = [Krang.emission_coordinates_fast_light(p, θs, isindir, n)[5] for p in cpu_pix]

function cmpq(name, f, cpu; isbool = true)
	t = time()
	re = Array(@jit f.(αr, βr))
	if isbool
		reb = re .== 1
		mism = findall(reb .!= cpu)
		@printf("%-10s cpu true %4d  re true %4d  mismatches %4d  (%.0f s)\n", name, count(cpu), count(reb), length(mism), time() - t)
		return reb, mism
	else
		@printf("%-10s (%.0f s)\n", name, time() - t)
		return re, Int[]
	end
end

println("CPU: hits $(count(cpu_ok)), roots: ", [count(==(k), cpu_numreals) for k in 0:4], " (numreals 0..4)")
flush(stdout)
re_nr, _ = cmpq("numreals", re_numreals, nothing; isbool = false)
println("  Reactant roots: ", [count(==(k), re_nr) for k in 0:4], "  numreals mismatches: ", count(re_nr .!= cpu_numreals))
bad = findall(re_nr .!= cpu_numreals)
re_mi, _ = cmpq("maximag", re_maximag, nothing; isbool = false)
for I in bad[1:min(8, end)]
	@printf("    α %.2f β %.2f: cpu numreals %d max|imag| %.3e | re numreals %d max|imag| %.3e  roots cpu %s\n",
		αs[I], βs[I], cpu_numreals[I], cpu_maximag[I], re_nr[I], re_mi[I], string(Krang.roots(cpu_pix[I])))
end
flush(stdout)
re_τv, _ = cmpq("τ", re_τ, nothing; isbool = false)
_, m = cmpq("τok", re_τok, cpu_τok)
both = cpu_τok
@printf("  max |Δτ| where cpu τ valid: %.2e\n", maximum(abs.(re_τv[both] .- cpu_τ[both]); init = 0.0))
re_rsv, _ = cmpq("rs", re_rs, nothing; isbool = false)
re_rsb, m = cmpq("rsok", re_rsok, cpu_rsok)
for I in m[1:min(8, end)]
	@printf("    α %.2f β %.2f: numreals cpu %d re %d | τ cpu %.4f re %.4f | rs cpu %.4f ok %s | re %.4f ok %s\n",
		αs[I], βs[I], cpu_numreals[I], re_nr[I], cpu_τ[I], re_τv[I], cpu_rs[I], cpu_rsok[I], re_rsv[I], re_rsb[I])
end
flush(stdout)
_, m = cmpq("gϕok", re_gϕok, cpu_gϕok)
_, m = cmpq("ok", re_ok, cpu_ok)
for I in m[1:min(8, end)]
	@printf("    α %.2f β %.2f: cpu ok %s τok %s rsok %s gϕok %s numreals %d\n", αs[I], βs[I], cpu_ok[I], cpu_τok[I], cpu_rsok[I], cpu_gϕok[I], cpu_numreals[I])
end
println("done")
