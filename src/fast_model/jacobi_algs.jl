# Per-function choice of JacobiElliptic algorithm, for benchmarking (Krang calls the unprefixed entry points, which
# otherwise follow JacobiElliptic.default_algorithm(): AGM for the complete K and E, Carlson for everything else, plus
# CarlsonAlg._F_E directly). Include after `import JacobiElliptic` and before anything that calls the model, so the
# compiled ray tracing picks these methods up.
#   JE_ALGS (environment): ";"-separated key=algorithm pairs, algorithm one of Carlson, Fukushima, AGM. Keys:
#     K, E               complete K(m), E(m)
#     Einc, F            incomplete E(φ, m), F(φ, m); "Carlson|Fukushima" uses the second for |φ| > π/2
#     Pi, Piinc          Pi(n, m), Pi(n, φ, m)
#     J, Jinc            J(n, m), J(n, φ, m)
#     sn, cn, sc         Jacobi functions of (u, m)
#     FE                 the joint F and E that Krang calls as CarlsonAlg._F_E (Carlson or Fukushima)
#   e.g. JE_ALGS="K=Fukushima;E=Fukushima;Einc=Carlson;F=Carlson|Fukushima;Pi=Fukushima;Piinc=Carlson;J=Fukushima;Jinc=Carlson;sn=Fukushima;cn=Fukushima;sc=Fukushima;FE=Fukushima"
# Each key adds a method on Real arguments that is more specific than the default `f(args...)` entry point, so it only
# changes which algorithm runs (JacobiElliptic's Enzyme rules are defined for every algorithm).
const _JE = JacobiElliptic
_je_alg(s) = s == "Carlson" ? _JE.Carlson() : s == "Fukushima" ? _JE.Fukushima() : s == "AGM" ? _JE.AGM() :
    error("unknown JacobiElliptic algorithm $s")
# Default (JE_ALGS unset): the fastest table of the reverse-pass benchmark (jacobi_algs/ab_53515889.out, −4% per
# gradient; Carlson kept for the incomplete E, Π and J, where Fukushima is less accurate or NaN at n < 0).
# JE_ALGS=none: JacobiElliptic's own defaults.
const JE_ALGS_DEFAULT = "K=Fukushima;E=Fukushima;Einc=Carlson;F=Carlson|Fukushima;Pi=Fukushima;Piinc=Carlson;" *
    "J=Fukushima;Jinc=Carlson;sn=Fukushima;cn=Fukushima;sc=Fukushima;FE=Fukushima"
const _je_env = get(ENV, "JE_ALGS", JE_ALGS_DEFAULT)
const _je_spec = Dict(Pair(split(kv, "=")...) for kv in split(_je_env == "none" ? "" : _je_env, ";") if !isempty(kv))
for (key, val) in _je_spec
    parts = split(val, "|")
    a1 = _je_alg(parts[1])
    if key == "K"
        @eval _JE.K(m::Real) = _JE.K($a1, m)
    elseif key == "E"
        @eval _JE.E(m::Real) = _JE.E($a1, m)
    elseif key == "Einc"
        @eval _JE.E(φ::Real, m::Real) = _JE.E($a1, φ, m)
    elseif key == "F"
        if length(parts) == 2
            a2 = _je_alg(parts[2])
            @eval _JE.F(φ::Real, m::Real) = abs(φ) > π / 2 ? _JE.F($a2, φ, m) : _JE.F($a1, φ, m)
        else
            @eval _JE.F(φ::Real, m::Real) = _JE.F($a1, φ, m)
        end
    elseif key == "Pi"
        @eval _JE.Pi(n::Real, m::Real) = _JE.Pi($a1, n, m)
    elseif key == "Piinc"
        @eval _JE.Pi(n::Real, φ::Real, m::Real) = _JE.Pi($a1, n, φ, m)
    elseif key == "J"
        @eval _JE.J(n::Real, m::Real) = _JE.J($a1, n, m)
    elseif key == "Jinc"
        @eval _JE.J(n::Real, φ::Real, m::Real) = _JE.J($a1, n, φ, m)
    elseif key in ("sn", "cn", "sc")
        f = Symbol(key)
        @eval _JE.$f(u::Real, m::Real) = _JE.$f($a1, u, m)
    elseif key == "FE"
        # Krang calls CarlsonAlg._F_E explicitly; the Enzyme rule on CarlsonAlg._F_E stays in charge of the derivative
        # (it needs only F and E), and its primal now comes from the chosen algorithm.
        a1 isa _JE.Fukushima && @eval _JE.CarlsonAlg._F_E(φ::Float64, m::Float64) = _JE.FukushimaAlg._F_E(φ, m)
    else
        error("unknown JE_ALGS key $key")
    end
end
isempty(_je_spec) || println("JacobiElliptic algorithms: ", join(["$k=$v" for (k, v) in sort(collect(_je_spec))], ", "))
