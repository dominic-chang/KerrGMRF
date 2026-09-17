include("overloads.jl")
include("utils.jl")
include("EmissivityModel.jl")

struct KerrGMRF{A, S, F} <: ComradeBase.AbstractModel
	met::Krang.Kerr{A}
	θo::A
	scene::S
	metadata::F
	function KerrGMRF(θ, metadata)
		(; frac, m_d, spin, θs, θo, χ, ι, βv, spec, η, spec, rpeak, p1, p2, ρpr, νpr, c1) = θ
		(; bulkgrid, transform1, transform2, raster_size, offset) = metadata
		A = typeof(θo)
		bulkint1 = transform1(c1, ρpr, νpr)
		bulkmodel1 = bulk(bulkint1, θ.σimg, bulkgrid)

		bulkint2 = transform2(c1, ρpr, νpr)
		bulkmodel2 = bulk(bulkint2, θ.σimg, bulkgrid)

		vel = Krang.SVector(βv, A(π / 2), χ)

		magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
		material1 = EmissivityModel(magfield1, vel, bulkmodel1, spec, rpeak, p1, p2, m_d, raster_size, offset)
		geometry1 = Krang.ConeGeometry(θs * π / 180, (; frac,))
		mesh1 = Krang.Mesh(geometry1, material1)

		magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
		material2 = EmissivityModel(magfield2, vel, bulkmodel2, spec, rpeak, p1, p2, m_d, raster_size, offset)
		geometry2 = Krang.ConeGeometry(π-θs * π / 180, (; frac,))
		mesh2 = Krang.Mesh(geometry2, material2)

		scene = Krang.Scene((mesh1, mesh2))

		new{typeof(θo), typeof(scene), typeof(metadata)}(Krang.Kerr(spin), θo, scene, metadata)
	end
end

function Comrade.intensity_point(m::KerrGMRF{A, S, F}, p) where {A, S, F}
	(; X, Y) = p
	(; scene, θo) = m

	pix = Krang.SlowLightIntensityPixel(m.met, -X, Y, θo * A(π) / 180)
	ans = render(pix, scene)
	return ans 
end