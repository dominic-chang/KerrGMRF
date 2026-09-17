@inline function Krang._raytrace(
	observation,
	pix::Krang.AbstractPixel,
	mesh::Krang.Mesh{<:Krang.ConeGeometry{T, A}, <:Krang.AbstractMaterial};
	res,
) where {T, A}
	geometry = mesh.geometry
	material = mesh.material
	θs = geometry.opening_angle
	subimgs = material.subimgs


	for n in subimgs
		for isindir in (true, false)
			#νθ = cos(θs) < abs(cos(θo)) ? (θo > θs) ⊻ (n % 2 == 1) : !isindir
			rs, ϕs, νr, νθ, issuccess = @inline emission_coordinates_fast_light(pix, θs, isindir, n)
			intersection = Krang.Intersection(zero(rs), rs, θs, ϕs, νr, νθ)

			if issuccess && (Krang.horizon(Krang.metric(pix)) < rs < T(Inf))
				observation += (@inline material(pix, intersection))
			end
		end
	end

	return observation
end