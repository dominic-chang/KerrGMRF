struct EmissivityModel{N,T,B} <: Krang.AbstractMaterial
    magnetic_field::Krang.SVector{3,T}
    fluid_velocity::Krang.SVector{3,T}
    bulkmodel::B
    spectral_index::T
    rpeak::T
    p1::T
    p2::T
    m_d::T
    raster_size::T
    subimgs::NTuple{N,Int}

    function EmissivityModel(magfield, vel, bulkmodel::B, spec::T, rpeak::T, p1::T, p2::T, m_d::T, raster_size::T) where {T,B}
        new{2,T,B}(magfield, vel, bulkmodel, spec, rpeak, p1, p2, m_d, raster_size, (0, 1))
    end
end
Krang.isFastLight(material::EmissivityModel) = true
Krang.isAxisymmetric(material::EmissivityModel) = false

@inline function (prof::EmissivityModel{N,T,B})(pix::Krang.AbstractPixel{T}, intersection) where {T,N,B}
    (; m_d, magnetic_field, fluid_velocity, bulkmodel, spectral_index, rpeak, p1, p2, raster_size) = prof
    (; rs, ϕs, θs, νr, νθ) = intersection

    θo = Krang.inclination(pix)
    met = Krang.metric(pix)
    α, β = @inline Krang.screen_coordinate(pix)

    norm, redshift, lp = @inline Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magnetic_field, fluid_velocity, νr, νθ)

    rs_h = rs
    # grid has maximum radius of 30 units
    rs_grid = (rs) / (raster_size) # convert to microarcseconds
    if rs_grid < 0
        return zero(T)
    end

    ϕks = Krang.ϕ_kerr_schild(met, rs, ϕs)
    dim = (X=rs_grid*cos(ϕks), Y=rs_grid*sin(ϕks))
    rat = (rs_h / rpeak)

    bulkpix = bulkmodel.img.X.len
    cp = ComradeBase.intensity_point(bulkmodel, dim) / (bulkpix^2) # the pixel area is 1/(bulkpix^2)
    ans = rat^p1 / (one(T) + rat^(p1 + p2)) * max(redshift, eps(T))^(T(3) + spectral_index) * exp(cp)
    ans = norm^(1 + spectral_index) * min(lp, 1e1) * ans
    # Add a clamp to lp to help remove hot pixels
    return ans
end

struct KerrGMRF{A,S,F} <: ComradeBase.AbstractModel
    met::Krang.Kerr{A}
    θo::A
    scene::S
    metadata::F
    function KerrGMRF(θ, metadata)
        (; frac, m_d, spin, θs, θo, χ, ι, βv, spec, η, spec, rpeak, p1, p2, ρpr, νpr, c1) = θ
        (; bulkgrid, transform1, raster_size) = metadata
        A = typeof(θo)
        bulkint1 = transform1(c1, ρpr, νpr)
        bulkmodel1 = bulk(bulkint1, θ.σimg, bulkgrid)

        vel = Krang.SVector(βv, A(π / 2), χ)

        magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
        material1 = EmissivityModel(magfield1, vel, bulkmodel1, spec, rpeak, p1, p2, m_d, raster_size)
        geometry1 = Krang.ConeGeometry(θs * π / 180, (; frac,))
        mesh1 = Krang.Mesh(geometry1, material1)

        scene = Krang.Scene((mesh1, ))#mesh2))

        new{typeof(θo),typeof(scene),typeof(metadata)}(Krang.Kerr(spin), θo, scene, metadata)
    end
end

function write_to_disk(θ)
    f = open(joinpath((@__DIR__), "err.txt"), "w")
    write(f, string(θ))
    close(f)
end
Enzyme.EnzymeRules.inactive(::typeof(write_to_disk), args...) = nothing

function bulk(transformedc, σimg, bulkgrid)
    bulkimg = IntensityMap(σimg * transformedc, bulkgrid)
    return BicubicInterpolatedImage(bulkimg)#, bulkint)
end
#TODO: Add a seperate GMRF for each cone

function Comrade.intensity_point(m::KerrGMRF{A,S,F}, p) where {A,S,F}
    (; X, Y) = p
    (; scene, θo) = m

    pix = Krang.SlowLightIntensityPixel(m.met, -X, Y, θo * A(π) / 180)
    ans = render(pix, scene)
    return ans #+ eps(A) 
end


@inline function Krang._raytrace(
    observation,
    pix::Krang.AbstractPixel,
    mesh::Krang.Mesh{<:Krang.ConeGeometry{T,A},<:Krang.AbstractMaterial};
    res,
) where {T,A}
    geometry = mesh.geometry
    frac = geometry.attributes.frac
    material = mesh.material
    θs = geometry.opening_angle
    subimgs = material.subimgs

    isindir = false
    for _ = 1:2 # Looping over isindir this way is needed to get Metal to work
        isindir ⊻= true
        for n in subimgs
            #νθ = cos(θs) < abs(cos(θo)) ? (θo > θs) ⊻ (n % 2 == 1) : !isindir
            rs, ϕs, νr, νθ, issuccess = @inline emission_coordinates_fast_light(pix, θs, isindir, n)
            intersection = Krang.Intersection(zero(rs), rs, θs, ϕs, νr, νθ)

            if issuccess && (Krang.horizon(Krang.metric(pix)) < rs < T(Inf))
                observation += (@inline material(pix, intersection)) * (frac ^n)
            end
        end
    end

    return observation
end