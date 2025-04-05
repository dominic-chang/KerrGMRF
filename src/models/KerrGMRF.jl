struct EmissivityModel{N, T, B} <: Krang.AbstractMaterial
    magnetic_field::Krang.SVector{3, T}
    fluid_velocity::Krang.SVector{3, T}
    bulkmodel::B
    spectral_index::T
    rpeak::T
    p1::T
    p2::T
    subimgs::NTuple{N,Int}
    
    function EmissivityModel(magfield, vel, bulkmodel::B, spec::T, rpeak::T, p1::T, p2::T) where {T, B}
        new{2, T, B}(magfield, vel, bulkmodel, spec, rpeak, p1, p2, (0,1))
    end
end
Krang.isFastLight(material::EmissivityModel) = true
Krang.isAxisymmetric(material::EmissivityModel) = false

function (prof::EmissivityModel{N,T,B})(pix::Krang.AbstractPixel{T}, intersection) where {T,N,B}
    (;magnetic_field, fluid_velocity, bulkmodel, spectral_index, rpeak, p1, p2) = prof
    (;rs, ϕs, θs, νr, νθ) = intersection

    θo = Krang.inclination(pix)
    met = Krang.metric(pix)
    α, β = Krang.screen_coordinate(pix)

    norm, redshift, lp = Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magnetic_field, fluid_velocity, νr, νθ)
    
    ϕks = Krang.ϕ_kerr_schild(met, rs, ϕs)
    dim = (X=rs*cos(ϕks), Y=rs*sin(ϕks))

    rat = (rs / rpeak)
    ans = rat^p1 / (one(T) + rat^(p1 + p2)) * max(redshift, eps(T))^(T(3) + spectral_index)*exp(ComradeBase.intensity_point(bulkmodel, dim))
    # Add a clamp to lp to help remove hot pixels
    return norm^(1 + spectral_index) * min(lp, T(1e2)) * ans
end

struct KerrGMRF{A, S, F} <: ComradeBase.AbstractModel
    met::Krang.Kerr{A}
    θo::A
    scene::S
    metadata::F
    function KerrGMRF(θ, metadata)
        (;spin, θo, θs, χ, ι, βv, spec, η, spec, rpeak, p1, p2) = θ
        (;bulkgrid, bulkint) = metadata
        A = typeof(θo)
        bulkmodel1 = bulk(θ.c1, θ.σimg, bulkgrid, bulkint)
        bulkmodel2 = bulk(θ.c2, θ.σimg, bulkgrid, bulkint)

        vel = Krang.SVector(βv, A(π / 2), χ)

        magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
        material1 = EmissivityModel(magfield1, vel, bulkmodel1, spec, rpeak, p1, p2)
        geometry1 = Krang.ConeGeometry(π/2, )
        mesh1 = Krang.Mesh(geometry1, material1)

        magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
        material2 = EmissivityModel(magfield2, vel, bulkmodel2, spec, rpeak, p1, p2)
        geometry2 = Krang.ConeGeometry(π-θs*π/180, )
        mesh2 = Krang.Mesh(geometry2, material2)

        scene = Krang.Scene((mesh1, mesh2))

        new{typeof(θo), typeof(scene), typeof(metadata)}(Krang.Kerr(spin), θo, scene, metadata)
    end
end

function write_to_disk(θ)
    f = open(joinpath((@__DIR__),"err.txt"), "w")
    write(f, string(θ))
    close(f)
end
Enzyme.EnzymeRules.inactive(::typeof(write_to_disk), args...) = nothing

function bulk(c, σimg, bulkgrid, bulkint)
    bulkimg = IntensityMap(σimg * c.params, bulkgrid)
    return VLBISkyModels.InterpolatedImage(bulkimg, bulkint)
end
#TODO: Add a seperate GMRF for each cone

function Comrade.intensity_point(m::KerrGMRF{A,S,F}, p) where {A, S, F}
    (; X, Y) = p
    (;scene, θo) = m

    pix = Krang.SlowLightIntensityPixel(m.met, -X, Y, θo*A(π)/180)
    ans = render(pix,scene)
    return ans #+ eps(A)
end


