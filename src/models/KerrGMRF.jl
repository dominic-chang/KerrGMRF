struct EmissivityModel{N, T, B} <: Krang.AbstractMaterial
    magnetic_field::Krang.SVector{3, T}
    fluid_velocity::Krang.SVector{3, T}
    bulkmodel::B
    spectral_index::T
    subimgs::NTuple{N,Int}

    function EmissivityModel(magfield, vel, bulkmodel::B, spec::T) where {T, B}
        new{2, T, B}(magfield, vel, bulkmodel, spec, (0,1))
    end
end
Krang.isFastLight(material::EmissivityModel) = true
Krang.isAxisymmetric(material::EmissivityModel) = false

function (prof::EmissivityModel{N,T,B})(pix::Krang.AbstractPixel{T}, intersection) where {T,N,B}
    (;magnetic_field, fluid_velocity, bulkmodel, spectral_index) = prof
    (;rs, ϕs, θs, νr, νθ) = intersection

    θo = Krang.inclination(pix)
    met = Krang.metric(pix)
    α, β = Krang.screen_coordinate(pix)

    norm, redshift, lp = Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magnetic_field, fluid_velocity, νr, νθ)
    
    ϕks = Krang.ϕ_kerr_schild(met, rs, ϕs)
    dim = (X=rs*cos(ϕks), Y=rs*sin(ϕks))

    prof = ComradeBase.intensity_point(bulkmodel, dim) * max(redshift, eps(T))^(T(3) + spectral_index)
    return norm^(1 + spectral_index) * lp * prof
end

struct KerrGMRF{A, B, F, S} <: ComradeBase.AbstractModel
    met::Krang.Kerr{A}
    θo::A
    scene::S
    bulkEmissionModel::B
    metadata::F
    function KerrGMRF(θ, metadata)
        (;spin, θo, χ, ι, βv, spec, η) = θ
        A = typeof(θo)
        bulkmodel = bulk(θ, metadata)

        magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
        vel = Krang.SVector(βv, A(π / 2), χ)

        material1 = EmissivityModel(magfield1, vel, bulkmodel, spec)

        geometry1 = Krang.ConeGeometry(π/2, )

        mesh1 = Krang.Mesh(geometry1, material1)
        scene = Krang.Scene((mesh1, ))

        new{typeof(θo), typeof(bulkmodel), typeof(metadata), typeof(scene)}(Krang.Kerr(spin), θo, scene, bulkmodel, metadata)
    end
end

function bulk(θ, metadata)
    (;c, σimg, rpeak, p1, p2) = θ
    (;bulkimg) = metadata
    rad = RadialDblPower(p1, p2-1)

    mpr  = modify(RingTemplate(rad, AzimuthalUniform()), Stretch(rpeak))
    intensitymap!(bulkimg, mpr)
    
    bulkimg ./= flux(bulkimg)
    rast = apply_fluctuations(CenteredLR(), bulkimg, σimg .* c.params) 
    return VLBISkyModels.InterpolatedImage(rast)
end
#TODO: Add a seperate GMRF for each cone

function Comrade.intensity_point(m::KerrGMRF{A,B,F}, p) where {A, B, F}
    (; X, Y) = p
    (;scene, θo) = m

    pix = Krang.SlowLightIntensityPixel(m.met, -X, Y, θo*A(π)/180)
    ans = render(pix,scene)#mesh.material(pix, (mesh.geometry))
    return ans #+ eps(A)
end

