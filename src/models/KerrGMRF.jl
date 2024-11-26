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
    #(;rs, θs, νr, νθ) = intersection

    θo = Krang.inclination(pix)
    met = Krang.metric(pix)
    α, β = Krang.screen_coordinate(pix)

    #isindir = false
    norm, redshift, lp = Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magnetic_field, fluid_velocity, νr, νθ)
    dim = (X=rs*cos(ϕs), Y=rs*sin(ϕs))

    prof = ComradeBase.intensity_point(bulkmodel, dim) * max(redshift, eps(T))^(T(3) + spectral_index)
    #rat = (rs/T(4.7))
    #prof = rat^0.5/(1+rat^(4.2)) * max(redshift, eps(T))^(T(3) + spectral_index)
    return norm^(1 + spectral_index) * lp * prof
end

struct KerrGMRF{A, B, F, S} <: ComradeBase.AbstractModel
    met::Krang.Kerr{A}
    θo::A
    scene::S
    bulkEmissionModel::B
    metadata::F
    function KerrGMRF(θ, metadata)
        (;spin, θo, θs, χ, ι, βv, spec, η) = θ
        A = typeof(θo)
        bulkmodel = bulk(θ, metadata)

        magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
        magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
        vel = Krang.SVector(βv, A(π / 2), χ)

        material1 = EmissivityModel(magfield1, vel, bulkmodel, spec)
        material2 = EmissivityModel(magfield2, vel, bulkmodel, spec)

        geometry1 = Krang.ConeGeometry(θs*A(π)/180, )
        geometry2 = Krang.ConeGeometry(A(π)-θs*A(π)/180)

        mesh1 = Krang.Mesh(geometry1, material1)
        mesh2 = Krang.Mesh(geometry2, material2)
        scene = Krang.Scene((mesh1, mesh2))

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
    return ans + one(A)
end

