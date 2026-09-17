struct JuKeBOX{T,F} <: ComradeBase.AbstractModel
    spin::T
    θo::T
    scene::F
end

function JuKeBOX(θ::NamedTuple)
    (;
        spin,
        θo,
        θs,
        rpeak,
        p1,
        p2,
        χ,
        ι,
        βv,
        spec,
        η,
    ) = θ
    T = typeof(θo)
    magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
    vel = Krang.SVector(βv, T(π / 2), χ)

    subimgs = (0,1)

    magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
    material1 = Krang.ElectronSynchrotronPowerLawIntensity(magfield1..., vel..., spec, rpeak, p1, p2, subimgs)
    geometry1 = Krang.ConeGeometry(θs*π/180, (;frac=1.0,))
    mesh1 = Krang.Mesh(geometry1, material1)

    magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
    material2 = Krang.ElectronSynchrotronPowerLawIntensity(magfield2..., vel..., spec, rpeak, p1, p2, subimgs)
    geometry2 = Krang.ConeGeometry(π-θs*π/180, (;frac=1.0,))
    mesh2 = Krang.Mesh(geometry2, material2)

    scene = Krang.Scene((mesh1,mesh2))

    return JuKeBOX(
        spin,
        θo,
        scene
    )
end

@inline function ComradeBase.intensity_point(m::JuKeBOX{T}, p) where {T}
    (; X, Y) = p
    (;scene, θo,) = m 
    
    pix = Krang.IntensityPixel(Krang.Kerr(m.spin), -X, Y, θo*T(π/180))
    ans = Krang.render(pix, scene)
    return ans #+ one(T)
end

function (linpol::Krang.ElectronSynchrotronPowerLawIntensity{N,T})(
    pix::Krang.AbstractPixel,
    intersection;
    n=0
) where {N,T}
    (; magnetic_field, fluid_velocity, R, p1, p2, spectral_index) = linpol
    (; rs, θs, νr, νθ) = intersection

    θo = Krang.inclination(pix)
    met = Krang.metric(pix)
    α, β = Krang.screen_coordinate(pix)


    norm, redshift, lp =
        Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magnetic_field, fluid_velocity, νr, νθ)


    rat = (rs / R)
    prof = rat^p1 / (one(T) + rat^(p1 + p2)) * redshift^(T(3) + spectral_index)

    # Add a clamp to lp to help remove hot pixels
    return norm^(one(T) + spectral_index) * min(lp, T(1e2)) * prof
end

