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
    magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
    vel = Krang.SVector(βv, T(π / 2), χ)

    subimgs = (0,1)

    geometry1 = Krang.ConeGeometry(θs*T(π/180))
    material1 = Krang.ElectronSynchrotronPowerLawIntensity(magfield1..., vel..., spec, rpeak, p1, p2, subimgs)
    mesh1 = Krang.Mesh(geometry1, material1)

    geometry2 = Krang.ConeGeometry(π-θs*T(π/180))
    material2 = Krang.ElectronSynchrotronPowerLawIntensity(magfield2..., vel..., spec, rpeak, p1, p2, subimgs)
    mesh2 = Krang.Mesh(geometry2, material2)

    scene = Krang.Scene((mesh1, mesh2))

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
    return ans + one(T)
end
