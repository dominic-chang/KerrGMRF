struct JuKeBOX{T} <: ComradeBase.AbstractModel
    spin::T
    θo::T
    θs::T
    rpeak::T
    p1::T
    p2::T
    χ::T
    ι::T
    βv::T
    spec::T
    η::T
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
    return JuKeBOX(
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
    )
end

@inline function ComradeBase.intensity_point(m::JuKeBOX{T}, p) where {T}
    (; X, Y) = p
    (;ι, η, χ, βv, θs, rpeak, p1, p2, spec) = m 

    #η2 = π - η
    magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
    magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
    vel = Krang.SVector(βv, T(π / 2), χ)
    material = Krang.ElectronSynchrotronPowerLawIntensity()

    # Create the Geometry
    @inline profile(r) = let R=rpeak, p1=p1, p2=p2
        return (r/R)^p1/(1+(r/R)^(p1+p2))
    end
        
    subimgs = (0,1)
    geometry1 = Krang.ConeGeometry((θs)*T(π/180), (magfield1, vel, subimgs, profile, spec))
    geometry2 = Krang.ConeGeometry((π - θs)*T(π/180), (magfield2, vel, subimgs, profile, spec))
    geometry = geometry1 ⊕ geometry2

    mesh = Krang.Mesh(geometry, material)

    pix = Krang.IntensityPixel(Krang.Kerr(m.spin), -X, Y, m.θo*T(π/180))
    ans = mesh.material(pix, (mesh.geometry))
    return ans
end
