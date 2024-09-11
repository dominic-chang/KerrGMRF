struct EmissivityModel <: Krang.AbstractMaterial end
function (linpol::EmissivityModel)(pix::Krang.AbstractPixel, geometry::Krang.UnionGeometry)
    return linpol(pix, geometry.geometry1) + linpol(pix, geometry.geometry2)
end
function (prof::EmissivityModel)(pix::Krang.AbstractPixel, geometry::Krang.ConeGeometry{T,A}) where {T,A}
    magfield, fluid_velocity, subimgs, profile, σ = geometry.attributes

    θs = geometry.opening_angle
    θo = Krang.inclination(pix)
    met = Krang.metric(pix)
    α, β = Krang.screen_coordinate(pix)

    observation = zero(T)

    #isindir = false
    for n in subimgs
        α, β = Krang.screen_coordinate(pix)
        νθ = cos(θs) < abs(cos(θo)) ? (θo > θs) ⊻ (n % 2 == 1) : β < 0
        rs, θs, ϕs, νr = (Krang.emission_coordinates_fast_light(pix, θs, β>0, n)[1:4])
        dim = (X=rs*sin(θs) * cos(ϕs), Y=rs*sin(θs) * sin(ϕs))

        if rs ≤ Krang.horizon(met)
            continue
        end
        norm, redshift, lp = Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magfield, fluid_velocity, νr, νθ)

        prof = profile(dim) * max(redshift, eps(T))^(T(3) + σ)
        i = prof * norm^(1 + σ) * lp 
        observation += i
    end
    return observation
end

struct KerrGMRF{A, T, F} <: ComradeBase.AbstractModel
    met::Krang.Kerr{A}
    θo::A
    θs::A
    χ::A
    ι::A
    βv::A
    spec::A
    η::A
    bulkEmissionModel::T
    metadata::F
    function KerrGMRF(θ, metadata)
        (;spin, θo, θs, χ, ι, βv, spec, η) = θ
        bulkmodel = bulk(θ, metadata)
        new{typeof(θo), typeof(bulkmodel), typeof(metadata)}(Krang.Kerr(spin), θo, θs, χ, ι, βv, spec, η, bulkmodel, metadata)
    end
end

function bulk(θ, metadata)
    (;c, σimg, rpeak, p1, p2) = θ
    (;ftot, bulkimg) = metadata
    rad = RadialDblPower(p1, p2)

    mpr  = modify(RingTemplate(rad, AzimuthalCosine(0.0, 0.0)), Stretch(rpeak))
    intensitymap!(bulkimg, mpr)
    
    bulkimg ./= flux(bulkimg)
    rast = apply_fluctuations(CenteredLR(), bulkimg, σimg.*c.params) 
    rast .*= (ftot)
    #parent(rast) .*=  ftot # Do this if Enzyme has an issue
    #return ContinuousImage((ftot).*rast, BSplinePulse{3}())
    return VLBISkyModels.InterpolatedImage(rast)
end

#TODO: Add a seperate GMRF for each cone

function Comrade.intensity_point(m::KerrGMRF{T}, p) where {T}
    (; X, Y) = p
    (;ι, η, χ, βv, spec, θo, θs) = m 

    #η2 = π - η
    magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
    magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
    vel = Krang.SVector(βv, T(π / 2), χ)

    material = EmissivityModel()

    # Create the Geometry
    @inline profile(dim) = let bulk = m.bulkEmissionModel
        return ComradeBase.intensity_point(bulk, dim)
    end
        
    subimgs = (0,1)
    geometry1 = Krang.ConeGeometry(θs*T(π/180), (magfield1, vel, subimgs, profile, spec))
    geometry2 = Krang.ConeGeometry(π-θs*T(π/180), (magfield2, vel, subimgs, profile, spec))
    geometry = geometry1 ⊕ geometry2

    mesh = Krang.Mesh(geometry, material)

    pix = Krang.IntensityPixel(m.met, -X, Y, θo*T(π/180))
    ans = mesh.material(pix, (mesh.geometry))
    return ans
end

