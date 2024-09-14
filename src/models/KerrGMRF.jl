struct EmissivityModel <: Krang.AbstractMaterial end
function (linpol::EmissivityModel)(pix::Krang.AbstractPixel, geometry::Krang.UnionGeometry)
    return linpol(pix, geometry.geometry1) + linpol(pix, geometry.geometry2)
end
function (prof::EmissivityModel)(pix::Krang.AbstractPixel, geometry::Krang.ConeGeometry{T,A}) where {T,A}
    magfield, fluid_velocity, subimgs, bulk, σ = geometry.attributes

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
        dim = (X=rs*cos(ϕs), Y=rs*sin(ϕs))

        if !(Krang.horizon(met) ≤ rs < T(Inf))
            continue
        end
        norm, redshift, lp = Krang.synchrotronIntensity(met, α, β, rs, θs, θo, magfield, fluid_velocity, νr, νθ)

        prof = ComradeBase.intensity_point(bulk, dim) * max(redshift, eps(T))^(T(3) + σ)
        i = prof * norm^(1 + σ) * lp 
        observation += i
    end
    return observation
end

struct KerrGMRF{A, T, F, M} <: ComradeBase.AbstractModel
    met::Krang.Kerr{A}
    θo::A
    mesh::M
    bulkEmissionModel::T
    metadata::F
    function KerrGMRF(θ, metadata)
        (;spin, θo, θs, χ, ι, βv, spec, η) = θ
        A = typeof(θo)
        bulkmodel = bulk(θ, metadata)

        magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
        magfield2 = Krang.SVector(-sin(ι) * cos(η), -sin(ι) * sin(η), cos(ι))
        vel = Krang.SVector(βv, A(π / 2), χ)

        material = EmissivityModel()

        ## Create the Geometry
        #@inline profile(dim) = let bulk = bulkmodel 
        #    return ComradeBase.intensity_point(bulk, dim)
        #end

        subimgs = (0,1)
        geometry1 = Krang.ConeGeometry(θs*A(π)/180, (magfield1, vel, subimgs, bulkmodel, spec))
        geometry2 = Krang.ConeGeometry(A(π)-θs*A(π)/180, (magfield2, vel, subimgs, bulkmodel, spec))
        geometry = geometry1 ⊕ geometry2

        mesh = Krang.Mesh(geometry, material)
        #new{typeof(θo), typeof(bulkmodel), typeof(metadata), typeof(mesh)}(Krang.Kerr(spin), θo, θs, χ, ι, βv, spec, η, bulkmodel, metadata, mesh)
        new{typeof(θo), typeof(bulkmodel), typeof(metadata), typeof(mesh)}(Krang.Kerr(spin), θo, mesh, bulkmodel, metadata)
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
    #rast .*= (ftot)
    #parent(rast) .*=  ftot # Do this if Enzyme has an issue
    #return ContinuousImage((ftot).*rast, BSplinePulse{3}())
    return VLBISkyModels.InterpolatedImage(rast)
end

#TODO: Add a seperate GMRF for each cone

function Comrade.intensity_point(m::KerrGMRF{A,T,F}, p) where {A, T, F}
    (; X, Y) = p
    (;mesh, θo) = m

    pix = Krang.SlowLightIntensityPixel(m.met, -X, Y, θo*A(π)/180)
    ans = mesh.material(pix, (mesh.geometry))
    return ans
end

