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

struct KerrGMRF{A, S, F} <: ComradeBase.AbstractModel
    met::Krang.Kerr{A}
    θo::A
    scene::S
    metadata::F
    function KerrGMRF(θ, metadata)
        (;spin, θo, χ, ι, βv, spec, η, spec) = θ
        A = typeof(θo)
        bulkmodel = bulk(θ, metadata)

        magfield1 = Krang.SVector(sin(ι) * cos(η), sin(ι) * sin(η), cos(ι))
        vel = Krang.SVector(βv, A(π / 2), χ)

        material1 = EmissivityModel(magfield1, vel, bulkmodel, spec)

        geometry1 = Krang.ConeGeometry(π/2, )

        mesh1 = Krang.Mesh(geometry1, material1)
        scene = Krang.Scene((mesh1, ))

        new{typeof(θo), typeof(scene), typeof(metadata)}(Krang.Kerr(spin), θo, scene, metadata)
    end
end

function write_to_disk(θ)
    f = open(joinpath((@__DIR__),"err.txt"), "w")
    write(f, string(θ))
    close(f)
end
Enzyme.EnzymeRules.inactive(::typeof(write_to_disk), args...) = nothing
function bulk(θ, metadata)
    (;c, σimg, rpeak, p1, p2) = θ
    (;bulkimg, bulkflucs, bulkint) = metadata
    #(;bulkgrid, bulkint) = metadata
    rad = RadialDblPower(p1, p2-1)

    mpr  = Comrade.modify(RingTemplate(rad, AzimuthalUniform()), Stretch(rpeak))
    #bulkimg = intensitymap(mpr, bulkgrid)
    intensitymap!(bulkimg, mpr)
    #write_to_disk(θ)
    #bulkimg .= [ComradeBase.intensity_point(mpr, (X=x, Y=y)) for x in bulkimg.X, y in bulkimg.Y]
    #bulkflucs = σimg .* c.params
    f  = flux(bulkimg)
    for i in 1:length(bulkimg)
        bulkimg[i] /= f
        bulkflucs[i] = σimg * c.params[i]    
    end
    rast = apply_fluctuations(CenteredLR(), bulkimg, bulkflucs) 
    return VLBISkyModels.InterpolatedImage(rast, bulkint)
end
#TODO: Add a seperate GMRF for each cone

function Comrade.intensity_point(m::KerrGMRF{A,S,F}, p) where {A, S, F}
    (; X, Y) = p
    (;scene, θo) = m

    pix = Krang.SlowLightIntensityPixel(m.met, -X, Y, θo*A(π)/180)
    ans = render(pix,scene)
    return ans #+ eps(A)
end


