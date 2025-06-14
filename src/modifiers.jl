struct BicubicInterpolatedImage{I,P} <: ComradeBase.AbstractModel
    img::I
    itp::P
    function BicubicInterpolatedImage(img::SpatialIntensityMap)
        g = axisdims(img)
        itp = BicubicInterpolator(g.X, g.Y, img, NoBoundaries())
        return new{typeof(img),typeof(itp)}(img, itp)
    end
    function BicubicInterpolatedImage(img::SpatialIntensityMap, itp::BicubicInterpolator)
        return new{typeof(img),typeof(itp)}(img, itp)
    end
end


imanalytic(::Type{<:BicubicInterpolatedImage}) = IsAnalytic()
visanalytic(::Type{<:BicubicInterpolatedImage}) = NotAnalytic()
ispolarized(::Type{<:BicubicInterpolatedImage{<:IntensityMap{T}}}) where {T<:Real} = NotPolarized()
function ispolarized(::Type{<:BicubicInterpolatedImage{<:IntensityMap{T}}}) where {T<:Comrade.StokesParams}
    return IsPolarized()
end

function ComradeBase.intensity_point(m::BicubicInterpolatedImage, p) 
    g = axisdims(m.img)
    dx, dy = pixelsizes(g)
    X, Y = g.X, g.Y
    # rm = rotmat(g)'
    # X2 = VLBISkyModels._rotatex(p.X, p.Y, rm)
    # Y2 = VLBISkyModels._rotatey(p.X, p.Y, rm)
    (X[begin] > p.X || p.X > X[end]) && return zero(eltype(m.img))
    (Y[begin] > p.Y || p.Y > Y[end]) && return zero(eltype(m.img))
    return (m.itp(p.X, p.Y))*inv(dx * dy)
end

# modest type piracy since we need to include the array since we 
# do not want to have Enzyme allocate stuff for it
@inline function BasicInterpolators.cubic(x::Real, xₚ, yₚ)
    @assert length(xₚ) == length(yₚ) == 4 "Four (4) points/coordinates are required for simple cubic interpolation/"
    #short names
    @inbounds x₁, x₂, x₃, x₄ = xₚ[1], xₚ[2], xₚ[3], xₚ[4]
    @inbounds y₁, y₂, y₃, y₄ = yₚ[1], yₚ[2], yₚ[3], yₚ[4]
    #first stage
    p₁₂ = ((x - x₂)*y₁ + (x₁ - x)*y₂)/(x₁ - x₂)
    p₂₃ = ((x - x₃)*y₂ + (x₂ - x)*y₃)/(x₂ - x₃)
    p₃₄ = ((x - x₄)*y₃ + (x₃ - x)*y₄)/(x₃ - x₄)
    #second stage
    p₁₂₃ = ((x - x₃)*p₁₂ + (x₁ - x)*p₂₃)/(x₁ - x₃)
    p₂₃₄ = ((x - x₄)*p₂₃ + (x₂ - x)*p₃₄)/(x₂ - x₄)
    #final stage
    ((x - x₄)*p₁₂₃ + (x₁ - x)*p₂₃₄)/(x₁ - x₄)
end






struct RenormalizedFlux{M, T} <: ComradeBase.AbstractModel
    model::M
    flux::T
end
ComradeBase.visanalytic(::Type{<:RenormalizedFlux{M}}) where {M} = ComradeBase.visanalytic(M)
ComradeBase.imanalytic(::Type{<:RenormalizedFlux{M}}) where {M} = ComradeBase.imanalytic(M)

function ComradeBase.flux(m::RenormalizedFlux)
    return m.flux
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap, m::RenormalizedFlux)
    ComradeBase.intensitymap_analytic!(img, m.model)
    pimg = baseimage(img)
    pimg .*= flux(m) ./sum(pimg)
    return nothing
end


function ComradeBase.intensitymap_analytic!(img::IntensityMap{T,N,D,
                                                  <:ComradeBase.AbstractRectiGrid{D,
                                                                                  <:ThreadsEx{S}}},
                                m::RenormalizedFlux) where {T,N,D,S}
    ComradeBase.intensitymap_analytic!(img, m.model)
    pimg = baseimage(img)
    pimg .*= flux(m) ./ sum(pimg)
    return nothing
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap{T,N,D,
                                                  <:ComradeBase.AbstractRectiGrid{D,
                                                                                  <:ThreadsEx{:Enzyme}}},
                                m::RenormalizedFlux) where {T,N,D}
    ComradeBase.intensitymap_analytic!(img, m.model)
    pimg = baseimage(img)
    pimg .*= flux(m) ./ sum(pimg)
    return nothing
end

function ComradeBase.intensitymap_analytic!(img::IntensityMap{T,N,D,
                                                  <:ComradeBase.AbstractRectiGrid{D,
                                                                                  <:ThreadsEx{:Polyester}}},
                                m::RenormalizedFlux) where {T,N,D}
    ComradeBase.intensitymap_analytic!(img, m.model)
    pimg = baseimage(img)
    pimg .*= flux(m) ./ sum(pimg)
    return nothing
end



