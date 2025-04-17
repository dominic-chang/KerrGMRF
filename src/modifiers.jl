struct BicubicInterpolatedImage{I,P} <: ComradeBase.AbstractModel
    img::I
    itp::P
    function BicubicInterpolatedImage(img::SpatialIntensityMap)
        itp = BicubicInterpolator(img.X, img.Y, img, StrictBoundaries())
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
    rm = rotmat(g)'
    X2 = VLBISkyModels._rotatex(p.X, p.Y, rm)
    Y2 = VLBISkyModels._rotatey(p.X, p.Y, rm)
    (X[begin] > X2 || X2 > X[end]) && return zero(eltype(m.img))
    (Y[begin] > Y2 || Y2 > Y[end]) && return zero(eltype(m.img))
    return m.itp(p.X, p.Y) / (dx * dy)
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



