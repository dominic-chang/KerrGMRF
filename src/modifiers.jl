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
    pimg .= pimg * flux(m)/sum(pimg)
    return nothing
end


function ComradeBase.intensitymap_analytic!(img::IntensityMap{T,N,D,
                                                  <:ComradeBase.AbstractRectiGrid{D,
                                                                                  <:ThreadsEx{S}}},
                                m::RenormalizedFlux) where {T,N,D,S}
    ComradeBase.intensitymap_analytic!(img, m.model)
    pimg = baseimage(img)
    pimg .= pimg .* flux(m) ./ sum(pimg)
    return nothing
end

