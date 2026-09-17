function write_to_disk(θ)
    f = open(joinpath((@__DIR__), "err.txt"), "w")
    write(f, string(θ))
    close(f)
end
Enzyme.EnzymeRules.inactive(::typeof(write_to_disk), args...) = nothing

function bulk(transformedc, σimg, bulkgrid)
    bulkimg = IntensityMap(σimg * transformedc, bulkgrid)
    return BicubicInterpolatedImage(bulkimg)#, bulkint)
end