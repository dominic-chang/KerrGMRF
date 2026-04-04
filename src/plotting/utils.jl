function _imgviz!(
        fig, ax, img::IntensityMap{<:Real}; scale_length = fieldofview(img).X / 4,
        kwargs...
    )
    colorrange_default = (minimum(img), maximum(img))
    dkwargs = Dict(kwargs)
    crange = get(dkwargs, :colorrange, colorrange_default)
    delete!(dkwargs, :colorrange)
    cmap = get(dkwargs, :colormap, :inferno)
    delete!(dkwargs, :colormap)

    hm = CM.heatmap!(ax, img; colorrange = crange, colormap = cmap, dkwargs...)
    CM.rotate!(hm, -ComradeBase.posang(axisdims(img)))

    color = :white#CM.Makie.to_colormap(cmap)[end]
    add_scalebar!(ax, img, scale_length, color)

    CM.Colorbar(fig[1:num_data_prods, 3], hm; label = "Brightness (Jy/μas²)", tellheight = true)
    CM.colgap!(fig.layout, 15)

    x1, y1 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), first(img.Y))
    x2, y2 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), last(img.Y))
    x3, y3 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), last(img.Y))
    x4, y4 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), first(img.Y))

    xl = min(x1, x2, x3, x4)
    xu = max(x1, x2, x3, x4)
    yl = min(y1, y2, y3, y4)
    yu = max(y1, y2, y3, y4)

    # Flip x l and u for astronomer conventions
    CM.xlims!(ax, (xu, xl))
    CM.ylims!(ax, (yl, yu))
    CM.trim!(fig.layout)

    return CM.Makie.FigureAxisPlot(fig, ax, hm)
end

function _imgviz!(
        fig, ax, img::IntensityMap{<:StokesParams};
        scale_length = fieldofview(img).X / 4, kwargs...
    )
    colorrange_default = (0.0, maximum(stokes(img, :I)))
    dkwargs = Dict(kwargs)
    crange = get(dkwargs, :colorrange, colorrange_default)
    delete!(dkwargs, :colorrange)
    cmap = get(dkwargs, :colormap, :grayC)
    delete!(dkwargs, :colormap)
    delete!(dkwargs, :size)

    pt = get(dkwargs, :plot_total, true)

    hm = polimage!(ax, img; colorrange = crange, colormap = cmap, dkwargs...)

    color = Makie.to_colormap(cmap)[end]
    add_scalebar!(ax, img, scale_length, color)

    Colorbar(
        fig[1, 2], getfield(hm, :plots)[1]; label = "Brightness (Jy/μas²)",
        tellheight = true
    )

    if pt
        plabel = "Signed Fractional Total Polarization sign(V)|mₜₒₜ|"
    else
        plabel = "Fractional Linear Polarization |m|"
    end
    Colorbar(
        fig[2, 1], getfield(hm, :plots)[2]; tellwidth = true, tellheight = true,
        label = plabel, vertical = false, flipaxis = false
    )
    colgap!(fig.layout, 15)
    rowgap!(fig.layout, 15)
    trim!(fig.layout)

    x1, y1 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), first(img.Y))
    x2, y2 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), last(img.Y))
    x3, y3 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), last(img.Y))
    x4, y4 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), first(img.Y))

    xl = min(x1, x2, x3, x4)
    xu = max(x1, x2, x3, x4)
    yl = min(y1, y2, y3, y4)
    yu = max(y1, y2, y3, y4)

    xlims!(ax, (xu, xl))
    ylims!(ax, (yl, yu))
    return Makie.FigureAxisPlot(fig, ax, hm)
end

function add_scalebar!(ax, img, scale_length, color)
    fovx, fovy = fieldofview(img)
    x0 = (last(img.X))
    y0 = (first(img.Y))

    sl = (scale_length)
    barx = [x0 - fovx / 32, x0 - fovx / 32 - sl]
    bary = fill(y0 + fovy / 32, 2)

    CM.lines!(ax, barx, bary; color = color)
    return CM.text!(
        ax, (barx[1] + (barx[2] - barx[1]) / 2), bary[1] + fovy / 64;
        text = "$(round(Int, rad2μas(sl))) μas",
        align = (:center, :bottom), color = color
    )
end


