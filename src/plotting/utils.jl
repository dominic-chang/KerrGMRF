using PairPlots
using Comrade

struct MarginMakieHist <: PairPlots.VizTypeDiag
	kwargs::Any
	MarginMakieHist(; kwargs...) = new(kwargs)
end
function PairPlots.diagplot(
	ax::CM.Makie.Axis,
	viz::MarginMakieHist,
	series::PairPlots.AbstractSeries,
	colname,
)

	cn = PairPlots.columnnames(series)
	if colname ∉ cn
		return
	end
	dat = getproperty(series.table, colname)

	bins = get(series.kwargs, :bins, 16)
	bins = get(viz.kwargs, :bins, bins)

	CM.Makie.hist!(ax, dat; series.kwargs..., viz.kwargs..., bins = bins, scale_to = 1.0)#normalization = :pdf)
	CM.Makie.ylims!(ax, low = 0)
end

function _imgviz!(
	fig, ax, img::IntensityMap{<:Real}; scale_length = fieldofview(img).X / 4, show_colorbar = true, show_scalebar = true,
	kwargs...,
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
	show_scalebar && add_scalebar!(ax, img, scale_length, color)

	num_data_prods = fig.layout.size[1]
	show_colorbar && CM.Colorbar(fig[1:num_data_prods, 3], hm; label = "Brightness (Jy/μas²)", tellheight = true)
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
	scale_length = fieldofview(img).X / 4, kwargs...,
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
		tellheight = true,
	)

	if pt
		plabel = "Signed Fractional Total Polarization sign(V)|mₜₒₜ|"
	else
		plabel = "Fractional Linear Polarization |m|"
	end
	Colorbar(
		fig[2, 1], getfield(hm, :plots)[2]; tellwidth = true, tellheight = true,
		label = plabel, vertical = false, flipaxis = false,
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
		align = (:center, :bottom), color = color,
	)
end

function _kde_empirical_percentile(values, p)
	xs = sort(Float64.(filter(isfinite, collect(values))))
	if isempty(xs)
		return NaN
	elseif length(xs) == 1
		return only(xs)
	end

	idx = 1 + (length(xs) - 1) * p
	lo = floor(Int, idx)
	hi = ceil(Int, idx)
	return xs[lo] + (idx - lo) * (xs[hi] - xs[lo])
end

function _kde_sample_std(xs)
	n = length(xs)
	n <= 1 && return 0.0
	μ = sum(xs) / n
	return sqrt(sum(abs2, xs .- μ) / (n - 1))
end

function kde_percentiles(values; probs = (0.16, 0.5, 0.84), ngrid = 2048)
	xs = Float64.(filter(isfinite, collect(values)))
	if length(xs) < 2
		return map(p -> _kde_empirical_percentile(xs, p), probs)
	end

	xmin, xmax = extrema(xs)
	if xmin == xmax
		return fill(xmin, length(probs))
	end

	σ = _kde_sample_std(xs)
	iqr = _kde_empirical_percentile(xs, 0.75) - _kde_empirical_percentile(xs, 0.25)
	scale = min(σ, iqr / 1.34)
	if !isfinite(scale) || scale <= 0
		scale = σ > 0 ? σ : (xmax - xmin)
	end

	bw = 0.9 * scale * length(xs)^(-1 / 5)
	if !isfinite(bw) || bw <= 0
		bw = (xmax - xmin) / max(length(xs) - 1, 1)
	end

	grid = collect(range(xmin - 4bw, xmax + 4bw; length = ngrid))
	density = zeros(Float64, length(grid))
	for x in xs
		@. density += exp(-0.5 * ((grid - x) / bw)^2)
	end
	density ./= length(xs) * bw * sqrt(2π)

	cdf = zeros(Float64, length(grid))
	for i in 2:length(grid)
		cdf[i] = cdf[i-1] + 0.5 * (density[i-1] + density[i]) * (grid[i] - grid[i-1])
	end
	if cdf[end] <= 0 || !isfinite(cdf[end])
		return map(p -> _kde_empirical_percentile(xs, p), probs)
	end
	cdf ./= cdf[end]

	return map(probs) do p
		idx = searchsortedfirst(cdf, p)
		if idx <= 1
			grid[1]
		elseif idx > length(grid)
			grid[end]
		else
			frac = (p - cdf[idx-1]) / (cdf[idx] - cdf[idx-1])
			grid[idx-1] + frac * (grid[idx] - grid[idx-1])
		end
	end
end

function kde_estimate(values; digits = 2)
	q16, q50, q84 = kde_percentiles(values)
	return (
		center = round(q50; digits),
		lower = round(max(q50 - q16, 0); digits),
		upper = round(max(q84 - q50, 0); digits),
	)
end

function kde_estimate_richtext(values; digits = 2, color = nothing)
	estimate = kde_estimate(values; digits)
	text = CM.Makie.rich(
		string(estimate.center),
		CM.Makie.subsup("-$(estimate.lower)", "+$(estimate.upper)"),
	)
	return isnothing(color) ? text : CM.Makie.rich(text; color)
end

function add_kde_estimate_text!(
	ax,
	distributions...;
	labels = ("Dual-Cone", "PHIBI"),
	colors = nothing,
	digits = 2,
	x = 0.03,
	y = 0.95,
	lineheight = 0.14,
	fontsize = 20.0,
	kwargs...,
)
	n = length(distributions)
	label_values = labels === nothing ? ["dist $i" for i in 1:n] : collect(labels)
	if length(label_values) != n
		throw(ArgumentError("expected $n KDE labels, got $(length(label_values))"))
	end

	color_values = if colors === nothing
		palette = CM.Makie.wong_colors()
		[palette[mod1(i, length(palette))] for i in 1:n]
	else
		collect(colors)
	end
	if length(color_values) != n
		throw(ArgumentError("expected $n KDE colors, got $(length(color_values))"))
	end

	plots = Any[]
	for i in 1:n
		text = CM.Makie.rich(
			"$(label_values[i]): ",
			kde_estimate_richtext(distributions[i]; digits);
			color = color_values[i],
		)
		push!(
			plots,
			CM.text!(
				ax,
				x,
				y - (i - 1) * lineheight;
				text,
				space = :relative,
				align = (:left, :top),
				font = :bold,
				fontsize,
				kwargs...,
			),
		)
	end
	return plots
end

function add_percentile_text!(ax, dual_cone, phibi; kwargs...)
	return add_kde_estimate_text!(ax, dual_cone, phibi; kwargs...)
end

struct MarginMakieHist <: PairPlots.VizTypeDiag
	kwargs::Any
	MarginMakieHist(; kwargs...) = new(kwargs)
end
function PairPlots.diagplot(
	ax::CM.Makie.Axis,
	viz::MarginMakieHist,
	series::PairPlots.AbstractSeries,
	colname,
)

	cn = PairPlots.columnnames(series)
	if colname ∉ cn
		return
	end
	dat = getproperty(series.table, colname)

	bins = get(series.kwargs, :bins, 16)
	bins = get(viz.kwargs, :bins, bins)

	CM.Makie.hist!(ax, dat; series.kwargs..., viz.kwargs..., bins = bins, scale_to = 1.0)#normalization = :pdf)
	CM.Makie.ylims!(ax, low = 0)
end
