
struct Band <: PairPlots.AbstractSeries
	label::Union{Nothing, String, Makie.RichText, Makie.LaTeXString}
	table::Any
	bottomleft::Bool
	topright::Bool
	kwargs::Any
end
function Band(truths; fullgrid = false, bottomleft = true, topright = fullgrid, label = nothing, kwargs...)
	if !(keytype(truths) isa Symbol)
		table = NamedTuple([
			Symbol(k) => v
			for (k, v) in pairs(truths)
		])
	end
	Band(label, truths, bottomleft, topright, kwargs)
end
Band(table::NamedTuple; label = nothing, fullgrid = false, bottomleft = true, topright = fullgrid, kwargs...) = Band(label, table, bottomleft, topright, kwargs)


struct MarginBand <: PairPlots.VizTypeDiag
	kwargs::Any
	MarginBand(; kwargs...) = new(kwargs)
end

function PairPlots.diagplot(ax::Makie.Axis, viz::MarginBand, series::Band, colname)

	data = getproperty(series.table, colname)

	for (low, high) in data
		Makie.vspan!(ax, low, high;
			series.kwargs...,
			viz.kwargs...)
		Makie.ylims!(ax, low = 0)
	end
end

const bands_default_viz = (
	MarginBand()
)

function PairPlots.pairplot(
	grid::Makie.GridLayout,
	@nospecialize datapairs::Any...;
	fullgrid = false, bottomleft = true, topright = fullgrid,
	bins = Dict{Symbol, Any}(),
	kwargs...,
)
	# Default to grayscale for a single series.
	# Otherwise fall back to cycling the colours ourselves.
	# The Makie color cycle functionality isn't quite flexible enough (but almost!).

	bins = convert(Dict{Symbol, Any}, bins)

	series_i = 0
	linestyles = [:solid, :dot, :dash, :dashdot, :dashdotdot]
	function SeriesDefaults(dat)
		series_i += 1
		wc = Makie.wong_colors()
		color = wc[mod1(series_i, length(wc))]
		linestyle = linestyles[1+div(series_i-1, length(wc))] # if running out of colors
		return PairPlots.Series(dat; color, bottomleft, topright, bins, strokecolor = color#=, linestyle=#) # TODO: not all series (e.g. Scatter) support linestyle attribute
	end

	countser((data, vizlayers)::Pair) = countser(data)
	countser(series::PairPlots.Series) = 1
	countser(truths::PairPlots.Truth) = 0
    countser(bands::Band) = 0
	countser(data::Any) = 1
	len_datapairs_not_truth = sum(countser, datapairs)

	if len_datapairs_not_truth == 1
		defaults1((data, vizlayers)::Pair) = PairPlots.Series(data; bottomleft, topright, bins, color = single_series_color) => vizlayers
		defaults1(series::PairPlots.Series) = series => single_series_default_viz
		defaults1(truths::PairPlots.Truth) = truths => PairPlots.truths_default_viz
		defaults1(bands::Band) = bands => bands_default_viz
		defaults1(data::Any) = PairPlots.Series(data; bottomleft, topright, bins, color = single_series_color) => single_series_default_viz
		return PairPlots.pairplot(grid, map(defaults1, datapairs)...; kwargs...)
	elseif len_datapairs_not_truth <= 5
		defaults_upto5((data, vizlayers)::Pair) = SeriesDefaults(data) => vizlayers
		defaults_upto5(series::PairPlots.Series) = series => multi_series_default_viz
		defaults_upto5(truths::PairPlots.Truth) = truths => PairPlots.truths_default_viz
		defaults_upto5(bands::Band) = bands => bands_default_viz
		defaults_upto5(data::Any) = SeriesDefaults(data) => multi_series_default_viz
		return PairPlots.pairplot(grid, map(defaults_upto5, datapairs)...; kwargs...)
	else # More than 5 series
		defaults_morethan5((data, vizlayers)::Pair) = SeriesDefaults(data) => vizlayers
		defaults_morethan5(series::PairPlots.Series) = series => many_series_default_viz
		defaults_morethan5(truths::PairPlots.Truth) = truths => PairPlots.truths_default_viz
		defaults_morethan5(bands::Band) = bands => bands_default_viz
		defaults_morethan5(data::Any) = SeriesDefaults(data) => many_series_default_viz
		return PairPlots.pairplot(grid, map(defaults_morethan5, datapairs)...; kwargs...)
	end

end

