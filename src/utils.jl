function moving_average(data, window_size)
    return [mean(data[i:window_size+i]) for i in 1:(length(data)-window_size)]
end

mutable struct Callback
	counter::Int
	stride::Int
    fpost
    loss_arr::Vector
	const f::Function
end

Callback(stride, fpost, loss_arr, f) = Callback(0, stride, fpost, loss_arr, f)
function (c::Callback)(state, loss, others...)
    fpost = c.fpost
    metadata = fpost.lpost.skymodel.metadata
    loss_arr = c.loss_arr
    grid = fpost.lpost.skymodel.grid
	c.counter += 1
	
	if c.counter % c.stride == 0
		append!(loss_arr, loss)
		println(
			"Iteration: $(c.counter), Loss: $(loss), m_d: $(rad2μas(Comrade.transform(fpost, state.u).sky.m_d)) μas, spin: $(Comrade.transform(fpost, state.u).sky.spin), β: $(Comrade.transform(fpost, state.u).sky.βv), θs: $(Comrade.transform(fpost, state.u).sky.θs) deg, σimg: $(Comrade.transform(fpost, state.u).sky.σimg)",
		)
		tsol = Comrade.transform(fpost, state.u)
		img=imageviz(intensitymap(ModifiedKerrGMRF(Comrade.transform(fpost, state.u).sky, metadata), grid), colorscale = log10, colorrange = (1e-6, 1e-3), colormap = :inferno)
		text!(img.axis, (77-20), (65-20); text = latexstring("M/D: $(round(tsol.sky.m_d  |> rad2μas,digits=2))\\ \\mu as"), color = :white, fontsize = 35)
		text!(img.axis, (77-20), (52-20); text = latexstring("a: $(round(tsol.sky.spin, digits=2))"), color = :white, fontsize = 35)
		text!(img.axis, (77-20), (35-20); text = latexstring("\\theta_o: $(round(tsol.sky.θo, digits=2))\\degree"), color = :white, fontsize = 35)
		display(img)
		return false
	else
		return false
	end
end
