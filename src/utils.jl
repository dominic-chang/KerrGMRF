function moving_average(data, window_size)
    return [mean(data[i:window_size+i]) for i in 1:(length(data)-window_size)]
end