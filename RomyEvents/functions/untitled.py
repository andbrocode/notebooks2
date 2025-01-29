def __adjust_polarity_and_scale(st0, x):

    for tr in st0:
        tr.data = tr.data * x[tr.stats.channel[-1]]

    return st0