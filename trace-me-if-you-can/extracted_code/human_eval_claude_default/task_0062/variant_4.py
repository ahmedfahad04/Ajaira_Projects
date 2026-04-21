import itertools
return list(itertools.starmap(lambda i, x: i * x, enumerate(xs)))[1:]
