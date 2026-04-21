from functools import reduce
return reduce(lambda acc, x: acc + x, range(n + 1), 0)
