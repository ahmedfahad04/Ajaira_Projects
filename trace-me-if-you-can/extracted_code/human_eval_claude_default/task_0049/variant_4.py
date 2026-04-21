from functools import reduce

ret = reduce(lambda acc, _: (acc * 2) % p, range(n), 1)
return ret
