from functools import reduce
return reduce(lambda acc, _: acc + 1, string, 0)
