from functools import reduce
return reduce(lambda acc, e: acc and e < t, l, True)
