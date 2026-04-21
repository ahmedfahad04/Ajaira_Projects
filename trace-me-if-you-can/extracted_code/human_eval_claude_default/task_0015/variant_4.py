from functools import reduce
return reduce(lambda acc, x: acc + (' ' if acc else '') + str(x), range(n + 1), '')
