from itertools import compress

mask = [isinstance(x, int) for x in values]
return list(compress(values, mask))
