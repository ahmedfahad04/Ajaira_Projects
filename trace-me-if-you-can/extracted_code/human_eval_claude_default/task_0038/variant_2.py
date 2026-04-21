# Variant 2: Using functools.reduce for function composition
from functools import reduce
return reduce(lambda x, f: f(x), [encode_cyclic, encode_cyclic], s)
