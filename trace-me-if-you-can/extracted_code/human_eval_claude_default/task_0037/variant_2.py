# Version 2: Using itertools and functional approach
from itertools import chain, zip_longest
even_elements = sorted(l[::2])
odd_elements = l[1::2]
pairs = zip_longest(even_elements, odd_elements)
return list(chain.from_iterable(filter(lambda x: x is not None, pair) for pair in pairs))
