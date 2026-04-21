if not numbers:
    return []

from itertools import chain, repeat
delimiters = repeat(delimeter, len(numbers) - 1)
pairs = zip(numbers[:-1], delimiters)
return list(chain.from_iterable(pairs)) + [numbers[-1]]
