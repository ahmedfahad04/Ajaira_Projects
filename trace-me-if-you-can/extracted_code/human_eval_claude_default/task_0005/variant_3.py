if not numbers:
    return []

from functools import reduce
return reduce(lambda acc, x: acc + [delimeter, x], 
              numbers[1:], [numbers[0]])
