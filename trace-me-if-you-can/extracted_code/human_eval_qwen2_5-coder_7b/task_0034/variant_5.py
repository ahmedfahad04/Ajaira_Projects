from functools import reduce
   return reduce(lambda a, x: a if x in a else a + [x], l, [])
