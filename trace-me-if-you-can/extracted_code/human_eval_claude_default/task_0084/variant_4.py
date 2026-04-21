from functools import reduce
import operator

digits_sum = reduce(operator.add, map(int, str(N)), 0)
return f"{digits_sum:b}"
