# Variant 1: Using reduce with operator.add
from functools import reduce
import operator
return reduce(operator.add, strings, '')
