from functools import reduce
import operator

sum_value = sum(numbers)
prod_value = reduce(operator.mul, numbers, 1)
return sum_value, prod_value
