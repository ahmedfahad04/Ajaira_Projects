import operator
return list(map(operator.add, l, [1] * len(l)))
