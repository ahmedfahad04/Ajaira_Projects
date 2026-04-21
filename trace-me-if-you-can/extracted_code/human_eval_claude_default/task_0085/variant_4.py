from functools import reduce
return reduce(lambda acc, val: acc + val if val % 2 == 0 else acc, lst[1::2], 0)
