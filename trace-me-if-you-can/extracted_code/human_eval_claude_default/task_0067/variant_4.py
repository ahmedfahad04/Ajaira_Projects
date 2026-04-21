from functools import reduce
tokens = s.split(' ')
digit_sum = reduce(lambda acc, x: acc + int(x) if x.isdigit() else acc, tokens, 0)
return n - digit_sum
