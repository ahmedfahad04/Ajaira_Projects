from functools import reduce
if not arr:
    return []
def find_min_even(acc, curr):
    val, idx = curr
    if val % 2 == 0:
        if acc is None or val < acc[0]:
            return (val, idx)
    return acc
result = reduce(find_min_even, enumerate(arr), None)
return [] if result is None else [result[0], result[1]]
