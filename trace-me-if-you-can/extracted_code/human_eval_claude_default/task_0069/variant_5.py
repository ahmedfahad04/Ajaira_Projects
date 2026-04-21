from functools import reduce
from collections import Counter

def find_max_valid(acc, item):
    num, freq = item
    return max(acc, num) if freq >= num else acc

counter = Counter(lst)
return reduce(find_max_valid, counter.items(), -1)
