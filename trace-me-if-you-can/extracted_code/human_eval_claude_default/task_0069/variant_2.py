# Variant 2: Counter with sorted keys approach
from collections import Counter

counter = Counter(lst)
ans = -1

for num in sorted(counter.keys(), reverse=True):
    if counter[num] >= num:
        ans = num
        break

return ans
