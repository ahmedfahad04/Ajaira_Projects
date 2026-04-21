# Variant 1: Dictionary-based approach with reverse iteration
from collections import defaultdict

frq = defaultdict(int)
for num in lst:
    frq[num] += 1

ans = -1
for i in range(max(lst), 0, -1):
    if frq[i] >= i:
        return i

return ans
