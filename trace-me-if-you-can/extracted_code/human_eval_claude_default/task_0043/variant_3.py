from itertools import combinations
return any(a + b == 0 for a, b in combinations(l, 2))
