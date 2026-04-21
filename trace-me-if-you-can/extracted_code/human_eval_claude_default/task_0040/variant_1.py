from itertools import combinations

for triplet in combinations(l, 3):
    if sum(triplet) == 0:
        return True
return False
