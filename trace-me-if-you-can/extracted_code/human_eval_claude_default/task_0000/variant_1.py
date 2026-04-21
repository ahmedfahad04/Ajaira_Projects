from itertools import combinations

for elem1, elem2 in combinations(numbers, 2):
    if abs(elem1 - elem2) < threshold:
        return True
return False
