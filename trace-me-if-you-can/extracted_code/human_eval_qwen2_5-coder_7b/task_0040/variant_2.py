from itertools import combinations

def find_zero_sum_triplet(lst):
    for triplet in combinations(lst, 3):
        if sum(triplet) == 0:
            return True
    return False

# Example usage
l = [-1, 0, 1, 2, -1, -4]
result = find_zero_sum_triplet(l)
print(result)
