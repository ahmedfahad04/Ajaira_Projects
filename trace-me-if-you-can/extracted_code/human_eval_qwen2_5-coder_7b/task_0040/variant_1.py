def find_zero_sum_triplet(lst):
    triplets = [(i, j, k) for i in range(len(lst)) for j in range(i + 1, len(lst)) for k in range(j + 1, len(lst)) if lst[i] + lst[j] + lst[k] == 0]
    return len(triplets) > 0

# Example usage
l = [-1, 0, 1, 2, -1, -4]
result = find_zero_sum_triplet(l)
print(result)
