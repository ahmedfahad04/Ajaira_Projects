def find_zero_sum_triplet(lst):
    seen = set()
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            complement = - (lst[i] + lst[j])
            if complement in seen:
                return True
            seen.add(lst[j])
    return False

# Example usage
l = [-1, 0, 1, 2, -1, -4]
result = find_zero_sum_triplet(l)
print(result)
