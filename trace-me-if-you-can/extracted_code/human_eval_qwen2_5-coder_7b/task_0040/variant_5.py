def find_zero_sum_triplet(lst):
    seen = {}
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            complement = - (lst[i] + lst[j])
            if complement in seen:
                return True
            seen[lst[j]] = True
        seen.clear()
    return False

# Example usage
l = [-1, 0, 1, 2, -1, -4]
result = find_zero_sum_triplet(l)
print(result)
