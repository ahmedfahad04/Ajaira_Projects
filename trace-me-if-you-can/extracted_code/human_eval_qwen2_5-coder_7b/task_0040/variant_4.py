def find_zero_sum_triplet(lst):
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            for k in range(j + 1, len(lst)):
                if lst[i] + lst[j] + lst[k] == 0:
                    return True
                if lst[i] + lst[j] + lst[k] > 0:
                    break
            if lst[i] + lst[j] > 0:
                break
    return False

# Example usage
l = [-1, 0, 1, 2, -1, -4]
result = find_zero_sum_triplet(l)
print(result)
