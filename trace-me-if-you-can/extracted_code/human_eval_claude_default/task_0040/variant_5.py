def find_triplet(arr, start_idx, count, current_sum, target_sum):
    if count == 3:
        return current_sum == target_sum
    if start_idx >= len(arr) or count > 3:
        return False
    
    # Include current element
    if find_triplet(arr, start_idx + 1, count + 1, current_sum + arr[start_idx], target_sum):
        return True
    
    # Skip current element
    return find_triplet(arr, start_idx + 1, count, current_sum, target_sum)

return find_triplet(l, 0, 0, 0, 0)
