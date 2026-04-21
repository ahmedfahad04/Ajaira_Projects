# Variant 5: Dictionary-based caching with string manipulation
def count_sevens_v5(n):
    valid_indices = {}
    
    # Build dictionary of valid numbers
    for idx in range(n):
        is_multiple_11 = (idx % 11 == 0)
        is_multiple_13 = (idx % 13 == 0)
        
        if is_multiple_11 or is_multiple_13:
            valid_indices[idx] = str(idx)
    
    # Concatenate all valid number strings and count 7s
    full_string = ''.join(valid_indices.values())
    return len([ch for ch in full_string if ch == '7'])
