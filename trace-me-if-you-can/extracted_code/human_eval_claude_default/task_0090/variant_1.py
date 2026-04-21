# Version 1: Using min() with filtering approach
if len(set(lst)) < 2:
    return None
unique_vals = set(lst)
min_val = min(unique_vals)
remaining = [x for x in unique_vals if x != min_val]
return min(remaining)
