# Version 3: Two-pass linear scan approach
unique_vals = set(lst)
if len(unique_vals) < 2:
    return None
first_min = min(unique_vals)
second_min = min(x for x in unique_vals if x > first_min)
return second_min
