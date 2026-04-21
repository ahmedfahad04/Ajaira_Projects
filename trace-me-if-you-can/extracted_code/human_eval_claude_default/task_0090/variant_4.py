# Version 4: Using iterative approach with tracking variables
if len(set(lst)) < 2:
    return None
first_min = second_min = float('inf')
for val in set(lst):
    if val < first_min:
        second_min = first_min
        first_min = val
    elif val < second_min:
        second_min = val
return second_min
