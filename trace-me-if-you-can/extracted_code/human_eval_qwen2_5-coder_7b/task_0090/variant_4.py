def second_largest_value(lst):
    unique_values = sorted(set(lst))
    return None if len(unique_values) < 2 else unique_values[1]
