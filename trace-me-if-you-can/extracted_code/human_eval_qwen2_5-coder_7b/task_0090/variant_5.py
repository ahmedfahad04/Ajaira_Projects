def extract_second_largest(lst):
    unique_values = list(dict.fromkeys(lst))
    unique_values.sort(reverse=True)
    return None if len(unique_values) < 2 else unique_values[1]
