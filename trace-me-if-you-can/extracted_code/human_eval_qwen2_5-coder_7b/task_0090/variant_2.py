def find_second_largest(lst):
    unique_elements = set(lst)
    if len(unique_elements) < 2:
        return None
    sorted_elements = sorted(unique_elements)
    return sorted_elements[1]
