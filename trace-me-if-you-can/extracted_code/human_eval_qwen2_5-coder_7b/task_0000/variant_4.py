def contains_close_elements(values, max_distance):
    for i, current_value in enumerate(values):
        for j, next_value in enumerate(values):
            if i != j:
                separation = abs(current_value - next_value)
                if separation < max_distance:
                    return True
    return False
