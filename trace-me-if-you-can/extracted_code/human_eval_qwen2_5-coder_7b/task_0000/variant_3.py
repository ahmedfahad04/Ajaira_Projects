def detect_nearby_elements(numbers_list, limit):
    for primary_index, primary_value in enumerate(numbers_list):
        for secondary_index, secondary_value in enumerate(numbers_list):
            if primary_index != secondary_index:
                var_distance = abs(primary_value - secondary_value)
                if var_distance < limit:
                    return True
    return False
