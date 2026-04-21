is_valid = True
    for value in sequence:
        if value >= limit:
            is_valid = False
            break
    return is_valid
