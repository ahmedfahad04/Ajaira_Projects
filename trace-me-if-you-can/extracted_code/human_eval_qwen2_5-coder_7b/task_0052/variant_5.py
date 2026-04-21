result = True
    for item in collection:
        if item >= min_value:
            result = False
            break
    return result
