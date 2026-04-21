nesting_level = 0
    for character in brackets:
        if character == "<":
            nesting_level += 1
        else:
            nesting_level -= 1
        if nesting_level < 0:
            return False
    return nesting_level == 0
