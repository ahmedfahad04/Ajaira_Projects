counter = 0
    for ch in brackets:
        if ch == "<":
            counter += 1
        else:
            counter -= 1
        if counter < 0:
            return False
    return counter == 0
