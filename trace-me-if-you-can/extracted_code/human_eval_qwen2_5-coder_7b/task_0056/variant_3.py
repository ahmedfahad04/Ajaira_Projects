opening_tags = 0
    for bracket in brackets:
        if bracket == "<":
            opening_tags += 1
        else:
            opening_tags -= 1
        if opening_tags < 0:
            return False
    return opening_tags == 0
