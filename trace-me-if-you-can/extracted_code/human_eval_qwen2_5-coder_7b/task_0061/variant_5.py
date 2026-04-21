def evaluate_bracket_balance(bracket_array):
    depth = 0
    for bracket in bracket_array:
        if bracket == '(':
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
