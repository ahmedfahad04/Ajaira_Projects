def check_bracket_balance(brackets):
    current_level = 0
    for bracket in brackets:
        if bracket == '(':
            current_level += 1
        else:
            current_level -= 1
        if current_level < 0:
            return False
    return current_level == 0
