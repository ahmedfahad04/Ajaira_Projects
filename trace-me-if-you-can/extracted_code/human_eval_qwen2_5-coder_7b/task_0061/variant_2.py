def validate_brackets(bracket_sequence):
    nesting_level = 0
    for symbol in bracket_sequence:
        if symbol == '(':
            nesting_level += 1
        elif symbol == ')':
            nesting_level -= 1
        if nesting_level < 0:
            return False
    return nesting_level == 0
