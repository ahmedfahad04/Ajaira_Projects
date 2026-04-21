def calculate_parentheses_nesting_level(paren_input):
        depth = 0
        maximum_depth = 0
        for char in paren_input:
            if char == '(':
                depth += 1
                if depth > maximum_depth:
                    maximum_depth = depth
            elif char == ')':
                depth -= 1

        return maximum_depth

    paren_depths = [calculate_parentheses_nesting_level(group) for group in paren_string.split(' ') if group]
