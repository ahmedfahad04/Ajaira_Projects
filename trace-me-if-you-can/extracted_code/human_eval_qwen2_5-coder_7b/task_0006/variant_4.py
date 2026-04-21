def determine_max_parentheses_depth(paren_group):
        level = 0
        max_level = 0
        for symbol in paren_group:
            if symbol == '(':
                level += 1
                if level > max_level:
                    max_level = level
            elif symbol == ')':
                level -= 1

        return max_level

    depths = [determine_max_parentheses_depth(segment) for segment in paren_string.split(' ') if segment]
