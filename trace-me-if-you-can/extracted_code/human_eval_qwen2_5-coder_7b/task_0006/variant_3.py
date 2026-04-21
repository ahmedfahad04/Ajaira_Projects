def compute_max_parentheses_nesting(paren_groups):
        nesting_level = 0
        max_nesting_level = 0
        for character in paren_groups:
            if character == '(':
                nesting_level += 1
                if nesting_level > max_nesting_level:
                    max_nesting_level = nesting_level
            elif character == ')':
                nesting_level -= 1

        return max_nesting_level

    max_depths = [compute_max_parentheses_nesting(group) for group in paren_string.split(' ') if group]
