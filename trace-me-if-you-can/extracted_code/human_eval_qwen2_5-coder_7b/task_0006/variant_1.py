def find_max_parentheses_depth(input_str):
        current_level = 0
        highest_level = 0
        for char in input_str:
            if char == '(':
                current_level += 1
                if current_level > highest_level:
                    highest_level = current_level
            elif char == ')':
                current_level -= 1

        return highest_level

    result = [find_max_parentheses_depth(group) for group in paren_string.split() if group]
