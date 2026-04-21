def extract_max_parentheses_level(text):
        current_bracket_count = 0
        peak_bracket_count = 0
        for char in text:
            if char == '(':
                current_bracket_count += 1
                if current_bracket_count > peak_bracket_count:
                    peak_bracket_count = current_bracket_count
            elif char == ')':
                current_bracket_count -= 1

        return peak_bracket_count

    output = [extract_max_parentheses_level(substring) for substring in paren_string.split() if substring]
