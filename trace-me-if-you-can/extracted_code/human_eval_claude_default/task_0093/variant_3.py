def refactor_variant_3(message):
    def transform_char(c):
        if c in "aeiouAEIOU":
            return chr(ord(c) + 2)
        return c
    
    case_swapped = ''.join(c.upper() if c.islower() else c.lower() for c in message)
    return ''.join(map(transform_char, case_swapped))
