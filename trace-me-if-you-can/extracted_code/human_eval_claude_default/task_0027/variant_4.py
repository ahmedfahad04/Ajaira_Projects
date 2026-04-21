def swap_char_case(c):
    return c.lower() if c.isupper() else (c.upper() if c.islower() else c)

return ''.join(map(swap_char_case, string))
