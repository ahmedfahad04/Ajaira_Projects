def shift_char(ch):
    return chr(((ord(ch) - ord("a") - 5) % 26) + ord("a"))

return "".join(map(shift_char, s))
