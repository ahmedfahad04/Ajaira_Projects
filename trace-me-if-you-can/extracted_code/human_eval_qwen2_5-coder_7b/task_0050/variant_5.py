def left_shift_letters_by_five(s):
    return ''.join(chr(((ord(char) - 5 - ord('a')) % 26) + ord('a')) for char in s)
