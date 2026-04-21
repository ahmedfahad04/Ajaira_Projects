def left_rotate_alphabet_five(s):
    return ''.join(chr(((ord(char) - 5 - ord('a')) % 26) + ord('a')) for char in s)
