def rotate_string_left_by_five(s):
    return ''.join(chr((ord(c) - 5 - ord('a')) % 26 + ord('a')) for c in s)
