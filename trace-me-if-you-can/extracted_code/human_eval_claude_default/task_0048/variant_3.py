def is_palindrome_helper(text, start, end):
    if start >= end:
        return True
    if text[start] != text[end]:
        return False
    return is_palindrome_helper(text, start + 1, end - 1)

return is_palindrome_helper(text, 0, len(text) - 1)
