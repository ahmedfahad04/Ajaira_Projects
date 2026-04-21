def refine_string(string):
    if not string:
        return ''

    if is_palindrome(string):
        return string
    else:
        return string + string[:len(string)][::-1]

def is_palindrome(s):
    return s == s[::-1]
