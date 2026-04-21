if not string:
    return ''

def palindrome_positions(s):
    for pos in range(len(s) + 1):
        if is_palindrome(s[pos:]):
            yield pos

beginning_of_suffix = next(palindrome_positions(string))
return string + string[:beginning_of_suffix][::-1]
