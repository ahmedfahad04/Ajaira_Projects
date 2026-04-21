def refine_string(string):
    if not string:
        return ''

    suffix_start = 0
    while suffix_start < len(string) and not is_palindrome(string[suffix_start:]):
        suffix_start += 1

    return string + string[:suffix_start][::-1]
