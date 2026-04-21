if not string:
    return ''

for i in range(len(string)):
    if is_palindrome(string[i:]):
        prefix_to_reverse = string[:i]
        return string + prefix_to_reverse[::-1]

return string
