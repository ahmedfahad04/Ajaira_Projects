if not string:
    return ''

palindrome_starts = filter(
    lambda pos: is_palindrome(string[pos:]), 
    range(len(string) + 1)
)
beginning_of_suffix = next(palindrome_starts)

return string + string[:beginning_of_suffix][::-1]
