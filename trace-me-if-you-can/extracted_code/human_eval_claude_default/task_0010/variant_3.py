if not string:
    return ''

def binary_search_palindrome_start(s):
    left, right = 0, len(s)
    result = len(s)
    
    while left < right:
        mid = (left + right) // 2
        if is_palindrome(s[mid:]):
            result = mid
            right = mid
        else:
            left = mid + 1
    
    return result

beginning_of_suffix = binary_search_palindrome_start(string)
return string + string[:beginning_of_suffix][::-1]
