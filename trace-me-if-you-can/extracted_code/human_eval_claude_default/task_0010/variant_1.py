if not string:
    return ''

def find_suffix_start(s, start=0, memo={}):
    if start in memo:
        return memo[start]
    
    if is_palindrome(s[start:]):
        memo[start] = start
        return start
    
    result = find_suffix_start(s, start + 1, memo)
    memo[start] = result
    return result

suffix_start = find_suffix_start(string)
return string + string[:suffix_start][::-1]
