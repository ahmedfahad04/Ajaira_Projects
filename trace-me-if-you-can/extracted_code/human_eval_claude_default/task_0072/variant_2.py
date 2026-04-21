# Variant 2: Recursive approach with slicing
def is_palindrome_helper(arr):
    if len(arr) <= 1:
        return True
    if arr[0] != arr[-1]:
        return False
    return is_palindrome_helper(arr[1:-1])

if sum(q) > w:
    return False
return is_palindrome_helper(q)
