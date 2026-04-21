def refine_string(string):
    if not string:
        return ''

    flag = False
    for i in range(len(string)):
        if is_palindrome(string[i:]):
            flag = True
            break

    if flag:
        return string
    else:
        return string + string[::-1]
