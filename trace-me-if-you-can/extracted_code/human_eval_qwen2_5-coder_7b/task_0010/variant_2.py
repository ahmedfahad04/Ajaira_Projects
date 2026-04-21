def refine_string(string):
    if not string:
        return ''

    for i in range(len(string)):
        if string[i:] == string[i:][::-1]:
            return string + string[:i][::-1]

    return string + string[::-1]
