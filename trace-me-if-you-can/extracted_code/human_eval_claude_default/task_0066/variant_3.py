def solution(s):
    if len(s) == 0:
        return 0
    result = 0
    i = 0
    while i < len(s):
        if s[i].isupper():
            result += ord(s[i])
        i += 1
    return result
