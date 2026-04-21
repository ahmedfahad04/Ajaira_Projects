def refactor_v3(s):
    result = []
    for word in s.split(' '):
        chars = list(word)
        chars.sort()
        result.append(''.join(chars))
    return ' '.join(result)
