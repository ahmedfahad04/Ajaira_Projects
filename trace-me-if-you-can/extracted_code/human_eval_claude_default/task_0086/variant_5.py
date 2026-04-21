def refactor_v5(s):
    words = s.split(' ')
    return ' '.join(''.join(sorted(word)) for word in words)
