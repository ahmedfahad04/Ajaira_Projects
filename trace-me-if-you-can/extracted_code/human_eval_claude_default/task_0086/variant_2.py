def refactor_v2(s):
    return ' '.join(map(lambda word: ''.join(sorted(word)), s.split(' ')))
