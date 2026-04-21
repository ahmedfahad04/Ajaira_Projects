def refactor_v1(s):
    words = s.split(' ')
    sorted_words = []
    for word in words:
        sorted_word = ''.join(sorted(word))
        sorted_words.append(sorted_word)
    return ' '.join(sorted_words)
