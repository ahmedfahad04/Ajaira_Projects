def alphabetize_sentence(sentence):
    words = sentence.split(' ')
    alphabetized_words = [''.join(sorted(w)) for w in words]
    return ' '.join(alphabetized_words)

outcome = alphabetize_sentence(s)
return outcome
