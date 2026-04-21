def organize_words(sentence):
    words = sentence.split(' ')
    organized_words = [''.join(sorted(word)) for word in words]
    return ' '.join(organized_words)

res = organize_words(s)
return res
