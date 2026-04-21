def sort_words(sentence):
    words = sentence.split(' ')
    sorted_words = [''.join(sorted(word)) for word in words]
    return ' '.join(sorted_words)

result = sort_words(s)
return result
