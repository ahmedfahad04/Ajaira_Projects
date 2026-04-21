def process_sentence(sentence):
    words = sentence.split(' ')
    processed_words = [''.join(sorted(w)) for w in words]
    return ' '.join(processed_words)

output = process_sentence(s)
return output
