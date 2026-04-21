import re
def starts_with_i(sentence):
    return sentence.strip()[:2] == 'I '

sentences = re.split(r'[.?!]\s*', S)
return len(list(filter(starts_with_i, sentences)))
