import string

def caesar_decode(text, shift=5):
    decoded = []
    for char in text:
        if char in string.ascii_lowercase:
            char_index = string.ascii_lowercase.index(char)
            new_index = (char_index - shift) % 26
            decoded.append(string.ascii_lowercase[new_index])
        else:
            decoded.append(char)
    return "".join(decoded)

return caesar_decode(s)
