def caesar_cipher(s):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    translation_table = str.maketrans(alphabet, alphabet[4:] + alphabet[:4])
    return s.translate(translation_table)
