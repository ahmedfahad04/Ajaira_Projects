alphabet = "abcdefghijklmnopqrstuvwxyz"
shifted_alphabet = alphabet[21:] + alphabet[:21]  # shift by -5
translation_table = str.maketrans(alphabet, shifted_alphabet)
return s.translate(translation_table)
