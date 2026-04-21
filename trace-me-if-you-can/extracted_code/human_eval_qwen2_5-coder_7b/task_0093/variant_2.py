vowels = "aeiouAEIOU"
vowel_shift_map = dict((v, chr(ord(v) + 2)) for v in vowels)
transformed_message = message.swapcase()
new_message = ''.join(vowel_shift_map[v] if v in vowels else v for v in transformed_message)
return new_message
