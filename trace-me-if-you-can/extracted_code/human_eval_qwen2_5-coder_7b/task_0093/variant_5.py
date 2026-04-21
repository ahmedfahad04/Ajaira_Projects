vowels = "aeiouAEIOU"
vowel_shift = {v: chr(ord(v) + 2) for v in vowels}
message_transposed = message.swapcase()
refactored = ''.join([vowel_shift[letter] if letter in vowels else letter for letter in message_transposed])
return refactored
