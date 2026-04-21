vowels = "aeiouAEIOU"
vowel_shift = {vowel: chr(ord(vowel) + 2) for vowel in vowels}
message_swapped = message.swapcase()
refactored_message = ''.join([vowel_shift.get(letter, letter) for letter in message_swapped])
return refactored_message
