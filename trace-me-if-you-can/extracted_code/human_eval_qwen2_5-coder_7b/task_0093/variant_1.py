vowels = "aeiouAEIOU"
vowels_shifted = {vowel: chr(ord(vowel) + 2) for vowel in vowels}
message_lower = message.swapcase()
result = ''.join([vowels_shifted[letter] if letter in vowels else letter for letter in message_lower])
return result
