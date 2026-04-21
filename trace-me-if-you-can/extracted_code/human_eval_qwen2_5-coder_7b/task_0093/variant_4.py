vowels = "aeiouAEIOU"
vowel_shift = dict((v, chr(ord(v) + 2)) for v in vowels)
message_upper = message.swapcase()
output = ''.join([vowel_shift.get(c, c) for c in message_upper])
return output
