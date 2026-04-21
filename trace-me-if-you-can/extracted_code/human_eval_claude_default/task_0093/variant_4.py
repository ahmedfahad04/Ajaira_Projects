def refactor_variant_4(message):
    vowel_shift = lambda x: chr(ord(x) + 2) if x in "aeiouAEIOU" else x
    transformed_message = [c.upper() if c.islower() else c.lower() for c in message]
    return ''.join(vowel_shift(char) for char in transformed_message)
