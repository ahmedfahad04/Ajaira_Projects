def refactor_variant_5(message):
    VOWEL_SET = frozenset("aeiouAEIOU")
    result_chars = []
    
    for character in message.swapcase():
        if character in VOWEL_SET:
            result_chars.append(chr(ord(character) + 2))
        else:
            result_chars.append(character)
    
    return ''.join(result_chars)
