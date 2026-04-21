def refactor_variant_2(message):
    import string
    vowels_lower = "aeiou"
    vowels_upper = "AEIOU"
    translation_table = str.maketrans(
        vowels_lower + vowels_upper,
        ''.join(chr(ord(v) + 2) for v in vowels_lower + vowels_upper)
    )
    return message.swapcase().translate(translation_table)
