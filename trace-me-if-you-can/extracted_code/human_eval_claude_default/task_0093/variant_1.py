def refactor_variant_1(message):
    vowel_mapping = {'a': 'c', 'e': 'g', 'i': 'k', 'o': 'q', 'u': 'w',
                     'A': 'C', 'E': 'G', 'I': 'K', 'O': 'Q', 'U': 'W'}
    swapped_message = message.swapcase()
    result = ""
    for char in swapped_message:
        result += vowel_mapping.get(char, char)
    return result
