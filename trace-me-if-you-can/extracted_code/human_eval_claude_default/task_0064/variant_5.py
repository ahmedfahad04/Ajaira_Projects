def count_vowels(s):
    vowel_lookup = {ch: True for ch in "aeiouAEIOU"}
    total = 0
    
    for character in s:
        if vowel_lookup.get(character, False):
            total += 1
    
    if s:
        last_char = s[-1]
        if last_char == 'y' or last_char == 'Y':
            total += 1
    
    return total
