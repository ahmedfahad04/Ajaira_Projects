consonants = []
for character in text:
    if character.lower() not in {'a', 'e', 'i', 'o', 'u'}:
        consonants.append(character)
return ''.join(consonants)
