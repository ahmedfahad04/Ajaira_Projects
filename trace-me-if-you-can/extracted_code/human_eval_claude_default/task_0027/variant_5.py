swapped_chars = []
for i in range(len(string)):
    char = string[i]
    if char.isupper():
        swapped_chars.append(char.lower())
    elif char.islower():
        swapped_chars.append(char.upper())
    else:
        swapped_chars.append(char)
return ''.join(swapped_chars)
