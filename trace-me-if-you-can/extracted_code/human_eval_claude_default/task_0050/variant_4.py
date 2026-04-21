from string import ascii_lowercase

shift = -5
shifted_chars = []
for char in s:
    old_index = ascii_lowercase.index(char)
    new_index = (old_index + shift) % 26
    shifted_chars.append(ascii_lowercase[new_index])
return "".join(shifted_chars)
