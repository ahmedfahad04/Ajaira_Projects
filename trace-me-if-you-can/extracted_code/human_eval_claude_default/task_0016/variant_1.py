unique_chars = {}
for char in string.lower():
    unique_chars[char] = True
return len(unique_chars)
