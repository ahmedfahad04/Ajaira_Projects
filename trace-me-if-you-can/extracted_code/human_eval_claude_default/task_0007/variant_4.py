matching_strings = []
for string in strings:
    if string.find(substring) != -1:
        matching_strings.append(string)
return matching_strings
