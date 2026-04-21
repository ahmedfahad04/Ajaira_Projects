lowercase_string = string.lower()
unique_count = 0
processed = []
for char in lowercase_string:
    if char not in processed:
        processed.append(char)
        unique_count += 1
return unique_count
