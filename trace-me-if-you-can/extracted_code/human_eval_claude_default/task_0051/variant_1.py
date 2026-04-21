result = ""
for char in text:
    if char.lower() not in "aeiou":
        result += char
return result
