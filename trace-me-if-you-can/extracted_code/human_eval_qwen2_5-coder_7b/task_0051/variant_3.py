result = ""
vowels = set("aeiouAEIOU")
for char in text:
    if char not in vowels:
        result += char
return result
