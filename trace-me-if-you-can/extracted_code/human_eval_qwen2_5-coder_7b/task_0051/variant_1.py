vowels = set("aeiouAEIOU")
return "".join([char for char in text if char not in vowels])
