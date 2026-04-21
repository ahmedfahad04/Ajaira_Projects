vowels = set("aeiouAEIOU")
return "".join(filter(lambda char: char not in vowels, text))
