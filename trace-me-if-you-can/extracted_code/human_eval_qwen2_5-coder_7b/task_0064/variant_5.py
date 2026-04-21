vowels = set("aeiouAEIOU")
total_vowels = sum(1 for char in s if char in vowels)
if s[-1] in ('y', 'Y'):
    total_vowels += 1
return total_vowels
