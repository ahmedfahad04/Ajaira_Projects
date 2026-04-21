vowels = "aeiouAEIOU"
num_vowels = sum(1 for char in s if char in vowels)
if s[-1] == 'y' or s[-1] == 'Y':
    num_vowels += 1
return num_vowels
