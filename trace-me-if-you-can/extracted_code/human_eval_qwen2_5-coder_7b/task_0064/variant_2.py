vowel_chars = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
vowel_total = sum(c in vowel_chars for c in s)
if s[-1] == 'y' or s[-1] == 'Y':
    vowel_total += 1
return vowel_total
