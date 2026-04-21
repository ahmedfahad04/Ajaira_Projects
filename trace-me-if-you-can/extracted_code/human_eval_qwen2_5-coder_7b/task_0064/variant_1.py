vowels = "aeiouAEIOU"
vowel_count = sum(c in vowels for c in s)
if s[-1] in ('y', 'Y'):
    vowel_count += 1
return vowel_count
