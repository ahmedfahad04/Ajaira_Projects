vowels = "aeiouAEIOU"
count_vowels = lambda text: sum(c in vowels for c in text)
final_count = count_vowels(s)
if s[-1] in ('y', 'Y'):
    final_count += 1
return final_count
