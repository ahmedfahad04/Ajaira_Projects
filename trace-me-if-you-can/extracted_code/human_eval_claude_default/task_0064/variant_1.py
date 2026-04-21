def count_vowels(s):
    vowel_set = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
    count = 0
    for char in s:
        if char in vowel_set:
            count += 1
    if s and s[-1].lower() == 'y':
        count += 1
    return count
