def count_vowels(s):
    vowel_chars = "aeiouAEIOU"
    vowel_count = len([ch for ch in s if ch in vowel_chars])
    has_trailing_y = len(s) > 0 and s[-1] in "yY"
    return vowel_count + (1 if has_trailing_y else 0)
