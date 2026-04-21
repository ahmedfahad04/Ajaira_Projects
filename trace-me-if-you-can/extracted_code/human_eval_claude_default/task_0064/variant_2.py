def count_vowels(s):
    import re
    vowel_pattern = r'[aeiouAEIOU]'
    vowel_matches = re.findall(vowel_pattern, s)
    base_count = len(vowel_matches)
    y_bonus = 1 if s and s.endswith(('y', 'Y')) else 0
    return base_count + y_bonus
