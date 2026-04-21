def is_vowel(char):
        return char in "AEIOU"

    count = 0
    for i in range(0, len(s), 2):
        if is_vowel(s[i]):
            count += 1
    return count
