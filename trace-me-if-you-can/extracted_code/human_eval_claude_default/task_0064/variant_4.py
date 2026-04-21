def count_vowels(s):
    def is_vowel(char):
        return char.lower() in 'aeiou'
    
    def ends_with_y(string):
        return string and string[-1].lower() == 'y'
    
    regular_vowels = sum(1 for c in s if is_vowel(c))
    y_vowel = 1 if ends_with_y(s) else 0
    return regular_vowels + y_vowel
