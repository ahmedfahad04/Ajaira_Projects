# Variant 5: Recursive approach
def count_vowels_at_even_positions(s, index=0):
    if index >= len(s):
        return 0
    current_count = 1 if s[index] in "AEIOU" else 0
    return current_count + count_vowels_at_even_positions(s, index + 2)

return count_vowels_at_even_positions(s)
