class PalindromeAnalyzer:
    def __init__(self, input_str):
        self.input_str = input_str

    def expand_palindrome(self, center, diff, string):
        if (center - diff == -1 or center + diff == len(string)
                or string[center - diff] != string[center + diff]):
            return 0
        return 1 + self.expand_palindrome(center, diff + 1, string)

    def determine_longest_palindrome(self):
        max_palindrome_length = 0
        altered_string = ""
        final_palindrome = ""

        for char in self.input_str[:-1]:
            altered_string += char + "|"
        altered_string += self.input_str[-1]

        for idx in range(len(altered_string)):
            length = self.expand_palindrome(idx, 1, altered_string)

            if max_palindrome_length < length:
                max_palindrome_length = length
                start_idx = idx

        for char in altered_string[start_idx - max_palindrome_length:start_idx + max_palindrome_length + 1]:
            if char != "|":
                final_palindrome += char

        return final_palindrome
