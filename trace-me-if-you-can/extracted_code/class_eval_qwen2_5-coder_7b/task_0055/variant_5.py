class PalindromeExaminer:
    def __init__(self, string):
        self.string = string

    def check_palindrome_segment(self, center, diff, text):
        if (center - diff == -1 or center + diff == len(text)
                or text[center - diff] != text[center + diff]):
            return 0
        return 1 + self.check_palindrome_segment(center, diff + 1, text)

    def extract_longest_palindrome(self):
        max_palindrome_size = 0
        modified_text = ""
        palindrome_result = ""

        for char in self.string[:-1]:
            modified_text += char + "|"
        modified_text += self.string[-1]

        for pos in range(len(modified_text)):
            size = self.check_palindrome_segment(pos, 1, modified_text)

            if max_palindrome_size < size:
                max_palindrome_size = size
                start_pos = pos

        for char in modified_text[start_pos - max_palindrome_size:start_pos + max_palindrome_size + 1]:
            if char != "|":
                palindrome_result += char

        return palindrome_result
