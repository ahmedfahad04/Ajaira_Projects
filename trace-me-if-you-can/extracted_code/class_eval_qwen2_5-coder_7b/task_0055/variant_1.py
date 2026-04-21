class PalindromeFinder:
    def __init__(self, text) -> None:
        self.text = text

    def find_palindrome(self, index, offset, text):
        if (index - offset == -1 or index + offset == len(text)
                or text[index - offset] != text[index + offset]):
            return 0
        return 1 + self.find_palindrome(index, offset + 1, text)

    def get_longest_palindrome(self):
        max_palindrome_len = 0
        modified_text = ""
        result = ""

        for char in self.text[:-1]:
            modified_text += char + "|"
        modified_text += self.text[-1]

        for idx in range(len(modified_text)):
            length = self.find_palindrome(idx, 1, modified_text)

            if max_palindrome_len < length:
                max_palindrome_len = length
                start_idx = idx

        for char in modified_text[start_idx - max_palindrome_len:start_idx + max_palindrome_len + 1]:
            if char != "|":
                result += char

        return result
