class PalindromeSolver:
    def __init__(self, text):
        self.text = text

    def check_palindrome_radius(self, center, radius, text):
        if (center - radius == -1 or center + radius == len(text)
                or text[center - radius] != text[center + radius]):
            return 0
        return 1 + self.check_palindrome_radius(center, radius + 1, text)

    def locate_longest_palindrome(self):
        largest_palindrome_len = 0
        formatted_text = ""
        outcome = ""

        for character in self.text[:-1]:
            formatted_text += character + "|"
        formatted_text += self.text[-1]

        for index in range(len(formatted_text)):
            length = self.check_palindrome_radius(index, 1, formatted_text)

            if largest_palindrome_len < length:
                largest_palindrome_len = length
                start_index = index

        for character in formatted_text[start_index - largest_palindrome_len:start_index + largest_palindrome_len + 1]:
            if character != "|":
                outcome += character

        return outcome
