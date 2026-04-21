class PalindromeUtil:
    def __init__(self, string):
        self.string = string

    def expand_around_center(self, center, diff, string):
        if (center - diff == -1 or center + diff == len(string)
                or string[center - diff] != string[center + diff]):
            return 0
        return 1 + self.expand_around_center(center, diff + 1, string)

    def discover_longest_palindrome(self):
        max_palindrome_size = 0
        transformed_string = ""
        palindrome = ""

        for char in self.string[:-1]:
            transformed_string += char + "|"
        transformed_string += self.string[-1]

        for pos in range(len(transformed_string)):
            size = self.expand_around_center(pos, 1, transformed_string)

            if max_palindrome_size < size:
                max_palindrome_size = size
                start_pos = pos

        for char in transformed_string[start_pos - max_palindrome_size:start_pos + max_palindrome_size + 1]:
            if char != "|":
                palindrome += char

        return palindrome
