class Manacher:
    def __init__(self, input_string) -> None:
        self.input_string = input_string

    def palindromic_string(self):
        # Generator-based approach for memory efficiency
        def expand_around_center(s, left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                yield right - left + 1
                left -= 1
                right += 1

        # Create transformed string using different method
        if not self.input_string:
            return ""
        
        chars = list(self.input_string)
        new_input_string = chars[0]
        for char in chars[1:]:
            new_input_string += '|' + char

        best_palindrome = ""
        
        for center in range(len(new_input_string)):
            # Get all possible palindrome lengths from this center
            lengths = list(expand_around_center(new_input_string, center - 1, center + 1))
            
            if lengths:
                max_len = max(lengths)
                if max_len > len(best_palindrome):
                    start = center - (max_len - 1) // 2
                    end = center + (max_len - 1) // 2 + 1
                    candidate = new_input_string[start:end]
                    best_palindrome = candidate.replace('|', '')

        return best_palindrome
