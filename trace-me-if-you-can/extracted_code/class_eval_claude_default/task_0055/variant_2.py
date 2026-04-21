class Manacher:
    def __init__(self, input_string) -> None:
        self.input_string = input_string
        self.memo = {}

    def palindromic_length(self, center, diff, string):
        key = (center, diff, id(string))
        if key in self.memo:
            return self.memo[key]
        
        if (center - diff == -1 or center + diff == len(string)
                or string[center - diff] != string[center + diff]):
            result = 0
        else:
            result = 1 + self.palindromic_length(center, diff + 1, string)
        
        self.memo[key] = result
        return result

    def palindromic_string(self):
        # Use list comprehension and join for string building
        transformed_chars = []
        for i, char in enumerate(self.input_string):
            if i > 0:
                transformed_chars.append('|')
            transformed_chars.append(char)
        
        new_input_string = ''.join(transformed_chars)
        
        palindrome_data = []
        for center in range(len(new_input_string)):
            length = self.palindromic_length(center, 1, new_input_string)
            palindrome_data.append((length, center))
        
        max_length, optimal_center = max(palindrome_data)
        
        start_pos = optimal_center - max_length
        end_pos = optimal_center + max_length + 1
        result_chars = [c for c in new_input_string[start_pos:end_pos] if c != '|']
        
        return ''.join(result_chars)
