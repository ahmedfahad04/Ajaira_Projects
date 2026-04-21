class Manacher:
    def __init__(self, input_string) -> None:
        self.input_string = input_string

    def palindromic_string(self):
        # Object-oriented approach with helper classes
        class PalindromeAnalyzer:
            def __init__(self, text):
                self.text = text
                
            def get_expansion_length(self, center):
                radius = 0
                while (center - radius - 1 >= 0 and 
                       center + radius + 1 < len(self.text) and
                       self.text[center - radius - 1] == self.text[center + radius + 1]):
                    radius += 1
                return radius

        class StringTransformer:
            @staticmethod
            def add_separators(s):
                if not s:
                    return ""
                result_parts = [s[0]]
                for char in s[1:]:
                    result_parts.extend(['|', char])
                return ''.join(result_parts)
            
            @staticmethod
            def remove_separators(s):
                return s.replace('|', '')

        # Transform and analyze
        transformer = StringTransformer()
        transformed_string = transformer.add_separators(self.input_string)
        analyzer = PalindromeAnalyzer(transformed_string)
        
        max_radius = 0
        best_center = 0
        
        for center in range(len(transformed_string)):
            radius = analyzer.get_expansion_length(center)
            if radius > max_radius:
                max_radius = radius
                best_center = center
        
        # Extract final result
        start = best_center - max_radius
        end = best_center + max_radius + 1
        raw_palindrome = transformed_string[start:end]
        return transformer.remove_separators(raw_palindrome)
