class Manacher:
    def __init__(self, input_string) -> None:
        self.input_string = input_string

    def palindromic_string(self):
        # Iterative approach with explicit stack simulation
        def get_palindrome_length(center, string):
            stack = [(center, 1)]
            total_length = 0
            
            while stack:
                curr_center, diff = stack.pop()
                if (curr_center - diff == -1 or curr_center + diff == len(string) or 
                    string[curr_center - diff] != string[curr_center + diff]):
                    continue
                total_length += 1
                stack.append((curr_center, diff + 1))
            
            return total_length

        # Transform string with separators
        transformed = '|'.join(self.input_string)
        
        max_length = 0
        best_center = 0
        
        for center in range(len(transformed)):
            length = get_palindrome_length(center, transformed)
            if length > max_length:
                max_length = length
                best_center = center
        
        # Extract result without separators
        start_idx = best_center - max_length
        end_idx = best_center + max_length + 1
        return ''.join(c for c in transformed[start_idx:end_idx] if c != '|')
