class Manacher:
    def __init__(self, input_string) -> None:
        self.input_string = input_string

    def palindromic_string(self):
        # Functional programming approach with reduce and lambda
        from functools import reduce
        
        # Transform string using reduce
        if not self.input_string:
            return ""
            
        new_input_string = reduce(
            lambda acc, x: acc + ('|' if acc else '') + x[1],
            enumerate(self.input_string),
            ""
        )

        def check_palindrome_iterative(center, string):
            diff = 1
            length = 0
            while True:
                if (center - diff == -1 or center + diff == len(string) or 
                    string[center - diff] != string[center + diff]):
                    break
                length += 1
                diff += 1
            return length

        # Use map to get all palindrome lengths
        centers_and_lengths = list(map(
            lambda i: (i, check_palindrome_iterative(i, new_input_string)),
            range(len(new_input_string))
        ))

        # Find maximum using key function
        best_center, max_length = max(centers_and_lengths, key=lambda x: x[1])
        
        # Extract palindrome using slice and filter
        substring = new_input_string[best_center - max_length:best_center + max_length + 1]
        return ''.join(filter(lambda c: c != '|', substring))
