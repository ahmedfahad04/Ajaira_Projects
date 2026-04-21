class ParenthesesValidator:
    def __init__(self, input_string):
        self.bracket_pairs = {'(': ')', '{': '}', '[': ']'}
        self.stack = []
        self.input_string = input_string

    def clean_string(self):
        self.input_string = ''.join(c for c in self.input_string if c in self.bracket_pairs.keys() or c in self.bracket_pairs.values())

    def validate_brackets(self):
        self.clean_string()
        for char in self.input_string:
            if char in self.bracket_pairs:
                self.stack.append(char)
            else:
                if not self.stack or self.bracket_pairs[self.stack.pop()] != char:
                    return False
        return not self.stack
