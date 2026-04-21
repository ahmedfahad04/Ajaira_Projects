class BracketChecker:
    def __init__(self, input_str):
        self.stack = []
        self.open_brackets = "([{"
        self.close_brackets = ")]}"
        self.input = input_str

    def purify_input(self):
        self.input = ''.join(char for char in self.input if char in self.open_brackets + self.close_brackets)

    def is_balanced(self):
        self.purify_input()
        for char in self.input:
            if char in self.open_brackets:
                self.stack.append(char)
            else:
                if not self.stack or self.stack.pop() != self.close_brackets[self.open_brackets.index(char)]:
                    return False
        return not self.stack
