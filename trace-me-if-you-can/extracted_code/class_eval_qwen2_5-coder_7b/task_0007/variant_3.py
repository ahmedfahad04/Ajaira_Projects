class ParenthesesChecker:
    def __init__(self, equation):
        self.stack = []
        self.open_bracket = "([{"
        self.close_bracket = ")]}"
        self.equation = equation

    def remove_unnecessary_chars(self):
        self.equation = ''.join(char for char in self.equation if char in self.open_bracket + self.close_bracket)

    def check_balance(self):
        self.remove_unnecessary_chars()
        for char in self.equation:
            if char in self.open_bracket:
                self.stack.append(char)
            else:
                if not self.stack or self.stack.pop() != self.close_bracket[self.open_bracket.index(char)]:
                    return False
        return not self.stack
