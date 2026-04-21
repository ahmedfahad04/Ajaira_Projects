class BracketBalancer:
    def __init__(self, expr):
        self.stack = []
        self.opening_brackets = "([{"
        self.closing_brackets = ")]}"
        self.expression = expr

    def clean_expression(self):
        self.expression = ''.join(char for char in self.expression if char in self.opening_brackets + self.closing_brackets)

    def is_balanced(self):
        self.clean_expression()
        for char in self.expression:
            if char in self.opening_brackets:
                self.stack.append(char)
            else:
                if not self.stack or self.stack.pop() != self.closing_brackets[self.opening_brackets.index(char)]:
                    return False
        return not self.stack
