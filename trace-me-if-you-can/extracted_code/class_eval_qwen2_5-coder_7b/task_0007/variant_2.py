class BracketsValidator:
    def __init__(self, expression):
        self.stack = []
        self.brackets_map = {')': '(', '}': '{', ']': '['}
        self.expression = expression

    def sanitize_expression(self):
        self.expression = ''.join(c for c in self.expression if c in self.brackets_map.keys() or c in self.brackets_map.values())

    def validate_balance(self):
        self.sanitize_expression()
        for char in self.expression:
            if char in self.brackets_map.values():
                self.stack.append(char)
            else:
                if not self.stack or self.stack.pop() != self.brackets_map[char]:
                    return False
        return not self.stack
