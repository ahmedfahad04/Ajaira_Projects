class BalancedBrackets:
    def __init__(self, expr):
        self.expr = expr
        self.bracket_pairs = {"(": ")", "{": "}", "[": "]"}

    def check_balanced_brackets(self):
        stack = []
        filtered_expr = ''.join(c for c in self.expr if c in "(){}[]")
        
        for char in filtered_expr:
            if char in self.bracket_pairs:
                stack.append(char)
            elif stack and self.bracket_pairs[stack[-1]] == char:
                stack.pop()
            else:
                return False
        
        return len(stack) == 0
