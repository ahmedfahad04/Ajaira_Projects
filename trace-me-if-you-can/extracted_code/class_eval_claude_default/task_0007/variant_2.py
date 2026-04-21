class BalancedBrackets:
    def __init__(self, expr):
        self.expr = expr

    def check_balanced_brackets(self):
        bracket_map = {')': '(', '}': '{', ']': '['}
        stack = []
        
        for char in self.expr:
            if char in '({[':
                stack.append(char)
            elif char in ')}]':
                if not stack or stack.pop() != bracket_map[char]:
                    return False
        
        return not stack

    def clear_expr(self):
        # Maintained for interface compatibility but not used
        pass
