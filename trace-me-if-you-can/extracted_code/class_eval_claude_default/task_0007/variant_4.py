class BalancedBrackets:
    def __init__(self, expr):
        self.expr = expr

    def clear_expr(self):
        bracket_chars = "(){}[]"
        self.expr = ''.join(c for c in self.expr if c in bracket_chars)

    def check_balanced_brackets(self):
        self.clear_expr()
        balance_count = {'(': 0, '{': 0, '[': 0}
        
        for char in self.expr:
            if char == '(':
                balance_count['('] += 1
            elif char == ')':
                balance_count['('] -= 1
                if balance_count['('] < 0:
                    return False
            elif char == '{':
                balance_count['{'] += 1
            elif char == '}':
                balance_count['{'] -= 1
                if balance_count['{'] < 0:
                    return False
            elif char == '[':
                balance_count['['] += 1
            elif char == ']':
                balance_count['['] -= 1
                if balance_count['['] < 0:
                    return False
        
        return all(count == 0 for count in balance_count.values())
