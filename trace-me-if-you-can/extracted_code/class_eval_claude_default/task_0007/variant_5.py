class BalancedBrackets:
    def __init__(self, expr):
        self.expr = expr
        self.bracket_stack = []

    def clear_expr(self):
        valid_chars = {'(', ')', '{', '}', '[', ']'}
        self.expr = ''.join(char for char in self.expr if char in valid_chars)

    def _is_matching_pair(self, open_bracket, close_bracket):
        pairs = [('(', ')'), ('{', '}'), ('[', ']')]
        return (open_bracket, close_bracket) in pairs

    def check_balanced_brackets(self):
        self.clear_expr()
        self.bracket_stack.clear()
        
        for current_char in self.expr:
            if current_char in '({[':
                self.bracket_stack.append(current_char)
            elif current_char in ')}]':
                if not self.bracket_stack:
                    return False
                last_open = self.bracket_stack.pop()
                if not self._is_matching_pair(last_open, current_char):
                    return False
        
        return not bool(self.bracket_stack)
