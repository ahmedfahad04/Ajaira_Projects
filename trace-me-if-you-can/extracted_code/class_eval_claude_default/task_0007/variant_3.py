class BalancedBrackets:
    def __init__(self, expr):
        self.expr = expr
        self.opening = set('({[')
        self.closing = set(')}]')
        self.matches = dict(zip(')}]', '({['))

    def clear_expr(self):
        self.expr = ''.join(filter(lambda c: c in self.opening or c in self.closing, self.expr))

    def check_balanced_brackets(self):
        self.clear_expr()
        counter = []
        
        for bracket in self.expr:
            if bracket in self.opening:
                counter.append(bracket)
            else:
                try:
                    last_open = counter.pop()
                    if self.matches[bracket] != last_open:
                        return False
                except IndexError:
                    return False
        
        return len(counter) == 0
