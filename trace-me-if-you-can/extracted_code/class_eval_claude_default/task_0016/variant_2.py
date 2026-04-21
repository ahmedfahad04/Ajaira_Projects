class Calculator:
    def __init__(self):
        self.operations = {
            '+': self._add,
            '-': self._subtract,
            '*': self._multiply,
            '/': self._divide,
            '^': self._power
        }

    def _add(self, a, b): return a + b
    def _subtract(self, a, b): return a - b
    def _multiply(self, a, b): return a * b
    def _divide(self, a, b): return a / b
    def _power(self, a, b): return a ** b

    def calculate(self, expression):
        return self._evaluate_recursive(expression, 0)[0]

    def _evaluate_recursive(self, expr, pos):
        def parse_number(start_pos):
            end_pos = start_pos
            while end_pos < len(expr) and (expr[end_pos].isdigit() or expr[end_pos] == '.'):
                end_pos += 1
            return float(expr[start_pos:end_pos]), end_pos

        def parse_factor(curr_pos):
            if expr[curr_pos] == '(':
                result, new_pos = self._evaluate_recursive(expr, curr_pos + 1)
                return result, new_pos + 1  # skip ')'
            else:
                return parse_number(curr_pos)

        def parse_power(curr_pos):
            left, curr_pos = parse_factor(curr_pos)
            while curr_pos < len(expr) and expr[curr_pos] == '^':
                right, curr_pos = parse_factor(curr_pos + 1)
                left = self._power(left, right)
            return left, curr_pos

        def parse_term(curr_pos):
            left, curr_pos = parse_power(curr_pos)
            while curr_pos < len(expr) and expr[curr_pos] in '*/':
                op = expr[curr_pos]
                right, curr_pos = parse_power(curr_pos + 1)
                left = self.operations[op](left, right)
            return left, curr_pos

        def parse_expression(curr_pos):
            left, curr_pos = parse_term(curr_pos)
            while curr_pos < len(expr) and expr[curr_pos] in '+-':
                op = expr[curr_pos]
                right, curr_pos = parse_term(curr_pos + 1)
                left = self.operations[op](left, right)
            return left, curr_pos

        return parse_expression(pos)

    def precedence(self, operator):
        return {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}.get(operator, 0)

    def apply_operator(self, operand_stack, operator_stack):
        operator = operator_stack.pop()
        operand2 = operand_stack.pop()
        operand1 = operand_stack.pop()
        result = self.operations[operator](operand1, operand2)
        operand_stack.append(result)
        return operand_stack, operator_stack
