class MathParser:
    def __init__(self):
        self.ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y
        }

    def parse(self, expression):
        number_buffer = ''
        operator_stack = []
        value_stack = []

        for char in expression:
            if char.isdigit() or char == '.':
                number_buffer += char
            else:
                if number_buffer:
                    value_stack.append(float(number_buffer))
                    number_buffer = ''

                if char in '+-*/^':
                    while (operator_stack and
                           operator_stack[-1] != '(' and
                           self.precedence_op(operator_stack[-1]) >= self.precedence_op(char)):
                        value_stack, operator_stack = self.apply_operation(value_stack, operator_stack)
                    operator_stack.append(char)
                elif char == '(':
                    operator_stack.append(char)
                elif char == ')':
                    while operator_stack and operator_stack[-1] != '(':
                        value_stack, operator_stack = self.apply_operation(value_stack, operator_stack)
                    operator_stack.pop()

        if number_buffer:
            value_stack.append(float(number_buffer))

        while operator_stack:
            value_stack, operator_stack = self.apply_operation(value_stack, operator_stack)

        return value_stack[-1] if value_stack else None

    def precedence_op(self, operator):
        precedence_map = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3
        }
        return precedence_map.get(operator, 0)

    def apply_operation(self, value_stack, operator_stack):
        operator = operator_stack.pop()
        if operator == '^':
            operand2 = value_stack.pop()
            operand1 = value_stack.pop()
            result = self.ops[operator](operand1, operand2)
            value_stack.append(result)
        else:
            operand2 = value_stack.pop()
            operand1 = value_stack.pop()
            result = self.ops[operator](operand1, operand2)
            value_stack.append(result)
        return value_stack, operator_stack
