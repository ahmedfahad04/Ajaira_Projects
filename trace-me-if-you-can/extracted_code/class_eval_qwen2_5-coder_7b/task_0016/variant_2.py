class ExpressionCalculator:
    def __init__(self):
        self.operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y
        }

    def compute(self, input_string):
        number_buffer = ''
        operator_stack = []
        value_stack = []

        for char in input_string:
            if char.isdigit() or char == '.':
                number_buffer += char
            else:
                if number_buffer:
                    value_stack.append(float(number_buffer))
                    number_buffer = ''

                if char in '+-*/^':
                    while (operator_stack and
                           operator_stack[-1] != '(' and
                           self.priority(operator_stack[-1]) >= self.priority(char)):
                        value_stack, operator_stack = self.execute_operation(value_stack, operator_stack)
                    operator_stack.append(char)
                elif char == '(':
                    operator_stack.append(char)
                elif char == ')':
                    while operator_stack and operator_stack[-1] != '(':
                        value_stack, operator_stack = self.execute_operation(value_stack, operator_stack)
                    operator_stack.pop()

        if number_buffer:
            value_stack.append(float(number_buffer))

        while operator_stack:
            value_stack, operator_stack = self.execute_operation(value_stack, operator_stack)

        return value_stack[-1] if value_stack else None

    def priority(self, operator):
        priorities = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3
        }
        return priorities.get(operator, 0)

    def execute_operation(self, value_stack, operator_stack):
        operator = operator_stack.pop()
        if operator == '^':
            operand2 = value_stack.pop()
            operand1 = value_stack.pop()
            result = self.operations[operator](operand1, operand2)
            value_stack.append(result)
        else:
            operand2 = value_stack.pop()
            operand1 = value_stack.pop()
            result = self.operations[operator](operand1, operand2)
            value_stack.append(result)
        return value_stack, operator_stack
