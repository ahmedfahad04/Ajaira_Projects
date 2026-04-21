class ExpressionEvaluator:
    def __init__(self):
        self.operators = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y
        }

    def evaluate_expression(self, equation):
        num_buffer = ''
        op_stack = []
        val_stack = []

        for char in equation:
            if char.isdigit() or char == '.':
                num_buffer += char
            else:
                if num_buffer:
                    val_stack.append(float(num_buffer))
                    num_buffer = ''

                if char in '+-*/^':
                    while (op_stack and
                           op_stack[-1] != '(' and
                           self.rank_op(op_stack[-1]) >= self.rank_op(char)):
                        val_stack, op_stack = self.execute_operation(val_stack, op_stack)
                    op_stack.append(char)
                elif char == '(':
                    op_stack.append(char)
                elif char == ')':
                    while op_stack and op_stack[-1] != '(':
                        val_stack, op_stack = self.execute_operation(val_stack, op_stack)
                    op_stack.pop()

        if num_buffer:
            val_stack.append(float(num_buffer))

        while op_stack:
            val_stack, op_stack = self.execute_operation(val_stack, op_stack)

        return val_stack[-1] if val_stack else None

    def rank_op(self, operator):
        ranks = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3
        }
        return ranks.get(operator, 0)

    def execute_operation(self, val_stack, op_stack):
        operator = op_stack.pop()
        if operator == '^':
            operand2 = val_stack.pop()
            operand1 = val_stack.pop()
            result = self.operators[operator](operand1, operand2)
            val_stack.append(result)
        else:
            operand2 = val_stack.pop()
            operand1 = val_stack.pop()
            result = self.operators[operator](operand1, operand2)
            val_stack.append(result)
        return val_stack, op_stack
