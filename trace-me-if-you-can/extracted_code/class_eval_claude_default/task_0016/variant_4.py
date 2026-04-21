class Calculator:
    OPERATORS = {'+', '-', '*', '/', '^'}
    PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    
    def __init__(self):
        self.operators = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y
        }

    def calculate(self, expression):
        # State machine approach
        state = 'EXPECTING_OPERAND'
        operands = []
        operators = []
        current_number = []
        
        def flush_number():
            if current_number:
                operands.append(float(''.join(current_number)))
                current_number.clear()
                
        def reduce_stack():
            if len(operands) >= 2 and operators:
                op = operators.pop()
                right = operands.pop()
                left = operands.pop()
                operands.append(self.operators[op](left, right))

        for char in expression + ' ':  # Add sentinel to flush last number
            if char.isdigit() or char == '.':
                current_number.append(char)
                state = 'BUILDING_NUMBER'
            elif char in self.OPERATORS:
                flush_number()
                
                # Reduce operators with higher or equal precedence
                while (operators and 
                       operators[-1] != '(' and 
                       self.PRECEDENCE.get(operators[-1], 0) >= self.PRECEDENCE[char]):
                    reduce_stack()
                
                operators.append(char)
                state = 'EXPECTING_OPERAND'
            elif char == '(':
                flush_number()
                operators.append(char)
                state = 'EXPECTING_OPERAND'
            elif char == ')':
                flush_number()
                
                # Reduce until we find the matching '('
                while operators and operators[-1] != '(':
                    reduce_stack()
                
                if operators:
                    operators.pop()  # Remove the '('
                    
                state = 'EXPECTING_OPERATOR'
            else:  # Whitespace or sentinel
                flush_number()

        # Final reduction
        while operators:
            reduce_stack()

        return operands[0] if operands else None

    def precedence(self, operator):
        return self.PRECEDENCE.get(operator, 0)

    def apply_operator(self, operand_stack, operator_stack):
        operator = operator_stack.pop()
        operand2 = operand_stack.pop()
        operand1 = operand_stack.pop()
        result = self.operators[operator](operand1, operand2)
        operand_stack.append(result)
        return operand_stack, operator_stack
