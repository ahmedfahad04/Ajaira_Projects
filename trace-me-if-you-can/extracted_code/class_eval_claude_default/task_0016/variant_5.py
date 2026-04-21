import re

class Calculator:
    def __init__(self):
        self.operators = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '^': lambda x, y: x ** y
        }

    def calculate(self, expression):
        # Convert to list of tokens using regex
        tokens = re.findall(r'\d+\.?\d*|[+\-*/^()]', expression)
        
        # Convert numbers to floats
        for i, token in enumerate(tokens):
            if re.match(r'\d+\.?\d*', token):
                tokens[i] = float(token)
        
        return self._shunting_yard_eval(tokens)

    def _shunting_yard_eval(self, tokens):
        output_queue = []
        operator_stack = []
        
        for token in tokens:
            if isinstance(token, float):
                output_queue.append(token)
            elif token in self.operators:
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       operator_stack[-1] in self.operators and
                       self._get_precedence(operator_stack[-1]) >= self._get_precedence(token)):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack:
                    operator_stack.pop()  # Remove the '('
        
        # Pop remaining operators
        while operator_stack:
            output_queue.append(operator_stack.pop())
        
        # Evaluate RPN expression
        eval_stack = []
        for token in output_queue:
            if isinstance(token, float):
                eval_stack.append(token)
            else:
                if len(eval_stack) >= 2:
                    b = eval_stack.pop()
                    a = eval_stack.pop()
                    eval_stack.append(self.operators[token](a, b))
        
        return eval_stack[0] if eval_stack else None

    def _get_precedence(self, op):
        precedence_map = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
        return precedence_map.get(op, 0)

    def precedence(self, operator):
        return self._get_precedence(operator)

    def apply_operator(self, operand_stack, operator_stack):
        operator = operator_stack.pop()
        operand2 = operand_stack.pop()
        operand1 = operand_stack.pop()
        result = self.operators[operator](operand1, operand2)
        operand_stack.append(result)
        return operand_stack, operator_stack
