import re
from decimal import Decimal


class ExpressionCalculator:
    def __init__(self):
        self.precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '%': 2, '(': 0}
        
    def calculate(self, expression):
        tokens = list(self._generate_tokens(self._sanitize(expression)))
        return self._evaluate_expression(tokens)
    
    def _sanitize(self, expression):
        expression = re.sub(r"\s+", "", expression)
        expression = re.sub(r"=$", "", expression)
        
        # Handle unary minus by inserting zero
        sanitized = []
        for i, char in enumerate(expression):
            if char == '-' and (i == 0 or expression[i-1] in '+-*/(%eE'):
                if not (i > 0 and expression[i-1] == '('):
                    sanitized.append('0')
                sanitized.append('-')
            else:
                sanitized.append(char)
        
        return ''.join(sanitized)
    
    def _generate_tokens(self, expression):
        i = 0
        while i < len(expression):
            if expression[i] in '+-*/%()':
                yield expression[i]
                i += 1
            else:
                start = i
                while i < len(expression) and expression[i] not in '+-*/%()':
                    i += 1
                yield expression[start:i]
    
    def _evaluate_expression(self, tokens):
        values = []
        operators = []
        
        def apply_operation():
            if len(values) < 2 or not operators:
                return
            
            right = values.pop()
            left = values.pop()
            op = operators.pop()
            
            result = self._compute(left, right, op)
            values.append(result)
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if self._is_operand(token):
                values.append(Decimal(token))
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    apply_operation()
                operators.pop()  # Remove '('
            elif token in self.precedence:
                while (operators and 
                       operators[-1] != '(' and 
                       self.precedence[operators[-1]] >= self.precedence[token]):
                    apply_operation()
                operators.append(token)
            
            i += 1
        
        while operators:
            apply_operation()
        
        return float(values[0]) if values else 0.0
    
    def _is_operand(self, token):
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def _compute(self, left, right, operator):
        operations = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
            '%': lambda a, b: a % b
        }
        return operations[operator](left, right)
