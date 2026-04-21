import re
from decimal import Decimal


class ExpressionCalculator:
    def __init__(self):
        self.operators = {'+': 1, '-': 1, '*': 2, '/': 2, '%': 2}
        
    def calculate(self, expression):
        tokens = self._tokenize(self._normalize_expression(expression))
        postfix = self._to_postfix(tokens)
        return self._evaluate_postfix(postfix)
    
    def _normalize_expression(self, expression):
        expression = re.sub(r"\s+", "", expression)
        expression = re.sub(r"=$", "", expression)
        
        result = []
        for i, char in enumerate(expression):
            if char == '-' and (i == 0 or expression[i-1] in '+-*/(%eE'):
                result.append('~')
            else:
                result.append(char)
        
        normalized = ''.join(result)
        if normalized.startswith('~('):
            return '0-' + normalized[1:]
        return normalized
    
    def _tokenize(self, expression):
        tokens = []
        i = 0
        while i < len(expression):
            if expression[i] in '+-*/%()':
                tokens.append(expression[i])
                i += 1
            elif expression[i] == '~':
                j = i + 1
                while j < len(expression) and expression[j] not in '+-*/%()':
                    j += 1
                tokens.append('-' + expression[i+1:j])
                i = j
            else:
                j = i
                while j < len(expression) and expression[j] not in '+-*/%()~':
                    j += 1
                tokens.append(expression[i:j])
                i = j
        return tokens
    
    def _to_postfix(self, tokens):
        output = []
        operator_stack = []
        
        for token in tokens:
            if self._is_number(token):
                output.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                operator_stack.pop()  # Remove '('
            elif token in self.operators:
                while (operator_stack and 
                       operator_stack[-1] != '(' and
                       operator_stack[-1] in self.operators and
                       self._get_precedence(operator_stack[-1]) >= self._get_precedence(token)):
                    output.append(operator_stack.pop())
                operator_stack.append(token)
        
        while operator_stack:
            output.append(operator_stack.pop())
        
        return output
    
    def _evaluate_postfix(self, postfix):
        stack = []
        
        for token in postfix:
            if self._is_number(token):
                stack.append(Decimal(token))
            else:
                b = stack.pop()
                a = stack.pop()
                result = self._apply_operator(a, b, token)
                stack.append(result)
        
        return float(stack[0])
    
    def _is_number(self, token):
        return token not in '+-*/%()' 
    
    def _get_precedence(self, op):
        return self.operators.get(op if op != '%' else '/', 0)
    
    def _apply_operator(self, a, b, op):
        operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '%': lambda x, y: x % y
        }
        return operations[op](a, b)
