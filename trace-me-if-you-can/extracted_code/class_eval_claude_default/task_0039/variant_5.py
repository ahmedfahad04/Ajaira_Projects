import re
from decimal import Decimal


class ExpressionCalculator:
    def __init__(self):
        self.operators = {'+': 1, '-': 1, '*': 2, '/': 2, '%': 2}
        
    def calculate(self, expression):
        tokens = self._tokenize(self._normalize(expression))
        postfix = self._to_postfix(tokens)
        return self._evaluate_postfix(postfix)
    
    def _normalize(self, expression):
        expression = re.sub(r"\s+", "", expression)
        expression = re.sub(r"=$", "", expression)
        
        # Handle unary minus
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
                # Collect unary minus with number
                j = i + 1
                while j < len(expression) and expression[j] not in '+-*/%()':
                    j += 1
                tokens.append(expression[i:j])
                i = j
            else:
                # Collect number
                j = i
                while j < len(expression) and expression[j] not in '+-*/%()':
                    j += 1
                tokens.append(expression[i:j])
                i = j
        return tokens
    
    def _to_postfix(self, tokens):
        output = []
        stack = []
        
        for token in tokens:
            if self._is_number(token):
                output.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                stack.pop()  # Remove '('
            elif token in self.operators:
                while (stack and stack[-1] != '(' and 
                       stack[-1] in self.operators and
                       self.operators.get(stack[-1], 0) >= self.operators[token]):
                    output.append(stack.pop())
                stack.append(token)
        
        while stack:
            output.append(stack.pop())
        
        return output
    
    def _is_number(self, token):
        return token not in '+-*/%()' or token.startswith('~')
    
    def _evaluate_postfix(self, postfix):
        stack = []
        
        for token in postfix:
            if self._is_number(token):
                value = token.replace('~', '-')
                stack.append(Decimal(value))
            else:
                right = stack.pop()
                left = stack.pop()
                
                if token == '+':
                    result = left + right
                elif token == '-':
                    result = left - right
                elif token == '*':
                    result = left * right
                elif token == '/':
                    result = left / right
                elif token == '%':
                    result = left % right
                
                stack.append(result)
        
        return float(stack[0])
