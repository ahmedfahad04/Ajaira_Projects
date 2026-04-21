import re
from decimal import Decimal


class ExpressionCalculator:
    def __init__(self):
        pass
    
    def calculate(self, expression):
        return self._recursive_parse(self._preprocess(expression), 0)[0]
    
    def _preprocess(self, expression):
        expression = re.sub(r"\s+", "", expression)
        expression = re.sub(r"=$", "", expression)
        
        # Handle unary minus
        chars = list(expression)
        for i in range(len(chars)):
            if chars[i] == '-':
                if i == 0 or chars[i-1] in '+-*/(%eE':
                    chars[i] = '~'
        
        result = ''.join(chars)
        if result.startswith('~('):
            result = '0-' + result[1:]
        
        return result
    
    def _recursive_parse(self, expr, pos):
        result, pos = self._parse_term(expr, pos)
        
        while pos < len(expr) and expr[pos] in '+-':
            op = expr[pos]
            pos += 1
            right, pos = self._parse_term(expr, pos)
            
            if op == '+':
                result = float(Decimal(str(result)) + Decimal(str(right)))
            else:
                result = float(Decimal(str(result)) - Decimal(str(right)))
        
        return result, pos
    
    def _parse_term(self, expr, pos):
        result, pos = self._parse_factor(expr, pos)
        
        while pos < len(expr) and expr[pos] in '*/%':
            op = expr[pos]
            pos += 1
            right, pos = self._parse_factor(expr, pos)
            
            if op == '*':
                result = float(Decimal(str(result)) * Decimal(str(right)))
            elif op == '/':
                result = float(Decimal(str(result)) / Decimal(str(right)))
            else:  # %
                result = float(Decimal(str(result)) % Decimal(str(right)))
        
        return result, pos
    
    def _parse_factor(self, expr, pos):
        if pos < len(expr) and expr[pos] == '(':
            pos += 1  # Skip '('
            result, pos = self._recursive_parse(expr, pos)
            pos += 1  # Skip ')'
            return result, pos
        
        # Parse number (including unary minus)
        start = pos
        if pos < len(expr) and expr[pos] == '~':
            pos += 1
        
        while pos < len(expr) and (expr[pos].isdigit() or expr[pos] == '.'):
            pos += 1
        
        number_str = expr[start:pos].replace('~', '-')
        return float(number_str), pos
