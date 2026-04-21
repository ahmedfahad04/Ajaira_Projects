import re
from decimal import Decimal
from functools import reduce
import operator


class ExpressionCalculator:
    def __init__(self):
        self.ops = {
            '+': (1, operator.add),
            '-': (1, operator.sub), 
            '*': (2, operator.mul),
            '/': (2, operator.truediv),
            '%': (2, operator.mod)
        }
    
    def calculate(self, expression):
        cleaned = self._clean_expression(expression)
        ast = self._build_ast(cleaned)
        return self._evaluate_ast(ast)
    
    def _clean_expression(self, expr):
        expr = re.sub(r"\s+", "", expr)
        expr = re.sub(r"=$", "", expr)
        
        # Convert unary minus to binary subtraction from zero
        result = []
        i = 0
        while i < len(expr):
            if expr[i] == '-' and (i == 0 or expr[i-1] in '+-*/(%eE'):
                if i == 0 or expr[i-1] != '(':
                    result.append('(0-')
                    i += 1
                    paren_count = 1
                    while i < len(expr) and paren_count > 0:
                        if expr[i] == '(':
                            paren_count += 1
                        elif expr[i] == ')':
                            paren_count -= 1
                        elif expr[i] in '+-*/%' and paren_count == 1:
                            break
                        result.append(expr[i])
                        i += 1
                    result.append(')')
                else:
                    result.append(expr[i])
                    i += 1
            else:
                result.append(expr[i])
                i += 1
        
        return ''.join(result)
    
    def _build_ast(self, expr):
        tokens = self._tokenize(expr)
        return self._parse_expression(tokens, 0)[0]
    
    def _tokenize(self, expr):
        tokens = []
        i = 0
        while i < len(expr):
            if expr[i] in '+-*/%()':
                tokens.append(expr[i])
                i += 1
            else:
                start = i
                while i < len(expr) and expr[i] not in '+-*/%()':
                    i += 1
                tokens.append(('NUMBER', expr[start:i]))
        return tokens
    
    def _parse_expression(self, tokens, min_prec):
        left, pos = self._parse_primary(tokens, 0)
        
        while pos < len(tokens):
            if tokens[pos] not in self.ops:
                break
            
            op = tokens[pos]
            prec, _ = self.ops[op]
            
            if prec < min_prec:
                break
                
            pos += 1
            right, pos = self._parse_expression(tokens, prec + 1)
            left = ('BINOP', op, left, right)
        
        return left, pos
    
    def _parse_primary(self, tokens, pos):
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
            
        token = tokens[pos]
        
        if isinstance(token, tuple) and token[0] == 'NUMBER':
            return token, pos + 1
        elif token == '(':
            pos += 1
            result, pos = self._parse_expression(tokens, 0)
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError("Missing closing parenthesis")
            return result, pos + 1
        else:
            raise ValueError(f"Unexpected token: {token}")
    
    def _evaluate_ast(self, ast):
        if isinstance(ast, tuple):
            if ast[0] == 'NUMBER':
                return float(ast[1])
            elif ast[0] == 'BINOP':
                _, op, left, right = ast
                left_val = Decimal(str(self._evaluate_ast(left)))
                right_val = Decimal(str(self._evaluate_ast(right)))
                _, func = self.ops[op]
                return float(func(left_val, right_val))
        
        raise ValueError(f"Invalid AST node: {ast}")
